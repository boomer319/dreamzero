"""
Merge multiple LeRobot datasets (HF v3.0 or local v2/v3) into a single v2 dataset.

End-to-end pipeline:
  1. Downloads each HF dataset via snapshot_download (if not already local)
  2. Converts v3 → v2 (splits consolidated parquet/videos into per-episode files)
  3. Merges all episodes with contiguous episode_index
  4. Injects task descriptions via --task-names (one per input dataset)
  5. Writes meta/tasks.jsonl, meta/episodes.jsonl, meta/info.json
  6. Output is ready for convert_lerobot_to_gear.py

Usage:
  python scripts/data/merge_lerobot_hf_datasets.py \\
      --datasets \\
          unitreerobotics/G1_Dex3_BlockStacking_Dataset \\
          unitreerobotics/G1_Dex3_CameraPackaging_Dataset \\
          unitreerobotics/G1_Dex3_GraspSquare_Dataset \\
          unitreerobotics/G1_Dex3_ObjectPlacement_Dataset \\
          unitreerobotics/G1_Dex3_ToastedBread_Dataset \\
      --task-names block_stacking camera_packaging grasp_square object_placement toasted_bread \\
      --output ./datasets/G1_Dex3_AllMerged

  # Merge existing local v2 datasets:
  python scripts/data/merge_lerobot_hf_datasets.py \\
      --datasets ./local/ds1 ./local/ds2 \\
      --task-names task_a task_b \\
      --output ./datasets/merged
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirroring lerobot v2 conventions)
# ---------------------------------------------------------------------------

V21 = "v2.1"
V30 = "v3.0"

DEFAULT_CHUNK_SIZE = 1000

LEGACY_DATA_PATH_TEMPLATE = (
    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
)
LEGACY_VIDEO_PATH_TEMPLATE = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)

DEFAULT_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
DEFAULT_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
EPISODES_DIR = "meta/episodes"

MIN_VIDEO_DURATION = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    return value


def load_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"meta/info.json not found at {info_path}")
    with open(info_path) as f:
        return json.load(f)


def write_info(info: dict, root: Path) -> None:
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)


def get_version(root: Path) -> str:
    info = load_info(root)
    return info.get("codebase_version", "v2.1")


# ---------------------------------------------------------------------------
# v3 → v2 conversion (self-contained, no lerobot dependency)
# ---------------------------------------------------------------------------

def load_episode_records(root: Path) -> list[dict[str, Any]]:
    episodes_dir = root / EPISODES_DIR
    pq_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not pq_paths:
        raise FileNotFoundError(f"No episode parquet files found in {episodes_dir}.")
    records: list[dict[str, Any]] = []
    for pq_path in pq_paths:
        table = pq.read_table(pq_path)
        records.extend(table.to_pylist())
    records.sort(key=lambda rec: int(rec["episode_index"]))
    return records


def convert_tasks_v3_to_v2(root: Path, new_root: Path) -> None:
    """Convert meta/tasks.parquet (v3) → meta/tasks.jsonl (v2)."""
    tasks_parquet = root / "meta" / "tasks.parquet"
    if not tasks_parquet.exists():
        log.info("  No tasks.parquet found (v3), will create tasks.jsonl later")
        return
    tasks_df = pd.read_parquet(tasks_parquet)
    out_dir = new_root / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tasks.jsonl"
    with open(out_path, "w") as f:
        for _, row in tasks_df.iterrows():
            record = {
                "task_index": int(row.get("task_index", 0)),
                "task": row.get("task", ""),
            }
            if isinstance(record["task"], (list, np.ndarray)):
                record["task"] = str(record["task"][0]) if len(record["task"]) > 0 else ""
            f.write(json.dumps(record) + "\n")
    log.info("  Wrote tasks.jsonl from v3 tasks.parquet")


def convert_info_v3_to_v2(
    root: Path,
    new_root: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],
) -> None:
    info = load_info(root)
    total_episodes = len(episode_records)
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    info["codebase_version"] = V21
    info["data_path"] = LEGACY_DATA_PATH_TEMPLATE
    if info.get("video_path") is not None and len(video_keys) > 0:
        info["video_path"] = LEGACY_VIDEO_PATH_TEMPLATE
    else:
        info["video_path"] = None
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)

    for key, ft in info["features"].items():
        if ft.get("dtype") != "video":
            ft.pop("fps", None)

    info["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes > 0 else 0
    info["total_videos"] = total_episodes * len(video_keys)

    write_info(info, new_root)
    log.info("  Wrote v2.1 info.json")


def _group_episodes_by_data_file(
    episode_records: Iterable[dict[str, Any]],
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in episode_records:
        key = (int(record["data/chunk_index"]), int(record["data/file_index"]))
        grouped[key].append(record)
    return grouped


def convert_data_v3_to_v2(root: Path, new_root: Path, episode_records: list[dict[str, Any]], chunks_size: int) -> None:
    grouped = _group_episodes_by_data_file(episode_records)
    for (chunk_idx, file_idx), records in tqdm(grouped.items(), desc="  split parquet"):
        source_path = root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        if not source_path.exists():
            raise FileNotFoundError(f"Expected source parquet file not found: {source_path}")
        table = pq.read_table(source_path)
        records = sorted(records, key=lambda rec: int(rec["dataset_from_index"]))
        file_offset = int(records[0]["dataset_from_index"])
        for record in records:
            episode_index = int(record["episode_index"])
            start = int(record["dataset_from_index"]) - file_offset
            stop = int(record["dataset_to_index"]) - file_offset
            length = stop - start
            if length <= 0:
                raise ValueError(f"Invalid episode length: ep={episode_index}, len={length}")
            episode_table = table.slice(start, length)
            dest_chunk = episode_index // chunks_size
            dest_path = new_root / LEGACY_DATA_PATH_TEMPLATE.format(
                episode_chunk=dest_chunk, episode_index=episode_index,
            )
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(episode_table, dest_path)


def _group_episodes_by_video_file(
    episode_records: Iterable[dict[str, Any]], video_key: str,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    for record in episode_records:
        if chunk_col not in record or file_col not in record:
            continue
        chunk_idx, file_idx = record.get(chunk_col), record.get(file_col)
        if chunk_idx is None or file_idx is None:
            continue
        grouped[(int(chunk_idx), int(file_idx))].append(record)
    return grouped


def _extract_video_segment(src: Path, dst: Path, start: float, end: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(end - start, MIN_VIDEO_DURATION)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.6f}", "-i", str(src),
        "-t", f"{duration:.6f}", "-c", "copy",
        "-avoid_negative_ts", "1", "-y", str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300, capture_output=True, text=True)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out: {src} -> {dst}") from e
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found; required for video splitting") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed: {src} -> {dst}. Error: {e.stderr.strip()}"
        ) from e


def convert_videos_v3_to_v2(root: Path, new_root: Path, episode_records: list[dict[str, Any]], video_keys: list[str], chunks_size: int) -> None:
    if not video_keys:
        return
    for video_key in video_keys:
        grouped = _group_episodes_by_video_file(episode_records, video_key)
        if not grouped:
            log.info("  No video metadata for '%s', skipping", video_key)
            continue
        for (chunk_idx, file_idx), records in tqdm(grouped.items(), desc=f"  split videos ({video_key})"):
            src_path = root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx,
            )
            if not src_path.exists():
                raise FileNotFoundError(f"Expected video not found: {src_path}")
            records = sorted(records, key=lambda rec: float(rec[f"videos/{video_key}/from_timestamp"]))
            for record in records:
                episode_index = int(record["episode_index"])
                start = float(record[f"videos/{video_key}/from_timestamp"])
                end = float(record[f"videos/{video_key}/to_timestamp"])
                dest_chunk = episode_index // chunks_size
                dest_path = new_root / LEGACY_VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=dest_chunk, video_key=video_key, episode_index=episode_index,
                )
                _extract_video_segment(src_path, dest_path, start=start, end=end)


def convert_episodes_metadata_v3_to_v2(new_root: Path, episode_records: list[dict[str, Any]]) -> None:
    episodes_path = new_root / "meta" / "episodes.jsonl"
    stats_path = new_root / "meta" / "episodes_stats.jsonl"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)

    with open(episodes_path, "w") as ew, open(stats_path, "w") as sw:
        for record in sorted(episode_records, key=lambda rec: int(rec["episode_index"])):
            legacy = {
                key: value for key, value in record.items()
                if not key.startswith("data/")
                and not key.startswith("videos/")
                and not key.startswith("stats/")
                and not key.startswith("meta/")
                and key not in {"dataset_from_index", "dataset_to_index"}
            }
            if "length" not in legacy:
                if "dataset_from_index" in record and "dataset_to_index" in record:
                    legacy["length"] = int(record["dataset_to_index"]) - int(record["dataset_from_index"])
            serializable = {key: _to_serializable(value) for key, value in legacy.items()}
            ew.write(json.dumps(serializable) + "\n")

            stats_flat = {key: record[key] for key in record if key.startswith("stats/")}
            sw.write(json.dumps({
                "episode_index": int(record["episode_index"]),
                "stats": _to_serializable(stats_flat),
            }) + "\n")

    log.info("  Wrote episodes.jsonl and episodes_stats.jsonl")


def convert_v3_to_v2(root: Path) -> Path:
    """Convert a v3 dataset at `root` to v2 format. Returns path to the new v2 dataset."""
    info = load_info(root)
    version = info.get("codebase_version", "unknown")
    if version != V30:
        log.info("  Already v2 (version=%s), no conversion needed", version)
        return root

    episode_records = load_episode_records(root)
    video_keys = [key for key, ft in info["features"].items() if ft.get("dtype") == "video"]
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)

    # Create a staging directory for the v2 conversion
    v2_root = root.parent / f"{root.name}_v2_staging"
    if v2_root.exists():
        shutil.rmtree(v2_root)
    v2_root.mkdir(parents=True, exist_ok=True)

    log.info("  Converting v3 → v2 (%d episodes, %d video keys)", len(episode_records), len(video_keys))

    convert_info_v3_to_v2(root, v2_root, episode_records, video_keys)
    convert_tasks_v3_to_v2(root, v2_root)
    convert_data_v3_to_v2(root, v2_root, episode_records, chunks_size)
    convert_videos_v3_to_v2(root, v2_root, episode_records, video_keys, chunks_size)
    convert_episodes_metadata_v3_to_v2(v2_root, episode_records)

    # Copy any ancillary directories (images, etc.)
    for subdir in ["images"]:
        src = root / subdir
        if src.exists():
            shutil.copytree(src, v2_root / subdir, dirs_exist_ok=True)

    return v2_root


def copy_global_stats(root: Path, new_root: Path) -> None:
    src = root / "meta" / "stats.json"
    if src.exists():
        dst = new_root / "meta" / "stats.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Download + prepare each dataset
# ---------------------------------------------------------------------------

HF_CACHE = Path.home() / ".cache" / "huggingface" / "datasets"


def download_and_prepare(ds_name: str, force: bool) -> Path:
    """Download a HF dataset (or validate local path), convert to v2 if needed, return v2 root."""
    p = Path(ds_name)

    if p.exists() and (p / "meta" / "info.json").exists():
        # Local dataset
        log.info("Using local dataset: %s", p.resolve())
        root = p.resolve()
        version = get_version(root)
        if version == V30:
            log.info("Local dataset is v3.0, converting to v2...")
            root = convert_v3_to_v2(root)
        return root

    # Download from HF
    log.info("Downloading: %s", ds_name)
    local_dir = HF_CACHE / ds_name.replace("/", "_")
    if local_dir.exists() and force:
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(ds_name, repo_type="dataset", local_dir=local_dir)

    # Check version and convert if needed
    info = load_info(local_dir)
    version = info.get("codebase_version", "unknown")
    if version == V30:
        log.info("Downloaded dataset is v3.0, converting to v2...")
        root = convert_v3_to_v2(local_dir)
    else:
        root = local_dir

    return root


# ---------------------------------------------------------------------------
# Merge v2 datasets
# ---------------------------------------------------------------------------

def merge_datasets(v2_roots: list[Path], task_names: list[str] | None, output_path: Path) -> None:
    data_dir = output_path / "data"
    videos_dir = output_path / "videos"
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_frames = 0
    episode_offset = 0
    merged_info = None
    task_index_offset = 0
    all_tasks: list[dict] = []
    all_episodes: list[dict] = []

    for ds_idx, root in enumerate(v2_roots):
        info = load_info(root)
        if merged_info is None:
            merged_info = info.copy()
        else:
            for key in ("fps", "robot_type"):
                if key in merged_info and key in info and merged_info[key] != info[key]:
                    log.warning(
                        "Mismatched '%s': %s vs %s. Using first dataset's value.",
                        key, merged_info[key], info[key],
                    )

        parquet_paths = sorted(root.glob("data/**/episode_*.parquet"))
        if not parquet_paths:
            log.warning("  No parquet files in %s, skipping", root)
            continue

        log.info("Merging dataset %d/%d: %s (%d episodes)",
                 ds_idx + 1, len(v2_roots), root.name, info.get("total_episodes", "?"))

        dataset_episode_offset = episode_offset

        # Determine task info for this dataset
        task_name = None
        if task_names and ds_idx < len(task_names):
            task_name = task_names[ds_idx]
        elif task_names:
            task_name = task_names[-1]

        # Load existing tasks from source
        tasks_path = root / "meta" / "tasks.jsonl"
        source_tasks: list[dict] = []
        if tasks_path.exists():
            with open(tasks_path) as f:
                for line in f:
                    source_tasks.append(json.loads(line.strip()))

        # Map task_index from source → new task_index
        task_mapping: dict[int, int] = {}
        if task_name:
            # Inject a single task name for all episodes
            existing = next((t for t in all_tasks if t["task"] == task_name), None)
            if existing:
                new_task_idx = existing["task_index"]
            else:
                new_task_idx = len(all_tasks)
                all_tasks.append({"task_index": new_task_idx, "task": task_name})
            for src_task in source_tasks:
                task_mapping[int(src_task["task_index"])] = new_task_idx
        else:
            for src_task in source_tasks:
                task_text = src_task.get("task", "")
                if not task_text:
                    task_text = f"task_{ds_idx}"
                existing = next((t for t in all_tasks if t["task"] == task_text), None)
                if existing:
                    new_task_idx = existing["task_index"]
                else:
                    new_task_idx = len(all_tasks)
                    all_tasks.append({"task_index": new_task_idx, "task": task_text})
                task_mapping[int(src_task["task_index"])] = new_task_idx

        # Load episode metadata
        episodes_path = root / "meta" / "episodes.jsonl"
        source_episodes: list[dict] = []
        if episodes_path.exists():
            with open(episodes_path) as f:
                for line in f:
                    source_episodes.append(json.loads(line.strip()))

        # Build a lookup: episode_index → task_index from source
        orig_ep_to_task: dict[int, int] = {}
        for ep in source_episodes:
            ep_idx = int(ep["episode_index"])
            tasks = ep.get("tasks", [])
            if not tasks and "task_index" in ep:
                orig_ep_to_task[ep_idx] = int(ep["task_index"])
            elif tasks:
                orig_ep_to_task[ep_idx] = 0

        # Also try to read task_index from the source info.json features
        # (the parquet data itself may have a task_index column)

        for src_pp in tqdm(parquet_paths, desc=f"  merging {root.name}"):
            df = pd.read_parquet(src_pp)
            if len(df) == 0:
                log.warning("  Skipping empty parquet: %s", src_pp)
                continue
            orig_ep = int(df["episode_index"].iloc[0])
            length = len(df)

            # Assign new episode index
            df["episode_index"] = episode_offset
            # Assign new task_index
            orig_task = orig_ep_to_task.get(orig_ep, 0)
            df["task_index"] = task_mapping.get(orig_task, 0)

            # Write per-episode parquet
            chunk_idx = episode_offset // 1000
            chunk_dir = data_dir / f"chunk-{chunk_idx:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"episode_{episode_offset:06d}.parquet"
            df.to_parquet(chunk_dir / out_name, index=False)

            # Record episode metadata
            all_episodes.append({
                "episode_index": episode_offset,
                "task_index": int(df["task_index"].iloc[0]),
                "tasks": [task_name or ""],
                "length": length,
            })

            total_frames += length
            episode_offset += 1
            total_episodes += 1

        # Copy videos
        src_videos = root / "videos"
        if src_videos.exists():
            for video_pp in tqdm(list(src_videos.rglob("*.mp4")), desc=f"  videos {root.name}", leave=False):
                rel = video_pp.relative_to(root)
                # Compute new chunk dir based on episode_index
                # Parse original episode index from filename
                parts = str(rel).split("/")
                # path: videos/chunk-{xxx}/{camera_key}/episode_{idx}.mp4
                if len(parts) >= 4 and parts[-1].startswith("episode_"):
                    orig_ep = int(parts[-1].replace("episode_", "").replace(".mp4", ""))
                    if orig_ep in orig_ep_to_task:
                        new_ep = orig_ep + dataset_episode_offset
                    else:
                        new_ep = orig_ep + dataset_episode_offset
                    camera_key = parts[-2]
                    new_chunk = new_ep // 1000
                    dst = videos_dir / f"chunk-{new_chunk:03d}" / camera_key / f"episode_{new_ep:06d}.mp4"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(video_pp, dst)

    # Write tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for t in all_tasks:
            f.write(json.dumps(t) + "\n")
    log.info("Wrote tasks.jsonl (%d tasks)", len(all_tasks))

    # Write episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in all_episodes:
            f.write(json.dumps(ep) + "\n")
    log.info("Wrote episodes.jsonl (%d episodes)", len(all_episodes))

    # Write info.json
    if merged_info is None:
        merged_info = {
            "codebase_version": V21,
            "fps": 50,
            "robot_type": "unitree_g1",
        }

    merged_info["codebase_version"] = V21
    merged_info["total_episodes"] = total_episodes
    merged_info["total_frames"] = total_frames
    merged_info["total_tasks"] = len(all_tasks)
    merged_info["total_chunks"] = math.ceil(total_episodes / DEFAULT_CHUNK_SIZE) if total_episodes > 0 else 0
    merged_info["data_path"] = LEGACY_DATA_PATH_TEMPLATE

    # Detect video keys from first dataset with videos
    for root in v2_roots:
        vinfo = load_info(root)
        vid_keys = [k for k, ft in vinfo.get("features", {}).items() if ft.get("dtype") == "video"]
        if vid_keys:
            merged_info["video_path"] = LEGACY_VIDEO_PATH_TEMPLATE
            break
    if "video_path" not in merged_info:
        merged_info["video_path"] = None

    merged_info.pop("data_files_size_in_mb", None)
    merged_info.pop("video_files_size_in_mb", None)
    merged_info.pop("splits", None)

    write_info(merged_info, output_path)
    log.info("Wrote info.json (%d episodes, %d frames, %d tasks)",
             total_episodes, total_frames, len(all_tasks))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download, convert (v3→v2), merge, and annotate LeRobot datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--datasets", type=str, nargs="+", required=True,
                        help="HF repo IDs or local paths to merge.")
    parser.add_argument("--task-names", type=str, nargs="*", default=None,
                        help="Task names, one per dataset (positional). "
                             "If omitted, uses dataset names as task names.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for the merged v2 dataset.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite output directory + re-download if needed.")
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    if output_path.exists():
        if not args.force:
            log.error("Output '%s' exists. Use --force to overwrite.", output_path)
            sys.exit(1)
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    # Download / prepare each dataset
    v2_roots: list[Path] = []
    for ds_name in args.datasets:
        log.info("=" * 60)
        log.info("Processing: %s", ds_name)
        log.info("=" * 60)
        root = download_and_prepare(ds_name, args.force)
        v2_roots.append(root)

    # Default task names: use the last component of each dataset name
    task_names = args.task_names
    if task_names is None:
        task_names = []
        for ds_name in args.datasets:
            p = Path(ds_name)
            task_names.append(p.name)

    # Merge
    log.info("\n" + "=" * 60)
    log.info("Merging %d prepared datasets...", len(v2_roots))
    log.info("=" * 60)
    merge_datasets(v2_roots, task_names, output_path)

    print("\n" + "=" * 60)
    print("Merge complete!")
    print(f"  Output: {output_path}")
    print(f"  Datasets merged: {len(v2_roots)}")
    print(f"  Total episodes:  {load_info(output_path)['total_episodes']}")
    print(f"  Total frames:    {load_info(output_path)['total_frames']}")
    print(f"  Task names:      {task_names}")
    print("=" * 60)
    print("\nNext step:")
    print(f"  python scripts/data/convert_lerobot_to_gear.py \\")
    print(f"      --dataset-path {output_path} \\")
    print(f"      --embodiment-tag unitree_g1_upper_body_dex3")
    print("=" * 60)


if __name__ == "__main__":
    main()
