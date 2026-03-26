#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO_ID="${1:-data/Task_B/base}"
EPISODE_INDEX="${2:-0}"

python3 scripts/rerun_dataset_viz.py \
  --dataset-repo-id "${DATASET_REPO_ID}" \
  --episode-index "${EPISODE_INDEX}" \
  --show-skeleton \
  --show-ee-traj \
  --playback-sleep-s 0.03 \
  "${@:3}"
