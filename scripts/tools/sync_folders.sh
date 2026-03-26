#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/sync_folders.sh --src /path/to/source --dst /path/to/destination [options]

Options:
  --mode sync|copy        Sync mode. Default: sync
  --src PATH              Source folder
  --dst PATH              Destination folder
  --delete                Delete files in destination that no longer exist in source
  --dry-run               Show what would change without writing
  --log-dir DIR           Log directory. Default: ./logs/folder_sync
  --lock-file PATH        Lock file path. Default: /tmp/folder_sync.lock
  --extra-arg ARG         Extra rsync arg. Repeatable.
  --help                  Show this help

Environment variables:
  FOLDER_SYNC_MODE
  FOLDER_SYNC_SRC
  FOLDER_SYNC_DST
  FOLDER_SYNC_DELETE=1
  FOLDER_SYNC_DRY_RUN=1
  FOLDER_SYNC_LOG_DIR
  FOLDER_SYNC_LOCK_FILE

Examples:
  scripts/sync_folders.sh \
    --src /mnt/cloud_a/project \
    --dst /mnt/cloud_b/project \
    --delete

  scripts/sync_folders.sh \
    --mode copy \
    --src /data/export \
    --dst /backup/export
EOF
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

MODE="${FOLDER_SYNC_MODE:-sync}"
SRC="${FOLDER_SYNC_SRC:-}"
DST="${FOLDER_SYNC_DST:-}"
DELETE_FLAG="${FOLDER_SYNC_DELETE:-0}"
DRY_RUN="${FOLDER_SYNC_DRY_RUN:-0}"
LOG_DIR="${FOLDER_SYNC_LOG_DIR:-./logs/folder_sync}"
LOCK_FILE="${FOLDER_SYNC_LOCK_FILE:-/tmp/folder_sync.lock}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --src)
      SRC="$2"
      shift 2
      ;;
    --dst)
      DST="$2"
      shift 2
      ;;
    --delete)
      DELETE_FLAG=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --lock-file)
      LOCK_FILE="$2"
      shift 2
      ;;
    --extra-arg)
      EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --extra-arg=*)
      EXTRA_ARGS+=("${1#*=}")
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SRC" || -z "$DST" ]]; then
  echo "--src and --dst are required" >&2
  usage >&2
  exit 1
fi

case "$MODE" in
  sync|copy)
    ;;
  *)
    echo "Unsupported --mode: $MODE" >&2
    exit 1
    ;;
esac

if [[ ! -d "$SRC" ]]; then
  echo "Source folder does not exist: $SRC" >&2
  exit 1
fi

mkdir -p "$DST"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date '+%Y%m%d_%H%M%S')_${MODE}.log"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another sync job is already running: $LOCK_FILE" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync not found" >&2
  exit 1
fi

RSYNC_ARGS=(
  "-a"
  "--human-readable"
  "--stats"
  "--log-file=$LOG_FILE"
)

if [[ "$DELETE_FLAG" == "1" && "$MODE" == "sync" ]]; then
  RSYNC_ARGS+=("--delete")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_ARGS+=("--dry-run" "--itemize-changes")
fi

RSYNC_ARGS+=("${EXTRA_ARGS[@]}")

CMD=(rsync "${RSYNC_ARGS[@]}" "$SRC"/ "$DST"/)

log "Starting folder sync"
log "Mode: $MODE"
log "Source: $SRC"
log "Destination: $DST"
log "Delete extra destination files: $DELETE_FLAG"
log "Dry run: $DRY_RUN"
log "Log file: $LOG_FILE"
printf '[%s] Command:' "$(timestamp)"
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

log "Folder sync finished successfully"
