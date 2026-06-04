#!/usr/bin/env bash
# Usage: ./combine_clips.sh clip_1_hook.mp4 clip_4_drop.mp4 ...
# Concatenates provided clip files into clips/combined.mp4

set -e

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <clip1.mp4> <clip2.mp4> [...]"
  exit 1
fi

CONCAT_LIST=$(mktemp /tmp/concat_XXXXXX.txt)
for F in "$@"; do
  echo "file '$(realpath "$F")'" >> "$CONCAT_LIST"
done

OUT="clips/combined.mp4"
ffmpeg -y -f concat -safe 0 -i "$CONCAT_LIST" -c copy "$OUT"
rm "$CONCAT_LIST"
echo "Combined → $OUT"
