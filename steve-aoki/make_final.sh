#!/usr/bin/env bash
# Generates clips 1, 4, 6 from source video then stitches into a single
# 15-second 9:16 final edit ready for TikTok / Instagram.
#
# Usage: ./make_final.sh "Silo - Beauty and a Beat.mov"

set -e

VIDEO="$1"

if [[ -z "$VIDEO" ]]; then
  echo "Usage: $0 <video_file>"
  exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
  echo "Error: ffmpeg not found. Install with: brew install ffmpeg"
  exit 1
fi

mkdir -p clips

CROP="crop=ih*9/16:ih,scale=1080:1920"
ENCODE="-r 30 -c:v libx264 -crf 18 -preset fast -c:a aac -b:a 192k"

echo "Cutting clip 1 (hook)..."
ffmpeg -y -ss 0 -i "$VIDEO" -t 3 -vf "$CROP" $ENCODE clips/clip_1_hook.mp4

echo "Cutting clip 4 (drop payoff)..."
ffmpeg -y -ss 6.5 -i "$VIDEO" -t 3.5 -vf "$CROP" $ENCODE clips/clip_4_drop.mp4

echo "Cutting clip 6 (hold loop)..."
ffmpeg -y -ss 13 -i "$VIDEO" -t 2 -vf "$CROP" $ENCODE clips/clip_6_hold.mp4

echo "Stitching final edit..."
CONCAT=$(mktemp /tmp/concat_XXXXXX.txt)
echo "file '$(realpath clips/clip_1_hook.mp4)'" >> "$CONCAT"
echo "file '$(realpath clips/clip_4_drop.mp4)'" >> "$CONCAT"
echo "file '$(realpath clips/clip_6_hold.mp4)'" >> "$CONCAT"

ffmpeg -y -f concat -safe 0 -i "$CONCAT" -c copy clips/final_steveaoki.mp4
rm "$CONCAT"

echo ""
echo "✅ Done → clips/final_steveaoki.mp4"
echo "   Duration: ~8.5s (hook 3s + drop 3.5s + hold 2s)"
echo "   Format: 1080x1920 | 30fps | H.264"
