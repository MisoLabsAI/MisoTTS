#!/usr/bin/env bash
# Usage: ./generate_clips.sh <path-to-video> <clip_numbers...>
# Example: ./generate_clips.sh "Silo - Beauty and a Beat.mov" 1 4 6
#
# Cuts source video segments, crops to 9:16 (1080x1920), and saves to ./clips/
# Requires: ffmpeg

set -e

VIDEO="$1"
shift
CLIPS=("$@")

if [[ -z "$VIDEO" || ${#CLIPS[@]} -eq 0 ]]; then
  echo "Usage: $0 <video_file> <clip_numbers...>"
  exit 1
fi

if ! command -v ffmpeg &>/dev/null; then
  echo "Error: ffmpeg not found. Install with: sudo apt install ffmpeg"
  exit 1
fi

mkdir -p clips

# --- Clip definitions: [start_seconds] [duration_seconds] [description]
declare -A CLIP_START CLIP_DUR CLIP_DESC
CLIP_START[1]=0;    CLIP_DUR[1]=3;    CLIP_DESC[1]="hook_wide_shot"
CLIP_START[2]=3;    CLIP_DUR[2]=2;    CLIP_DESC[2]="build_crowd_closeup"
CLIP_START[3]=5;    CLIP_DUR[3]=1.5;  CLIP_DESC[3]="predrop_aoki_decks"
CLIP_START[4]=6.5;  CLIP_DUR[4]=3.5;  CLIP_DESC[4]="drop_payoff_crowd"
CLIP_START[5]=10;   CLIP_DUR[5]=3;    CLIP_DESC[5]="reaction_cta"
CLIP_START[6]=13;   CLIP_DUR[6]=2;    CLIP_DESC[6]="hold_loop"

# Crop filter: crops center 9:16 from source, scales to 1080x1920
CROP_FILTER="crop=ih*9/16:ih,scale=1080:1920"

for NUM in "${CLIPS[@]}"; do
  if [[ -z "${CLIP_START[$NUM]}" ]]; then
    echo "Unknown clip number: $NUM (valid: 1-6)"
    continue
  fi

  START="${CLIP_START[$NUM]}"
  DUR="${CLIP_DUR[$NUM]}"
  DESC="${CLIP_DESC[$NUM]}"
  OUT="clips/clip_${NUM}_${DESC}.mp4"

  echo "Generating clip $NUM ($DESC): ${START}s + ${DUR}s → $OUT"
  ffmpeg -y -ss "$START" -i "$VIDEO" -t "$DUR" \
    -vf "$CROP_FILTER" \
    -r 30 -c:v libx264 -crf 18 -preset fast \
    -c:a aac -b:a 192k \
    "$OUT"
  echo "  Done → $OUT"
done

echo ""
echo "All done. Clips saved to ./clips/"
