#!/usr/bin/env bash
set -euo pipefail

SLIDES="${1:?directorio de láminas requerido}"
OUT="${2:?archivo de salida requerido}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

SCRIPT="Crecimos cambiando de pantalla. Primero, el arcade. Después, la consola y el televisor. El cibercafé nos conectó. El móvil puso el juego en el bolsillo. Y la realidad virtual nos metió dentro. ¿En cuál mundo comenzó tu historia?"
say -v Paulina -r 188 -o "$WORK/voz.aiff" "$SCRIPT"

index=0
for file in "$SLIDES"/*.png; do
  index=$((index + 1))
  seconds="3.05"
  if [[ "$index" == "1" ]]; then seconds="2.65"; fi
  if [[ "$index" == "6" ]]; then seconds="4.25"; fi
  frames=$(awk -v s="$seconds" 'BEGIN { printf "%d", s*30 }')
  if (( index % 2 == 0 )); then
    xpos="iw/2-(iw/zoom/2)+24*sin(on/21)"
  else
    xpos="iw/2-(iw/zoom/2)-24*sin(on/21)"
  fi
  ffmpeg -loglevel error -y -loop 1 -i "$file" -vf \
    "split=2[bg][fg];[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=sigma=30,eq=brightness=-0.18:saturation=1.16[back];[fg]scale=1080:1350[front];[back][front]overlay=0:285,zoompan=z='min(zoom+0.00040,1.065)':x='$xpos':y='ih/2-(ih/zoom/2)':d=$frames:s=1080x1920:fps=30,drawbox=x='mod(t*520,1180)-100':y=0:w=100:h=1920:color=0x7DEBFF@0.06:t=fill,fade=t=in:st=0:d=0.10,fade=t=out:st=$(awk -v s="$seconds" 'BEGIN {print s-0.10}'):d=0.10,format=yuv420p" \
    -t "$seconds" -an -c:v libx264 -preset medium -crf 17 "$WORK/clip-$index.mp4"
done

for file in "$WORK"/clip-*.mp4; do printf "file '%s'\n" "$file"; done > "$WORK/list.txt"
ffmpeg -loglevel error -y -f concat -safe 0 -i "$WORK/list.txt" -c copy "$WORK/video.mp4"

ffmpeg -loglevel error -y -i "$WORK/voz.aiff" -filter_complex \
  "[0:a]highpass=f=105,lowpass=f=7800,acompressor=threshold=-18dB:ratio=2.3:attack=8:release=90,volume=1.0[clean];[0:a]highpass=f=500,lowpass=f=3500,flanger=delay=1.8:depth=0.8:regen=0:width=28:speed=0.25,volume=0.11[digital];[clean][digital]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7[a]" \
  -map '[a]' -ar 48000 "$WORK/voz-digital.wav"

duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WORK/video.mp4")
ffmpeg -loglevel error -y -i "$WORK/video.mp4" -i "$WORK/voz-digital.wav" \
  -f lavfi -i "sine=frequency=57:sample_rate=48000:duration=$duration" \
  -filter_complex "[1:a]apad=pad_dur=$duration[voice];[2:a]volume=0.012[hum];[voice][hum]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7[a]" \
  -map 0:v -map '[a]' -t "$duration" -c:v copy -c:a aac -b:a 192k -movflags +faststart "$OUT"

ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height -of default=noprint_wrappers=1 "$OUT"
