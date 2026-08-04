#!/usr/bin/env bash
set -euo pipefail

SLIDES="${1:?directorio de láminas requerido}"
OUT="${2:?archivo de salida requerido}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

SCRIPT="Una inteligencia artificial ya puede entrar a tu partida. Te ayuda si te atoras. Revisa tu historial y tus logros. Recomienda qué jugar. Y responde por voz sin soltar el control. Cuatro poderes. Una pregunta: ¿ayuda o trampa?"
say -v Juan -r 190 -o "$WORK/voz.aiff" "$SCRIPT"

index=0
for file in "$SLIDES"/*.png; do
  index=$((index + 1))
  seconds="2.75"
  if [[ "$index" == "1" ]]; then seconds="2.55"; fi
  if [[ "$index" == "6" ]]; then seconds="5.00"; fi
  frames=$(awk -v s="$seconds" 'BEGIN { printf "%d", s*30 }')
  direction=$((index % 2))
  if [[ "$direction" == "0" ]]; then
    xpos="iw/2-(iw/zoom/2)+18*sin(on/18)"
  else
    xpos="iw/2-(iw/zoom/2)-18*sin(on/18)"
  fi
  ffmpeg -loglevel error -y -loop 1 -i "$file" -vf \
    "split=2[bg][fg];[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=sigma=32,eq=brightness=-0.22:saturation=1.18[back];[fg]scale=1080:1350[front];[back][front]overlay=0:285,zoompan=z='min(zoom+0.00034,1.055)':x='$xpos':y='ih/2-(ih/zoom/2)':d=$frames:s=1080x1920:fps=30,drawbox=x=0:y='mod(t*420,1920)':w=1080:h=5:color=0x20E5F2@0.24:t=fill,fade=t=in:st=0:d=0.10,fade=t=out:st=$(awk -v s="$seconds" 'BEGIN {print s-0.10}'):d=0.10,format=yuv420p" \
    -t "$seconds" -an -c:v libx264 -preset medium -crf 17 "$WORK/clip-$index.mp4"
done

for file in "$WORK"/clip-*.mp4; do printf "file '%s'\n" "$file"; done > "$WORK/list.txt"
ffmpeg -loglevel error -y -f concat -safe 0 -i "$WORK/list.txt" -c copy "$WORK/video.mp4"

ffmpeg -loglevel error -y -i "$WORK/voz.aiff" -filter_complex \
  "[0:a]highpass=f=100,lowpass=f=7600,acompressor=threshold=-18dB:ratio=2.4:attack=8:release=90,volume=1.0[clean];[0:a]highpass=f=380,lowpass=f=3200,flanger=delay=2.2:depth=1.0:regen=0:width=32:speed=0.30,volume=0.16[machine];[clean][machine]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7[a]" \
  -map '[a]' -ar 48000 "$WORK/voz-machine.wav"

duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WORK/video.mp4")
ffmpeg -loglevel error -y -i "$WORK/video.mp4" -i "$WORK/voz-machine.wav" \
  -f lavfi -i "sine=frequency=64:sample_rate=48000:duration=$duration" \
  -filter_complex "[1:a]apad=pad_dur=$duration[voice];[2:a]volume=0.016[hum];[voice][hum]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7[a]" \
  -map 0:v -map '[a]' -t "$duration" -c:v copy -c:a aac -b:a 192k -movflags +faststart "$OUT"

ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height -of default=noprint_wrappers=1 "$OUT"
