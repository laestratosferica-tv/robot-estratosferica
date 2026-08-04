#!/usr/bin/env bash
set -euo pipefail

SLIDES="${1:?directorio de láminas requerido}"
OUT="${2:?archivo de salida requerido}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

SCRIPT="Éxbox está cambiando las reglas. Antes, un exclusivo vendía una consola. Ahora, Halo también llega a PlayStation. El negocio se mueve hacia el acceso, las suscripciones, la computadora y la nube. Si los juegos pueden viajar, ¿qué debería hacer única a cada consola?"
say -v Paulina -r 184 -o "$WORK/voz.aiff" "$SCRIPT"

index=0
for file in "$SLIDES"/*.png; do
  index=$((index + 1))
  seconds="3.15"
  if [[ "$index" == "1" ]]; then seconds="2.55"; fi
  if [[ "$index" == "6" ]]; then seconds="4.15"; fi
  frames=$(awk -v s="$seconds" 'BEGIN { printf "%d", s*30 }')
  phase=$((index * 23))
  ffmpeg -loglevel error -y -loop 1 -i "$file" -vf \
    "split=3[bg][front][light];\
     [bg]scale=1220:2168:force_original_aspect_ratio=increase,crop=1080:1920:x='70+34*sin((n+$phase)/29)':y='124+42*cos((n+$phase)/33)',gblur=sigma=31,eq=brightness=-0.20:saturation=1.24[back];\
     [front]scale=1160:1450,crop=1080:1350:x='40+25*sin((n+$phase)/22)':y='50+18*cos((n+$phase)/27)',eq=contrast=1.08:saturation=1.10[main];\
     [light]scale=1080:1350,format=rgba,colorchannelmixer=aa=0.12,gblur=sigma=3[halo];\
     [back][main]overlay=0:285[base];[base][halo]overlay=x='8*sin((n+$phase)/18)':y='285+5*cos((n+$phase)/21)',\
     drawbox=x='mod(t*760,1380)-210':y=0:w=170:h=1920:color=0x58F08A@0.055:t=fill,\
     drawbox=x=0:y='mod(t*380,2200)-120':w=1080:h=2:color=white@0.13:t=fill,\
     fade=t=in:st=0:d=0.13,fade=t=out:st=$(awk -v s="$seconds" 'BEGIN {print s-0.13}'):d=0.13,format=yuv420p" \
    -t "$seconds" -r 30 -an -c:v libx264 -preset medium -crf 17 "$WORK/clip-$index.mp4"
done

for file in "$WORK"/clip-*.mp4; do printf "file '%s'\n" "$file"; done > "$WORK/list.txt"
ffmpeg -loglevel error -y -f concat -safe 0 -i "$WORK/list.txt" -c copy "$WORK/video.mp4"

ffmpeg -loglevel error -y -i "$WORK/voz.aiff" -filter_complex \
  "[0:a]highpass=f=100,lowpass=f=7900,acompressor=threshold=-18dB:ratio=2.3:attack=7:release=90,volume=1.03[clean];\
   [0:a]highpass=f=620,lowpass=f=4100,flanger=delay=1.2:depth=0.45:regen=0:width=20:speed=0.20,volume=0.075[machine];\
   [clean][machine]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7[a]" \
  -map '[a]' -ar 48000 "$WORK/voz-final.wav"

duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WORK/video.mp4")
ffmpeg -loglevel error -y -i "$WORK/video.mp4" -i "$WORK/voz-final.wav" \
  -f lavfi -i "sine=frequency=64:sample_rate=48000:duration=$duration" \
  -filter_complex "[1:a]apad=pad_dur=$duration[voice];[2:a]volume=0.010[hum];[voice][hum]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7[a]" \
  -map 0:v -map '[a]' -t "$duration" -c:v copy -c:a aac -b:a 192k -movflags +faststart "$OUT"

ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height -of default=noprint_wrappers=1 "$OUT"
