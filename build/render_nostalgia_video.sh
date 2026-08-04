#!/usr/bin/env bash
set -euo pipefail

SLIDES="${1:?directorio de láminas requerido}"
OUT="${2:?archivo de salida requerido}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

SCRIPT="Si viviste esto, eres de los nuestros. Guardar era sagrado. Los secretos vivían en un cuaderno. Esperar el turno también era jugar. Y pasar el control era confiar. ¿Qué recuerdo gamer todavía tienes intacto?"
say -v Juan -r 178 -o "$WORK/voz.aiff" "$SCRIPT"

index=0
for file in "$SLIDES"/*.png; do
  index=$((index + 1))
  seconds="3.35"
  if [[ "$index" == "1" ]]; then seconds="2.85"; fi
  if [[ "$index" == "5" ]]; then seconds="4.70"; fi
  frames=$(awk -v s="$seconds" 'BEGIN { printf "%d", s*30 }')
  phase=$((index * 17))
  ffmpeg -loglevel error -y -loop 1 -i "$file" -vf \
    "split=3[bg][mid][fg];\
     [bg]scale=1220:2168:force_original_aspect_ratio=increase,crop=1080:1920:x='70+28*sin((n+$phase)/27)':y='124+38*cos((n+$phase)/31)',gblur=sigma=34,eq=brightness=-0.22:saturation=1.22[back];\
     [mid]scale=1180:1475,crop=1080:1350:x='50+22*sin((n+$phase)/24)':y='62+18*cos((n+$phase)/29)',eq=contrast=1.06:saturation=1.08[mid2];\
     [fg]scale=1080:1350,format=rgba,colorchannelmixer=aa=0.18,gblur=sigma=2[glow];\
     [back][mid2]overlay=0:285[base];[base][glow]overlay=x='10*sin((n+$phase)/19)':y='285+6*cos((n+$phase)/23)',\
     drawbox=x='mod(t*610,1320)-180':y=0:w=130:h=1920:color=0x62E7FF@0.045:t=fill,\
     drawbox=x=0:y='mod(t*270,2160)-160':w=1080:h=3:color=white@0.10:t=fill,\
     noise=alls=2.2:allf=t,fade=t=in:st=0:d=0.16,fade=t=out:st=$(awk -v s="$seconds" 'BEGIN {print s-0.16}'):d=0.16,format=yuv420p" \
    -t "$seconds" -r 30 -an -c:v libx264 -preset medium -crf 17 "$WORK/clip-$index.mp4"
done

for file in "$WORK"/clip-*.mp4; do printf "file '%s'\n" "$file"; done > "$WORK/list.txt"
ffmpeg -loglevel error -y -f concat -safe 0 -i "$WORK/list.txt" -c copy "$WORK/video.mp4"

ffmpeg -loglevel error -y -i "$WORK/voz.aiff" -filter_complex \
  "[0:a]highpass=f=95,lowpass=f=7900,acompressor=threshold=-18dB:ratio=2.2:attack=8:release=100,volume=1.04[voice];\
   [0:a]highpass=f=700,lowpass=f=3300,aecho=0.7:0.28:28:0.06,volume=0.045[memory];\
   [voice][memory]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7[a]" \
  -map '[a]' -ar 48000 "$WORK/voz-final.wav"

duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WORK/video.mp4")
ffmpeg -loglevel error -y -i "$WORK/video.mp4" -i "$WORK/voz-final.wav" \
  -f lavfi -i "sine=frequency=59:sample_rate=48000:duration=$duration" \
  -filter_complex "[1:a]apad=pad_dur=$duration[voice];[2:a]volume=0.008[hum];[voice][hum]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7[a]" \
  -map 0:v -map '[a]' -t "$duration" -c:v copy -c:a aac -b:a 192k -movflags +faststart "$OUT"

ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height -of default=noprint_wrappers=1 "$OUT"
