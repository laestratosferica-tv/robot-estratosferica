#!/usr/bin/env bash
set -euo pipefail

SLIDES="${1:?directorio del carrusel requerido}"
OUT="${2:?archivo de video requerido}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

SCRIPT_OPEN="¿Cuál de estos dos setaps existe? Mira los cables. Mira el control."
SCRIPT_DECIDE="Decide: ¿a o bé?"
SCRIPT_POST="Bé. Fue generado con inteligencia artificial. ¿Debería marcarse siempre?"
say -v Paulina -r 185 -o "$WORK/voz-open.aiff" "$SCRIPT_OPEN"
say -v Paulina -r 185 -o "$WORK/voz-decide.aiff" "$SCRIPT_DECIDE"
say -v Paulina -r 185 -o "$WORK/voz-post.aiff" "$SCRIPT_POST"
ffmpeg -loglevel error -y -i "$WORK/voz-open.aiff" -f lavfi -t 1 -i anullsrc=r=48000:cl=mono \
  -i "$WORK/voz-decide.aiff" -f lavfi -t 3 -i anullsrc=r=48000:cl=mono \
  -i "$WORK/voz-post.aiff" -filter_complex "[0:a][1:a][2:a][3:a][4:a]concat=n=5:v=0:a=1[voice]" \
  -map '[voice]' "$WORK/voz.aiff"

for spec in \
  "01-portada.png:2.80:0.00035" \
  "02-opcion-a.png:3.15:0.00042" \
  "03-opcion-b.png:3.15:0.00042" \
  "04-revelacion.png:3.55:0.00030" \
  "05-conversacion.png:4.85:0.00022"; do
  IFS=: read -r file seconds zoom <<<"$spec"
  frames=$(awk -v s="$seconds" 'BEGIN { printf "%d", s*30 }')
  ffmpeg -loglevel error -y -loop 1 -i "$SLIDES/$file" -vf \
    "split=2[bg][fg];[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=sigma=28,eq=brightness=-0.20[back];[fg]scale=1080:1350[front];[back][front]overlay=0:285,zoompan=z='min(zoom+$zoom,1.06)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=$frames:s=1080x1920:fps=30,fade=t=in:st=0:d=0.12,fade=t=out:st=$(awk -v s="$seconds" 'BEGIN {print s-0.12}'):d=0.12,format=yuv420p" \
    -t "$seconds" -an -c:v libx264 -preset medium -crf 17 "$WORK/${file%.png}.mp4"
done

printf "file '%s'\n" \
  "$WORK/01-portada.mp4" \
  "$WORK/02-opcion-a.mp4" \
  "$WORK/03-opcion-b.mp4" \
  "$WORK/04-revelacion.mp4" \
  "$WORK/05-conversacion.mp4" > "$WORK/list.txt"

ffmpeg -loglevel error -y -f concat -safe 0 -i "$WORK/list.txt" -c copy "$WORK/video.mp4"

ffmpeg -loglevel error -y -i "$WORK/voz.aiff" -filter_complex \
  "[0:a]highpass=f=105,lowpass=f=7600,acompressor=threshold=-18dB:ratio=2.5:attack=8:release=90,volume=1.00[clean];[0:a]highpass=f=420,lowpass=f=3300,flanger=delay=2.5:depth=1.2:regen=0:width=35:speed=0.32,volume=0.20[metal];[clean][metal]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7[a]" \
  -map '[a]' -ar 48000 "$WORK/voz-metal.wav"

duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WORK/video.mp4")
ffmpeg -loglevel error -y -i "$WORK/video.mp4" -i "$WORK/voz-metal.wav" \
  -f lavfi -i "sine=frequency=58:sample_rate=48000:duration=$duration" \
  -f lavfi -i "anoisesrc=color=pink:amplitude=0.012:sample_rate=48000:duration=$duration" \
  -filter_complex "[1:a]apad=pad_dur=$duration[voice];[2:a]volume=0.018[hum];[3:a]highpass=f=3500,volume=0.025[air];[voice][hum][air]amix=inputs=3:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=7[a]" \
  -map 0:v -map '[a]' -t "$duration" -c:v copy -c:a aac -b:a 192k -movflags +faststart "$OUT"

ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height,r_frame_rate -of default=noprint_wrappers=1 "$OUT"
