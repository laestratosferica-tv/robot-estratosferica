#!/usr/bin/env python3
"""Replenish the independent Contenido divertido queue with licensed gameplay."""
from __future__ import annotations

import hashlib, json, subprocess
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "config/scheduled_publications_v1.json"
OUT = ROOT / "artifacts/approved/scheduled/contenido_divertido-2026-08"
MANIFESTS = ROOT / "artifacts/publication-manifests"
SOURCE = ROOT / "artifacts/contenido_divertido/license/xonotic-0-8-2-gameplay.webm"
OVERLAY_RENDERER = ROOT / "tools/render_fun_gameplay_overlay.swift"
TZ = timezone(timedelta(hours=-5))
SOURCE_URL = "https://upload.wikimedia.org/wikipedia/commons/b/b9/Xonotic_0-8-2_gameplay.webm"
SOURCE_PAGE = "https://commons.wikimedia.org/wiki/File:Xonotic_0-8-2_gameplay.webm"
PIECES = [
    ("doble-eliminacion", 14, "DOS ELIMINACIONES", "El respawn ni alcanzó a calentar."),
    ("atajo-explosivo", 37, "ENCONTRÓ UN ATAJO", "El atajo encontró una pared."),
    ("salto-con-proposito", 59, "SALTO CON PROPÓSITO", "El propósito era caer con estilo."),
    ("cohete-demasiado-cerca", 82, "COHETE DEMASIADO CERCA", "La distancia era una sugerencia."),
    ("combo-improbable", 105, "COMBO IMPROBABLE", "El rival pidió repetición."),
    ("giro-y-castigo", 128, "GIRO Y CASTIGO", "Miró atrás. Mala idea."),
    ("final-sin-frenos", 151, "FINAL SIN FRENOS", "El caos también tiene puntería."),
]

def run(*args): subprocess.run(args, check=True)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def render(folder, slug, start, hook, payoff):
    out = folder/f"{slug}.mp4"; overlay = folder / "overlay.png"
    run("swift", str(OVERLAY_RENDERER), str(overlay), hook, payoff)
    vf=("[0:v]scale=1080:633,pad=1080:1920:0:644:color=0x060611[base];"
        "[base][1:v]overlay=0:0,format=yuv420p[v]")
    run("ffmpeg","-y","-v","error","-ss",str(start),"-t","8","-i",str(SOURCE),"-loop","1","-framerate","30","-i",str(overlay),"-filter_complex",vf,"-map","[v]","-map","0:a?","-t","8","-c:v","libx264","-preset","ultrafast","-crf","18","-c:a","aac","-b:a","192k","-ar","48000","-movflags","+faststart",str(out))
    return out

def manifest(slug, platform, video, digest, day, hook, payoff):
    ident=f"contenido-divertido-{slug}-{platform}-{day.isoformat()}"; rel=video.relative_to(ROOT).as_posix()
    credit="Gameplay: Xonotic 0.8.2, Drummyfish y desarrolladores de Xonotic (GPLv3+). Fuente: " + SOURCE_PAGE
    text=f"{hook.title()}. {payoff}\n\n{credit}\n\n#Xonotic #Gameplay #Gaming #LaEstratosferica"
    if platform=="youtube": data={"schema":"approved_youtube_short_publication_v1","slug":ident,"approval_id":ident,"video_path":rel,"video_sha256":digest,"title":hook.title()+" #Shorts","description":text+"\n#Shorts","privacy_status":"public"}
    elif platform=="threads": data={"schema":"approved_social_post_v1","slug":ident,"approval_id":ident,"platform":"threads","post_type":"video","asset_path":rel,"asset_sha256":digest,"text":text}
    else: data={"schema":"supervised_meta_publication_v1","slug":ident,"platform":platform,"approval_id":ident,"caption":text,"video_path":rel,"video_sha256":digest}
    path=MANIFESTS/f"{ident}.json"; path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n"); return path

def main():
    if not SOURCE.exists(): raise SystemExit("licensed_source_missing")
    queue=json.loads(QUEUE.read_text()); start_day=date(2026,8,10)
    keep=[]
    for item in queue["items"]:
        if item.get("category")=="contenido_divertido" and item.get("experiment")=="contenido-divertido-daily-2am-v1" and item.get("publish_at","") >= start_day.isoformat(): continue
        keep.append(item)
    for offset,(slug,clip_start,hook,payoff) in enumerate(PIECES):
        day=start_day+timedelta(days=offset); folder=OUT/f"{day.isoformat()}-{slug}"; folder.mkdir(parents=True,exist_ok=True)
        video=render(folder,slug,clip_start,hook,payoff); digest=sha(video)
        evidence={"schema":"contenido_divertido_license_evidence_v1","slug":slug,"category":"contenido_divertido","source_page":SOURCE_PAGE,"direct_asset":SOURCE_URL,"source_sha256":sha(SOURCE),"game":"Xonotic 0.8.2","author":"Drummyfish and Xonotic developers","license":"GPL-3.0-or-later","license_url":"https://www.gnu.org/licenses/gpl-3.0.html","authorization":"The uploader releases the recording under GPLv3 or later; source page confirms own recording against bots and no extra rights reserved.","credit":"Xonotic 0.8.2 gameplay by Drummyfish and Xonotic developers, GPLv3+.","game_audio_retained":True,"duration_seconds":8,"sha256":digest}
        (folder/"SOURCE_AND_LICENSE.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
        for platform in ("instagram","facebook","threads","youtube"):
            path=manifest(slug,platform,video,digest,day,hook,payoff); ident=f"contenido-divertido-{slug}-{platform}-{day.isoformat()}"
            keep.append({"content_id":ident,"manifest_path":path.relative_to(ROOT).as_posix(),"approval_id":ident,"publish_at":datetime.combine(day,time(2),TZ).isoformat(),"status":"approved","enabled":True,"experiment":"contenido-divertido-daily-2am-v1","category":"contenido_divertido","category_label":"Contenido divertido","license_evidence_path":(folder/"SOURCE_AND_LICENSE.json").relative_to(ROOT).as_posix()})
    queue["items"]=keep; QUEUE.write_text(json.dumps(queue,ensure_ascii=False,indent=2)+"\n")

if __name__ == "__main__": main()
