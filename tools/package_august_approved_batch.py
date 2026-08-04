#!/usr/bin/env python3
"""Package the editorial approvals scheduled through 2026-08-20."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = Path("/Volumes/RESPALDO/reviciones estratosferica")
ASSETS = ROOT / "artifacts/approved/scheduled/august-2026"
MANIFESTS = ROOT / "artifacts/publication-manifests"
QUEUE = ROOT / "config/scheduled_publications_v1.json"
ACCOUNTS = {"instagram_username": "laestratosfericatv", "facebook_page_name": "La Estratosférica TV", "threads_username": "laestratosfericatv"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".mp4":
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
                "-c:v", "libx264", "-preset", "slow", "-crf", "23",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart", str(destination),
            ],
            check=True,
        )
    else:
        shutil.copy2(source, destination)
    return destination


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_manifest(name: str, payload: dict) -> str:
    target = MANIFESTS / f"{name}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return relative(target)


def add_queue(items: list[dict], content_id: str, manifest_path: str, approval_id: str, publish_at: str) -> None:
    items.append({"content_id": content_id, "manifest_path": manifest_path, "approval_id": approval_id, "publish_at": publish_at, "status": "approved", "enabled": True})


def reel(items: list[dict], *, slug: str, source: Path, publish_at: str, approval: str, caption: str, title: str) -> None:
    video = copy_file(source, ASSETS / slug / source.name)
    for platform in ("instagram", "facebook"):
        aid = f"{approval}-{platform}"
        name = f"{slug}-{platform}"
        manifest = write_manifest(name, {"schema": "supervised_meta_publication_v1", "slug": name, "platform": platform, "approval_id": aid, "caption": caption, "video_path": relative(video), "video_sha256": digest(video)})
        add_queue(items, name, manifest, aid, publish_at)
    aid = f"{approval}-youtube"
    name = f"{slug}-youtube"
    manifest = write_manifest(name, {"schema": "approved_youtube_short_publication_v1", "slug": name, "approval_id": aid, "title": title, "description": caption + "\n\n#Shorts", "video_path": relative(video), "video_sha256": digest(video), "privacy_status": "public"})
    add_queue(items, name, manifest, aid, publish_at)


def carousel(items: list[dict], *, slug: str, sources: list[Path], publish_at: str, approval: str, caption: str, threads_text: str) -> None:
    assets = []
    for index, source in enumerate(sources, 1):
        target = copy_file(source, ASSETS / slug / f"{index:02d}-{source.name}")
        assets.append({"order": index, "path": relative(target), "sha256": digest(target)})
    manifest = write_manifest(slug, {"schema": "approved_carousel_publication_v1", "slug": slug, "approval_id": approval, "platforms": ["instagram", "facebook", "threads"], "expected_accounts": ACCOUNTS, "caption": caption, "threads_text": threads_text, "assets": assets})
    add_queue(items, slug, manifest, approval, publish_at)


def image(items: list[dict], *, slug: str, source: Path, publish_at: str, approval: str, text: str) -> None:
    asset = copy_file(source, ASSETS / slug / source.name)
    for platform in ("instagram", "facebook", "threads"):
        aid = f"{approval}-{platform}"
        name = f"{slug}-{platform}"
        manifest = write_manifest(name, {"schema": "approved_social_post_v1", "slug": name, "approval_id": aid, "platform": platform, "post_type": "image", "text": text, "asset_path": relative(asset), "asset_sha256": digest(asset)})
        add_queue(items, name, manifest, aid, publish_at)


def text_link(items: list[dict], *, slug: str, publish_at: str, approval: str, facebook: str, threads: str, link: str) -> None:
    for platform, text in (("facebook", facebook), ("threads", threads)):
        aid = f"{approval}-{platform}"
        name = f"{slug}-{platform}"
        payload = {"schema": "approved_social_post_v1", "slug": name, "approval_id": aid, "platform": platform, "post_type": "text_link" if platform == "facebook" else "text", "text": text}
        if platform == "facebook": payload["link"] = link
        manifest = write_manifest(name, payload)
        add_queue(items, name, manifest, aid, publish_at)


def main() -> None:
    queue = json.loads(QUEUE.read_text(encoding="utf-8"))
    retained = [
        item
        for item in queue["items"]
        if not item["content_id"].startswith("august-approved-")
        or item["content_id"].startswith("august-approved-real-o-ia-")
        or item["content_id"].startswith("august-approved-ia-gamer")
        or item["content_id"].startswith("august-approved-pantallas")
        or item["content_id"].startswith("august-approved-nostalgia")
    ]
    items: list[dict] = []

    reel(items, slug="august-approved-fosil-guardar", source=BACKUP / "nostalgia-save-icon-v1/Fosil-Boton-Guardar-v6.mp4", publish_at="2026-08-08T20:00:00-05:00", approval="calendar-7tsofoae5fjsddi8bmpvmorvkk", title="El fósil del botón Guardar 💾", caption="El ícono Guardar sobrevivió al disquete que lo originó. ¿Qué otro símbolo sobrevivió a su tecnología? #NostalgiaGamer #GamingLATAM #Tecnologia #LaEstratosferica")

    reel(items, slug="august-approved-dlss", source=BACKUP / "dlss-fluidity-v1/Tecnologia-que-cambia-como-juegas-v3-video-real.mp4", publish_at="2026-08-09T18:30:00-05:00", approval="calendar-7jrm2u6su1q9g6d5iqnp4goc84", title="Tu GPU no dibuja todo", caption="DLSS 4.5 usa IA y generación dinámica de cuadros para cambiar la fluidez del juego. ¿Fluidez real o truco visual? Fuente: NVIDIA GeForce, CES 2026. #Gaming #NVIDIA #IA #LaEstratosferica")

    image(items, slug="august-approved-hle-remontada", source=BACKUP / "esports-comeback-data-v1/3-dias-remontada-MSI-v2-epica.png", publish_at="2026-08-10T12:30:00-05:00", approval="calendar-1jpdksv2lgff6gdg07a4g6igao", text="HLE cayó 1–3 ante BLG. Tres días después venció 3–2 al mismo rival y levantó el trofeo del MSI 2026. Una remontada que convirtió la derrota en lectura de juego. Fuente: LoL Esports. #Esports #MSI2026 #LeagueOfLegends #LaEstratosferica")

    text_link(items, slug="august-approved-wolverine", publish_at="2026-08-14T12:30:00-05:00", approval="calendar-5h3mbuda3iae5b87rh15d2ubam", link="https://blog.playstation.com/2026/07/23/marvels-wolverine-story-trailer-new-art-composer-details-and-more/", facebook="Wolverine no viene a salvarte. Está intentando sobrevivirse a sí mismo. El tráiler enfrenta a Logan con Jean Grey, The Hand y Deathstrike. Marvel’s Wolverine llega a PS5 el 15 de septiembre de 2026. ¿Imparable o vulnerable?", threads="Wolverine no viene a salvarte. Está intentando sobrevivirse a sí mismo. Jean Grey, The Hand y Deathstrike rodean a un Logan tan peligroso para sus enemigos como para él mismo. Llega a PS5 el 15 de septiembre. https://blog.playstation.com/2026/07/23/marvels-wolverine-story-trailer-new-art-composer-details-and-more/")

    reel(items, slug="august-approved-drones", source=BACKUP / "aprobadas/reel-ia-vs-campeones-drones-aprobado.mp4", publish_at="2026-08-15T20:00:00-05:00", approval="calendar-5kj5c1v8gtefg95cltpncco4mu", title="IA contra campeones de drones", caption="La inteligencia artificial ya compite contra pilotos campeones de drones. Velocidad, cálculo y reacción se encuentran en el aire. ¿Quién tiene la ventaja? #Drones #InteligenciaArtificial #Competencia #LaEstratosferica")

    reel(items, slug="august-approved-esports-lectura", source=BACKUP / "aprobadas/reel-esports-lectura-profesional-aprobado.mp4", publish_at="2026-08-17T12:30:00-05:00", approval="calendar-05pvfpc07s0ukkeh9sfi5pofhj", title="Lee la jugada como un profesional", caption="Una pelea se decide antes del primer disparo. Visión. Posición. Tiempo. ¿Qué miras primero cuando empieza la presión? Fuente audiovisual: Ron Lach / Pexels. #Esports #GamingLATAM #Competitivo #LaEstratosferica")

    reel(items, slug="august-approved-unity", source=BACKUP / "aprobadas/reel-unity-7-ia-creacion-aprobado.mp4", publish_at="2026-08-18T20:00:00-05:00", approval="calendar-6ceo2f1k9dm42hgsqbmnupdc4g", title="La IA cruzó esta línea en Unity 7", caption="La IA ya no solo vive dentro del juego: también entra al equipo que lo construye. Unity 7 plantea colaboración entre creadores, equipos y agentes de código. ¿Herramienta creativa o reemplazo? Fuente: Unity. #IA #GameDev #Unity7 #LaEstratosferica")

    carousel(items, slug="august-approved-xbox-reglas", publish_at="2026-08-19T12:30:00-05:00", approval="calendar-7p55h1d8o4jsveg107c2skesah", sources=[BACKUP / f"aprobadas/carrusel-xbox-cambia-reglas-aprobado/slides/{name}.png" for name in ("01-portada","02-antes","03-halo","04-negocio","05-balance","06-pregunta")], caption="Xbox está cambiando las reglas. Halo en PlayStation muestra una competencia que se mueve hacia acceso, suscripciones, PC, nube y comunidad. ¿Comprarías una consola sin exclusivos? Fuentes: Xbox Wire y PlayStation Blog. #Xbox #Halo #PlayStation #LaEstratosferica", threads_text="Halo en PlayStation cambia el papel del exclusivo. Si las franquicias pueden viajar, ¿qué debería hacer diferente a cada consola?")

    reel(items, slug="august-approved-multiverso", source=BACKUP / "aprobadas/multiverso-estratosferica-aprobado/Multiverso-Estratosferica-aprobado.mp4", publish_at="2026-08-20T20:00:00-05:00", approval="calendar-30t3u3mr5sgf03vu954ig4s5kj", title="El multiverso de La Estratosférica", caption="La memoria nos enseñó a jugar. La competencia nos obligó a mejorar. La tecnología cambió la pantalla. La IA cambió quién puede crear. Y la comunidad decide qué mundo abrimos después. ¿Qué mundo quieres explorar ahora? #LaEstratosferica #GamingLATAM #Esports #Tecnologia #IA")

    queue["items"] = retained + items
    QUEUE.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Packaged {len(items)} platform outputs")


if __name__ == "__main__":
    main()
