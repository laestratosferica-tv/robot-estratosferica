# Publicador programado multiplataforma V1

El programador revisa cada cinco minutos una cola de piezas con aprobación,
fecha, zona horaria y activo protegido por SHA-256. El estado remoto impide
duplicados y una pieza reclamada no vuelve a ejecutarse automáticamente.

## Cobertura

| Formato | Instagram | Facebook | Threads | YouTube |
|---|---|---|---|---|
| Reel / video corto | Automático | Automático | Pendiente de video nativo | Short automático |
| Carrusel de 2–10 imágenes | Automático | Automático | Automático | No aplica |
| Imagen individual | Automático | Automático | Automático | No aplica |
| Texto | No aplica | Automático con enlace | Automático | No aplica |
| Story sin interacción | Imagen automática | Pendiente de endpoint validado | No aplica | No aplica |
| Story con encuesta/sticker | Paso nativo manual | Paso nativo manual | No aplica | No aplica |

## Esquemas aceptados

- `supervised_meta_publication_v1`: Reel de Instagram o Facebook.
- `approved_carousel_publication_v1`: carrusel conjunto IG/FB/Threads.
- `approved_youtube_short_publication_v1`: Short de YouTube.
- `approved_social_post_v1`: imagen, texto/enlace o Story no interactiva.

## Compuertas

1. `PRODUCTION_ARMED=true`.
2. `SCHEDULED_PUBLISHING_ARMED=true`.
3. Aprobación idéntica entre cola y manifiesto.
4. Activo y huella digital válidos.
5. Reclamo remoto previo a cualquier publicación.
6. Para Amazon/afiliados: enlace, divulgación, disponibilidad y activo final
   deben estar verificados explícitamente.

## Límite deliberado

Las encuestas y stickers de Stories conservan un paso manual dentro de Meta.
El sistema prepara el archivo, la hora y la alerta, pero bloquea una falsa
automatización que perdería la interacción nativa.
