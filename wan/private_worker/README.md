# Worker privado Wan para RunPod

Esta imagen derivada conserva el motor Wan 2.2 ya validado y endurece el handler antes de usar referencias privadas de Nova, Joseverso o Rami.

## Protecciones

- Nunca registra el contenido completo del `job_input`.
- Acepta imágenes únicamente por `image_base64`.
- Desactiva rutas locales y descargas URL proporcionadas por el trabajo.
- Rechaza Base64 inválido o imágenes superiores a 9 MiB.
- Falla durante la construcción si el handler base cambió y el parche ya no coincide.

## Construcción

```bash
docker build -t estratosferica/wan-private-worker:wan22-a9247705c .
```

La imagen base está fijada en `registry.runpod.net/wlsdml1114-generate-video-ksampler-dockerfile:a9247705c`. Antes de distribuir públicamente la imagen derivada debe revisarse la licencia del proveedor base; el uso previsto aquí es interno y privado.

El workflow manual `Build Wan Private Worker` publica dos etiquetas inmutables en GHCR. No ejecuta trabajos de video ni activa GPUs de RunPod.
