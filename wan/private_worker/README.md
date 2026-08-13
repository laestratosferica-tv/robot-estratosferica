# Worker oficial Wan 2.2 para RunPod

Worker privado basado exclusivamente en:

- código oficial `Wan-Video/Wan2.2`, fijado al commit `42bf4cf`;
- pesos oficiales `Wan-AI/Wan2.2-TI2V-5B` (Apache 2.0);
- una imagen pública de PyTorch/CUDA;
- un handler propio y auditable de La Estratosférica.

No incluye LoRAs, nodos comunitarios, descargas arbitrarias ni publicación en redes.

## Flujo

1. Construir y publicar la imagen desde el workflow manual.
2. Montar un volumen persistente en `/runpod-volume`.
3. Ejecutar una sola vez `python /app/download_model.py` para descargar los 34.2 GB del modelo oficial.
4. Configurar el endpoint Serverless con GPU de al menos 24 GB.
5. Probar primero con `frame_num=17` y `sample_steps=10`.

La GPU permanece apagada hasta que la construcción pase. La generación paga requiere además las compuertas existentes del cliente (`enabled`, `allow_paid_remote` y `WAN_PRIVATE_WORKER=1`).

## Entrada aceptada

```json
{
  "input": {
    "image_base64": "...",
    "prompt": "Movimiento natural y consistente del personaje",
    "size": "704*1280",
    "frame_num": 17,
    "sample_steps": 10,
    "seed": 72993276
  }
}
```

Solo se acepta imagen PNG/JPEG en Base64. URLs y rutas del cliente están deshabilitadas.

## Reversión

Cambiar el endpoint a la última etiqueta inmutable verificada de GHCR o escalar workers a cero. El workflow de construcción nunca invoca RunPod.
