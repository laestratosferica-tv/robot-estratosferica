# Wan 2.2 TI2V-5B — prueba de concepto segura

Cliente reproducible para animar una imagen de referencia mediante un endpoint ComfyUI. La configuración nace **bloqueada**: no consume GPU ni crea recursos pagados.

## Alcance

- Imagen + prompt -> workflow Wan 2.2 TI2V-5B -> MP4 H.264.
- Parámetros por defecto: 704x1280, 49 fotogramas, 24 fps, 20 pasos.
- Conserva métricas locales de tiempo y sondeos en un JSON junto al MP4.
- No contiene claves, modelos ni material privado.

## Preparación futura

1. Crear el endpoint ComfyUI solo después de la frase exacta `AUTORIZO RUNPOD`.
2. Instalar los tres modelos oficiales referenciados en el workflow.
3. Copiar `wan/.env.example` a un archivo `.env` local y definir `COMFYUI_URL`.
4. Cambiar `enabled` y `allow_paid_remote` a `true` en `wan/config/wan_poc.json`.
5. Ejecutar:

```bash
python -m wan.generate_video --image /ruta/referencia.png --prompt "movimiento cinematográfico sutil"
```

## Validación sin gasto

```bash
python -m unittest discover -s tests -p 'test_wan_poc.py'
```

La prueba levanta un servidor HTTP simulado y comprueba salud, subida, cola, espera y descarga sin conectarse a RunPod.

## Riesgo de identidad

TI2V anima píxeles; no garantiza identidad, rostro ni vestuario. Para personajes fijos se usará como capa de movimiento y se exigirá una compuerta visual antes de publicar. HeyGen sigue siendo la capa de presentador y sincronización labial.
