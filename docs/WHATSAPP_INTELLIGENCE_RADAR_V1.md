# Radar de Inteligencia por WhatsApp

## Propósito

Recibir señales enviadas por el equipo y convertirlas en candidatos trazables para
investigación editorial o mejora del producto. No publica contenido, no descarga
vídeos de TikTok y no incorpora conocimiento como verdad sin verificación.

## Categorías de entrada

| Prefijo | Destino |
| --- | --- |
| `NOTICIA` | Investigación editorial |
| `TENDENCIA` | Radar de tendencias |
| `PERSONAJE` | Banco creativo de personajes |
| `EFECTO` | Biblioteca de efectos visuales |
| `ROBOT` | Backlog de conocimiento o comportamiento del robot |
| `HERRAMIENTA` | Radar tecnológico |
| `APRENDIZAJE` | Base de aprendizajes |
| `IDEA` | Bandeja de triaje |

Ejemplo: `PERSONAJE https://… Robot archivista con casco de neón; podría presentar datos semanales.`

## Contrato seguro

- Endpoint: `GET` y `POST /webhooks/whatsapp`.
- Meta verifica el `GET` usando `WHATSAPP_WEBHOOK_VERIFY_TOKEN`.
- Todo `POST` requiere `x-hub-signature-256`, verificado con `WHATSAPP_APP_SECRET`.
- La identidad del remitente se guarda como hash; no se almacena su número.
- La pieza queda en `received`, con derechos y conocimiento `not_verified`.
- No existen rutas de publicación ni aprobación automática.

## Activación posterior (requiere aprobación del director)

1. El despliegue crea automáticamente un namespace KV dedicado mediante el binding `RADAR_KV`.
2. Crear los secretos de GitHub `WHATSAPP_RADAR_APP_SECRET` y `WHATSAPP_RADAR_VERIFY_TOKEN`; nunca versionarlos.
3. Ejecutar manualmente `Deploy WhatsApp Intelligence Radar`.
4. Registrar `https://<worker>/webhooks/whatsapp` en Meta y completar el reto de verificación.
5. Suscribir únicamente el campo `messages` y probar con un mensaje `IDEA`.
6. Confirmar que la señal queda en `received`, sin publicación ni aprobación automática.

La activación, despliegue y conexión de una cuenta real requieren autorización
explícita: son cambios de producción y de mensajería externa.
