# TikTok: integración y pruebas seguras

## Estado

TikTok continúa desactivado. Esta etapa prepara autenticación, permisos y
diagnóstico sin cargar ni publicar contenido.

El portal público sandbox para creadores está documentado en
`docs/TIKTOK_PUBLIC_CREATOR_SANDBOX_V1.md`. Su transferencia también permanece
bloqueada de forma predeterminada.

Los dos interruptores deben permanecer en `false`:

- `ENABLE_TIKTOK`
- `ENABLE_TIKTOK_PUBLISH`

## Alcance del diagnóstico

`tiktok_readiness_diagnostic.py`:

1. Comprueba que los interruptores de publicación estén apagados.
2. Comprueba la presencia de la configuración OAuth, sin mostrar valores.
3. Comprueba los permisos declarados.
4. Comprueba que las aprobaciones se hayan marcado explícitamente.
5. Si existe token y permiso `user.info.basic`, realiza únicamente:
   `GET https://open.tiktokapis.com/v2/user/info/`.
6. Compara internamente el `open_id` recibido con el configurado.
7. Produce un JSON redactado y sin información secreta.

El diagnóstico no llama endpoints `/post/publish/`, no sube archivos y no
publica.

## Variables seguras

### GitHub Secrets

- `TIKTOK_CLIENT_KEY`
- `TIKTOK_CLIENT_SECRET`
- `TIKTOK_ACCESS_TOKEN`
- `TIKTOK_REFRESH_TOKEN`
- `TIKTOK_OPEN_ID`

### GitHub Variables

- `TIKTOK_REDIRECT_URI`
- `TIKTOK_AUTHORIZED_SCOPES`
- `TIKTOK_APP_REVIEW_STATUS`
- `TIKTOK_CONTENT_POSTING_API_STATUS`

Valores esperados para las dos variables de estado, únicamente después de
comprobarlos en TikTok Developer Portal:

```text
approved
```

Los permisos declarados deben incluir:

- `user.info.basic` para el diagnóstico de identidad de solo lectura.
- `video.upload` para enviar una pieza al inbox y terminarla manualmente en
  TikTok, o `video.publish` para Direct Post.

Solicitar solo los permisos realmente aprobados. Nunca escribir un permiso en
la variable antes de que TikTok y el usuario lo hayan autorizado.

## OAuth

La aplicación debe usar OAuth v2:

1. Redirigir al usuario a la autorización de TikTok con `state` antifalsificación.
2. Recibir el código en un `redirect_uri` registrado exactamente.
3. Intercambiar el código en el servidor.
4. Guardar `access_token` y `refresh_token` únicamente como secretos.
5. Renovar el access token antes de su vencimiento.
6. Sustituir también el refresh token si TikTok devuelve uno nuevo.
7. Permitir revocación y desconexión.

El diagnóstico no implementa ni ejecuta este intercambio para evitar cambios
externos y exposición accidental.

## Requisitos antes de una prueba privada

- Aplicación aprobada en TikTok Developer Portal.
- Content Posting API aprobada.
- Cuenta objetivo autorizada mediante OAuth.
- Permisos correctos en el token.
- `open_id` validado por el diagnóstico.
- Dominio o prefijo de URL verificado si se usará `PULL_FROM_URL`.
- Una sola pieza aprobada y una prueba privada/manual autorizada.
- Idempotencia, seguimiento de estado y recibo auditable implementados.

## Ejecución

En GitHub Actions:

1. Abrir **TikTok - Manual Read-Only Readiness Diagnostic**.
2. Seleccionar **Run workflow**.
3. Descargar el artefacto `tiktok-readiness-diagnostic`.
4. Revisar `blockers` y `next_action`.

El workflow fuerza ambos interruptores a `false`.

Localmente, sin credenciales:

```bash
python tiktok_readiness_diagnostic.py
```

Esto produce un reporte de bloqueadores sin intentar conexión ni publicación.

## Criterio de avance

`ready_for_private_test: true` significa que la identidad, permisos y
aprobaciones declaradas están listas para diseñar una prueba privada. No
autoriza la prueba ni activa la publicación.

La activación futura requiere una aprobación separada del director del
proyecto.
