# Portal público de creadores TikTok — Sandbox V1

## Objetivo

Demostrar que la integración de La Estratosférica es un producto para creadores
externos y no una automatización interna. El creador conecta su propia cuenta,
selecciona un video propio y lo envía a sus borradores. La publicación final
siempre ocurre dentro de TikTok y por decisión del creador.

## Recorrido real

1. El creador abre el portal público.
2. Selecciona **Conectar TikTok**.
3. TikTok solicita `user.info.basic` y `video.upload`.
4. El callback valida un `state` firmado y consulta la identidad básica.
5. El creador selecciona un MP4 o MOV propio de hasta 50 MB.
6. El portal inicia `FILE_UPLOAD` mediante el endpoint de inbox.
7. TikTok recibe el archivo como borrador; el portal no usa Direct Post.
8. El creador abre TikTok, revisa, edita y decide si publica.

## Seguridad predeterminada

La aplicación inicia con la transferencia bloqueada:

```text
ENABLE_TIKTOK_DRAFT_TRANSFER=false
TIKTOK_SANDBOX_REVIEW_MODE=false
ENABLE_TIKTOK=false
ENABLE_TIKTOK_PUBLISH=false
```

`video.publish` no se solicita ni se implementa. Para una única demostración
sandbox se requieren simultáneamente:

```text
ENABLE_TIKTOK_DRAFT_TRANSFER=true
TIKTOK_SANDBOX_REVIEW_MODE=true
```

La activación es temporal y requiere aprobación específica del director. En
producción, la transferencia solo se permite cuando
`TIKTOK_APP_REVIEW_STATUS=approved`.

## Variables

Secretos del entorno:

- `TIKTOK_CLIENT_KEY`
- `TIKTOK_CLIENT_SECRET`
- `TIKTOK_SESSION_SECRET`

Configuración no secreta:

- `TIKTOK_REDIRECT_URI`
- `PUBLIC_BASE_URL`
- `TIKTOK_APP_REVIEW_STATUS`
- `ENABLE_TIKTOK_DRAFT_TRANSFER`
- `TIKTOK_SANDBOX_REVIEW_MODE`

No se guardan tokens en archivos, logs ni respuestas. V1 conserva las sesiones
en memoria y debe usar almacenamiento cifrado antes de escalar a múltiples
instancias.

## Ejecución local segura

```bash
uvicorn tiktok_creator_portal:app --host 127.0.0.1 --port 8000
```

Sin variables OAuth, la interfaz puede revisarse pero el botón de conexión queda
bloqueado. `/health` confirma que Direct Post está apagado.

## Evidencia para la próxima revisión

El nuevo video debe grabar sin cortes:

1. Dominio público visible en la barra del navegador.
2. Portada que explica el servicio para creadores externos.
3. Inicio de OAuth desde el portal.
4. Consentimiento sandbox de TikTok.
5. Regreso al mismo dominio con la cuenta conectada.
6. Selección de un video propio.
7. Envío a borradores.
8. Aparición del borrador dentro de la aplicación TikTok.

No se debe reenviar la solicitud hasta que ese recorrido funcione realmente.

