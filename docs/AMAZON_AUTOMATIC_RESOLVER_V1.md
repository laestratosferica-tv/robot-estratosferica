# Resolvedor automático Amazon V1

## Regla operativa

Ninguna pieza comercial entra en la cola por un enlace escrito a mano o por un
ASIN de un prototipo. El resolvedor consulta Amazon PA-API 5.0, exige una única
coincidencia de producto, imagen primaria HTTPS, disponibilidad inmediata,
marketplace y associate tag correctos, divulgación y cinco rutas nativas.

## Única configuración pendiente

Crear el secreto de GitHub Actions `AMAZON_PRODUCT_API_CONFIG` con un JSON:

```json
{
  "access_key": "...",
  "secret_key": "...",
  "associate_tag": "...",
  "marketplace": "www.amazon.com",
  "host": "webservices.amazon.com",
  "region": "us-east-1"
}
```

El diagnóstico solo informa si cada campo existe; nunca muestra valores.

## Mini Mic — 3 de agosto, 11:00 p. m. Colombia

La solicitud está en `config/amazon_commercial_requests_v1.json`, pero **no está
en la cola de publicación**. Sin el perfil anterior y una respuesta real única
de Amazon queda bloqueada automáticamente. No se solicita un enlace manual.

## Diagnóstico seguro

```bash
python3 tools/resolve_amazon_commercial.py \
  --request config/amazon_commercial_requests_v1.json \
  --output artifacts/amazon-config-diagnostic.json --diagnostic
```

Resolución de Mini Mic (solo después de configurar el secreto):

```bash
python3 tools/resolve_amazon_commercial.py \
  --request config/amazon_commercial_requests_v1.json \
  --request-id mini-mic-2026-08-03-2300 \
  --output artifacts/amazon-mini-mic-resolution.json
```

Para resolver una solicitud individual se pasa un JSON con `request_id`,
`query` y `expected_terms`. La salida solo puede conectarse a la cola cuando
`publishable` y todas las verificaciones sean verdaderas.
