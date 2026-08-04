# PromoDetector — bloque automático V1

## Resultado

El bloque `promodetector-block-01-2026-08-03` agrupa ocho fichas para una
sola revisión: cinco comerciales y tres editoriales.

## Regla automática

1. Detectar una oportunidad alineada con las categorías del target.
2. Verificar identidad del producto o servicio, fuente, vigencia y mercado.
3. Exigir HTTPS, destino exacto y divulgación cuando exista monetización.
4. Separar el criterio editorial del incentivo comercial.
5. Crear ficha con utilidad, límites, condiciones y salida segura.
6. Bloquear cualquier oportunidad que no supere la compuerta.
7. Entregar las fichas terminadas por bloques, no una a una.

## Evidencia

- Manifiesto: `artifacts/review/promodetector-v1/promodetector-batch-01.json`.
- Tablero: `artifacts/review/promodetector-v1/batch-review.html`.
- Evidencia Amazon: `artifacts/commercial-evidence/*-amazon.json`.
- Aprobaciones: `artifacts/approval-records/*-v1.json`.

## Límites

- El bloque no autoriza publicación pública.
- Un precio no se muestra si no está verificado para el mercado y momento.
- La disponibilidad en Amazon US no garantiza entrega internacional.
- Una promoción vencida o dudosa se oculta; no se recicla como vigente.
