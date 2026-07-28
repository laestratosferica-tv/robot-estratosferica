# Sprint Estratosférica 3.0 — Fase 1

## Radar de fuentes reales en modo lectura

Esta entrega reemplaza el archivo fechado como única entrada operativa. El
workflow manual consulta feeds RSS aprobados, normaliza las noticias y entrega
los candidatos actuales a la fábrica editorial segura.

## Fuentes iniciales

- Xbox Wire en Español: fuente primaria oficial para gaming y ecosistema Xbox.
- Google Blog para Latinoamérica: fuente primaria oficial en español para IA,
  tecnología e innovación.

El registro conserva fuentes adicionales sin activarlas como feed hasta
confirmar una ruta estable y su política de corroboración.

La ampliación por gaming, tecnología, gastronomía y lujo relacionado con el
target está clasificada en
[`SOURCE_PORTFOLIO_MONETIZATION.md`](SOURCE_PORTFOLIO_MONETIZATION.md). Las
fuentes secundarias con RSS permanecen como candidatas y las fuentes de
gastronomía/lujo como radar manual hasta demostrar suficiente densidad gamer.

## Flujo

```text
feeds aprobados
      |
      v
radar RSS de solo lectura
      |
      v
validación de dominio, fecha y temas bloqueados
      |
      v
Factory V1 en dry run
      |
      v
cola editorial para revisión
```

## Controles

- Ejecución únicamente manual.
- Sin horarios.
- Solo solicitudes de lectura a feeds públicos.
- Sin descarga de imágenes, audio o video.
- Sin tokens ni secretos.
- Sin llamadas a OpenAI o servicios pagos.
- Sin publicaciones ni escrituras externas.
- Bloqueo de apuestas, casino y temas equivalentes.
- Rechazo de fuentes, dominios y fechas no permitidos.
- Identificador estable por fuente y URL para detectar duplicados.
- Priorización editorial explicable por señales de IA, gaming, plataformas y
  ecosistema; promociones y sorteos pierden prioridad.

## Resultado

El workflow genera:

- `live_candidates.json`: candidatos actuales normalizados.
- `live-radar-report.json`: salud, rechazos y costo del escaneo.
- `live-editorial-queue.json`: cola editorial segura.
- reportes de seguridad y salud del coordinador.

## Reversión

Revertir esta entrega elimina el workflow y los campos de feed del registro.
No modifica las credenciales, los publicadores ni la línea base de seguridad.
