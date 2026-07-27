# Sprint Estratosférica 3.0 — Fase 0

## Línea base de seguridad operativa

Esta fase pone la casa en orden antes de conectar noticias reales, R2/Drive,
publicadores o medición automática.

## Estado alcanzado

- `factory-v1-dry-run.yml` es el único coordinador editorial.
- El coordinador se ejecuta únicamente de forma manual.
- Los doce workflows heredados quedan en cuarentena reversible.
- Ningún workflow conserva un horario automático.
- Los publicadores heredados exigen dos compuertas:
  `LEGACY_WORKFLOWS_ARMED` y `PRODUCTION_ARMED`.
- Las redes, Runway, escrituras externas y generación paga permanecen apagadas.
- Threads usa `auto_post=false`, límite cero y `dry_run=true`.
- El análisis en vivo de comunidad requiere activación explícita.
- La comprobación OAuth de Threads deja de ejecutarse automáticamente en PR o
  `push`; queda solo como diagnóstico manual.

## Arquitectura operativa

```text
candidato -> Factory V1 -> control -> cola de revisión -> artefacto
```

No existe conexión con distribuidores durante esta fase.

## Recuperación de componentes heredados

Los modos A–Z no se eliminan. Permanecen como referencia y podrán extraerse
por componentes cuando exista una prueba que demuestre:

1. fuente y derechos verificables;
2. ruta canónica;
3. límite de costo;
4. ejecución en `dry_run`;
5. una sola salida controlada.

## Reversión

El cambio se revierte restaurando los workflows anteriores. Reactivar horarios,
credenciales, costos o publicaciones requiere una decisión separada y aprobación
expresa de José Luis.
