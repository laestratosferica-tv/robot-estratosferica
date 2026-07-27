# Sprint Estratosférica 3.0 — Fase 1

## Coordinador manual y salud operativa

Esta entrega convierte la línea base de seguridad en un coordinador único y
observable. Continúa sin horarios, publicación, escrituras externas ni
generación paga.

## Flujo

```text
comprobar seguridad
        |
        v
Factory V1 en dry run
        |
        v
validar cola de revisión
        |
        v
inventariar presencia de credenciales
        |
        v
reporte único de salud
```

`phase1_coordinator.py` genera cuatro artefactos:

- `operations-safety.json`: compuertas y workflows.
- `editorial_queue.json`: candidatos en borrador o revisión.
- `platform-readiness.json`: presencia de configuración por plataforma.
- `coordinator-health.json`: resumen integral de la ejecución.

`phase1_acceptance.py` ejecuta el coordinador cinco veces seguidas con el
registro de fuentes obligatorio. La aceptación exige:

- cinco ejecuciones saludables;
- una cola idéntica y reproducible;
- cero publicaciones y cero publicaciones duplicadas;
- cero operaciones facturables;
- costo medido de USD 0.00.

El resultado se guarda en `phase1-acceptance.json`.

El inventario de credenciales recibe únicamente indicadores booleanos. No
recibe, muestra, valida por red ni almacena tokens.

## Threads

`threads-auth-check.yml` separa dos operaciones:

- `validate_existing`, predeterminada: valida el token existente con una
  solicitud de solo lectura.
- `prepare_rotation`, explícita: prepara un token nuevo y lo entrega cifrado,
  sin publicar ni reemplazar secretos automáticamente.

Esta separación evita que un diagnóstico intente reutilizar por defecto un
código OAuth temporal.

## Límites preservados

- Ejecución únicamente manual.
- Cero publicaciones y cero escrituras externas.
- Cero generación paga.
- Fuentes registradas y verificables obligatorias en el coordinador.
- Costo y operaciones facturables incluidos en el reporte de salud.
- No se cambian tokens, permisos ni credenciales.
- Las brechas de credenciales se informan; no bloquean el dry run editorial.
- Una renovación de Threads requiere seleccionar explícitamente
  `prepare_rotation`.

## Reversión

Revertir esta entrega restaura el workflow manual de Fase 0. No es necesario
reactivar workflows heredados ni cambiar secretos.
