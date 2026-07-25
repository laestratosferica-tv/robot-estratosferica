# Arquitectura V1 — Un coordinador, una cola, una decisión

## Objetivo

Reemplazar los múltiples workflows independientes por un circuito central,
observable y seguro. La V1 genera paquetes editoriales para revisión; no
publica automáticamente.

## Componentes

### 1. Radar

Consulta fuentes permitidas, normaliza eventos y conserva:

- título y URL;
- fuente y fecha;
- territorio editorial;
- país o región;
- señales de interés;
- evidencia disponible.

No descarga ni republica videos de terceros.

### 2. Editor

Puntúa cada candidato sobre 100:

- relevancia LATAM: 25;
- valor explicativo: 20;
- originalidad del ángulo: 20;
- verificabilidad: 15;
- potencial de conversación: 10;
- potencial comercial: 10.

Rechaza candidatos sin fuente, duplicados, rumores no confirmados y contenido
que dependa de material sin derechos.

### 3. Estudio

Produce un paquete, no una publicación:

- formato editorial;
- titular;
- resumen factual;
- guion;
- caption por plataforma;
- instrucciones visuales;
- fuentes;
- riesgos;
- oportunidad comercial opcional.

### 4. Control

Valida:

- fuentes y fechas;
- derechos y atribución;
- afirmaciones no sustentadas;
- repetición;
- tono;
- límites de palabras y duración;
- costo estimado.

### 5. Cola de aprobación

Estados permitidos:

- `draft`;
- `needs_review`;
- `approved`;
- `rejected`;
- `published`;
- `failed`.

La V1 solo puede crear `draft` y `needs_review`.

### 6. Distribuidor

Permanece desconectado durante el piloto. Más adelante adaptará una pieza
aprobada a Instagram, Facebook, YouTube y Threads. TikTok queda fuera de V1.

### 7. Aprendizaje

Registra decisiones y desempeño sin optimizar únicamente por vistas:

- aprobación o rechazo y motivo;
- formato y tema;
- retención y compartidos;
- guardados y comentarios útiles;
- costo y tiempo;
- señal comercial.

## Ejecución

Un único workflow manual ejecuta:

`radar -> editor -> estudio -> control -> cola`

El workflow:

- usa `DRY_RUN=true`;
- procesa máximo tres candidatos;
- produce máximo un paquete final;
- no recibe tokens sociales;
- no ejecuta publicación;
- no usa Runway por defecto;
- guarda artefactos para inspección.

## Estructura propuesta

```text
media_factory/
  __init__.py
  cli.py
  config.py
  models.py
  radar.py
  editor.py
  studio.py
  guardrails.py
  queue.py
  commercial.py
config/
  editorial_v1.json
tests/
  test_editorial_v1.py
.github/workflows/
  factory-v1-dry-run.yml
```

## Migración desde el robot anterior

Se reutilizará por extracción, no por conexión directa:

- clientes R2;
- memoria antirrepetición;
- adaptadores de publicación;
- utilidades de video verificadas;
- autenticación de redes cuando se reactive.

Los modos A–Z permanecerán desactivados como referencia hasta demostrar que la
V1 cubre sus funciones esenciales.

## Puertas de autonomía

### Puerta 1 — Paquete local

Genera y valida contenido sin subirlo ni publicarlo.

### Puerta 2 — Cola privada

Guarda paquetes revisables en R2.

### Puerta 3 — Publicación manual

Publica una pieza aprobada en una sola red.

### Puerta 4 — Publicación programada

Publica una pieza diaria con reversión y alertas.

### Puerta 5 — Autonomía editorial

Solo se habilita después de medir calidad, costo, seguridad y resultados.
