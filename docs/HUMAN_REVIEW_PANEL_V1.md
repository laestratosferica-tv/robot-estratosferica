# Panel de revisión humana V1

## Propósito

El panel convierte la cola segura en una decisión sencilla para José Luis:
revisar la fuente, el objetivo, la interacción, la métrica, la pregunta y los
textos; después aprobar editorialmente o rechazar la pieza.

La aprobación editorial **no publica**. Solo crea una decisión local y
auditable. `publish_allowed`, `publishing_enabled` y
`external_actions_enabled` permanecen en `false`.

## Uso local

Primero se genera `artifacts/editorial_queue.json` con el coordinador habitual.
Después:

```bash
python review_panel.py
```

El panel queda disponible únicamente en `http://127.0.0.1:8765`.

Opciones:

```bash
python review_panel.py \
  --queue artifacts/editorial_queue.json \
  --decisions artifacts/editorial_decisions.json \
  --port 8765
```

## Contenido visible

- historia y resumen;
- fuente original;
- producto editorial;
- objetivo e interacción esperada;
- pregunta y opciones de comunidad;
- métrica principal;
- puntaje y razones del selector;
- textos preparados por plataforma.

## Decisiones

- `approved_editorially`: la idea puede pasar al siguiente control, pero no
  concede permiso de publicación;
- `rejected`: exige un motivo para alimentar el aprendizaje editorial.

Las decisiones se guardan de forma atómica en
`artifacts/editorial_decisions.json`. Una misma pieza no puede decidirse dos
veces.

## Seguridad

El panel se niega a iniciar si la cola:

- no está en `dry_run`;
- intenta habilitar publicación o acciones externas;
- contiene más de una pieza;
- omite la aprobación humana;
- incluye una pieza con permiso de publicación.

Por defecto escucha solo en `127.0.0.1`, no en una interfaz pública.

## Reversión

Eliminar `review_panel.py`, su prueba y esta guía. El radar, el selector, la
cola y los publicadores no se modifican.
