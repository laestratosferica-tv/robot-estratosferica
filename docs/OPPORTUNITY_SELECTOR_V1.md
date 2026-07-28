# Selector de oportunidades V1

## Propósito

El selector convierte la regla **menos es más** en una puerta operativa. Después
de verificar y clasificar los candidatos del radar, compara las oportunidades y
permite que solo una avance como experimento editorial controlado.

No publica, transmite, descarga medios, contacta marcas ni usa servicios pagos.

## Cómo decide

Primero exige que el candidato:

- haya superado la evaluación editorial;
- tenga clasificación estratégica completa;
- tenga derechos suficientes para preparar un borrador;
- proponga una métrica distinta de vistas o reproducciones como éxito único.

Después puntúa seis dimensiones:

1. calidad editorial;
2. potencial de conversación;
3. valor explicativo;
4. potencial comercial;
5. relevancia para Latinoamérica;
6. originalidad del ángulo.

Los pesos viven en `config/editorial_v1.json`, suman 100 y se validan al cargar
la configuración. Los desempates son deterministas.

## Resultado

La cola conserva:

- ranking completo de candidatos;
- oportunidad seleccionada;
- explicación de sus señales más fuertes;
- objetivo editorial;
- interacción esperada;
- pregunta y opciones de comunidad;
- hipótesis de audiencia;
- métrica principal.

Solo la oportunidad seleccionada recibe textos y storyboard. Las demás quedan
en el informe para trazabilidad, sin generar piezas adicionales.

## Seguridad

- máximo una oportunidad por corrida;
- estado obligatorio `draft`;
- aprobación humana obligatoria;
- publicación y acciones externas apagadas;
- vistas solas no representan éxito;
- costo por selección: USD 0.

## Reversión

El selector está aislado entre la evaluación editorial y la fábrica. Para
revertirlo se elimina esa etapa y se restaura la creación de paquetes previa,
sin modificar radares, publicadores, credenciales ni workflows.
