# PERT_DIAGRAM

## Descripción
**PERT_DIAGRAM** es una herramienta diseñada para procesar datos desde un archivo Excel y generar tanto el diagrama PERT en formato gráfico como la representación en LaTeX. Este sistema permite analizar la planificación de proyectos, calculando tiempos de ejecución, holguras y caminos críticos.

## Formato de Entrada
El archivo de entrada debe contener dos secciones principales:

1. **Tiempos de las tareas**:

   |   | A | B | C | D | E |
   |---|---|---|---|---|---|
   | to | 2 | 1 | 5 | 4 | 2 |
   | tm | 3 | 2 | 7 | 5 | 6 |
   | tp | 4 | 4 | 10 | 7 | 10 |

   - **to**: Tiempo optimista
   - **tm**: Tiempo más probable
   - **tp**: Tiempo pesimista

2. **Relaciones entre tareas** (matriz de precedencias):

   |   | A | B | C | D | E |
   |---|---|---|---|---|---|
   | A |   | 1 |   |   |   |
   | B |   |   |   | 1 |   |
   | C |   |   |   |   |   |
   | D |   |   |   | 1 |   |
   | E |   |   |   |   |   |

## Salida
El sistema genera los siguientes archivos de salida:

- **Diagrama PERT en formato gráfico**: `output_graph.png`
- **Representación en LaTeX**: `output_pert.txt`
- **Tablas de análisis**:
  - Tiempos de inicio temprano y tardío por nodo.
  - Holguras de cada tarea.
  - Identificación del **camino crítico**.
- **Extras**:
  - **Diagrama de Gantt** (`output_gantt.png`) para visualizar la planificación temporal del proyecto.

## Uso
Este programa es útil para la gestión y planificación de proyectos, permitiendo analizar el flujo de trabajo, detectar tareas críticas y optimizar los tiempos de ejecución.

