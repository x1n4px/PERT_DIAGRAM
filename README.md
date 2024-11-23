# PERT_DIAGRAM

## Formato de entrada:
| | A | B | C | D | E |
| --- | --- | ---| --- | --- | --- |
| to | 2 | 1 |5 | 4| 2|
| tm |3 |2 |7 |5 | 6|
| tp |4 |4 |10 |7 |10 |
| | | | | | 
| | A | B | C | D | E |
| A | | 1 | | | |
| B | | | | 1 | |
| C | | | | | |
| D | | | | 1 | |
| E | | | | | |


## Salida:
- output_graph.png
- output_pert.txt (texto en formato latex)
-   tabla con early y last de cada nodo
    - tabla con holguras de cada tarea
     - camino crítico
- EXTRA: diagrama de Gantt (output_gantt.png)
