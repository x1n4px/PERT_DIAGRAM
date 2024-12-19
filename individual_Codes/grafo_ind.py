import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph


def draw_pert_diagram(relaciones, variables, archivo_salida="graph"):
    dot = Digraph(comment="Grafo PERT")
    print("relaciones: ", relaciones)
    # Configurar el grafo
    dot.body.extend([
        "rankdir=LR;",  # Dirección de izquierda a derecha
        "node [shape=circle];",  # Los nodos tienen forma circular
        "edge [splines=line];",  # Las aristas son líneas
    ])

    # Crear un conjunto con todos los nodos destino
    destinos = {destino for _, destino in relaciones}
    # Identificar nodos iniciales (no están como destino en ninguna relación)
    nodos_iniciales = [origen for origen, _ in relaciones if origen not in destinos]
    nodos_iniciales = list(set(nodos_iniciales))
    print("Nodos iniciales: ", nodos_iniciales)
    resultado = []
    nodo_actual = 2  # Partimos del nodo 2 como destino inicial

    # Agregar los nodos iniciales a la lista de resultados
    for nodo in nodos_iniciales:
        resultado.append((1, nodo_actual, nodo, 'solid'))
        nodo_actual += 1

    # Procesar las relaciones
    aux_relaciones = relaciones.copy()
    nuevas_relaciones = []
    print("Resultados iniciales: ", resultado)
    for _, nodo_destino, actividad, _ in resultado:
        for origen, destino in aux_relaciones[:]:
            if origen == actividad:
                nuevas_relaciones.append((nodo_destino, nodo_actual, destino, 'solid'))
                nodo_actual += 1
                aux_relaciones.remove((origen, destino))

    resultado.extend(nuevas_relaciones)

    # Crear un diccionario de relaciones ya creadas para evitar duplicados
    relaciones_creadas = {actividad: destino for _, destino, actividad, _ in resultado}

    finales_dashed = []
    # Buscar relaciones finales dashed
    for origen, destino in aux_relaciones:
        for act_origen, act_destino, actividad, sol in resultado:
            if actividad == origen:
                act_destino_rel = act_destino
            elif actividad == destino:
                act_origen_rel = act_origen

        if 'act_destino_rel' in locals() and 'act_origen_rel' in locals():
            finales_dashed.append((act_destino_rel, act_origen_rel, 'F1', 'dashed'))

    resultado.extend(finales_dashed)

    # Encontrar el destino más pequeño entre nodos no precedidos
    nodos_no_precedidos = {destino for _, destino in relaciones} - {origen for origen, _ in relaciones}
    min_destino = min((destino for origen, destino, actividad, sol in resultado if actividad in nodos_no_precedidos), default=None)

    # Modificar los destinos para los nodos no precedidos
    nuevo_resultado = [(origen, destino if actividad not in nodos_no_precedidos else min_destino, actividad, sol) for origen, destino, actividad, sol in resultado]

    # Reemplazar la lista original con la lista modificada
    resultado = nuevo_resultado

    # Diccionario de variables a números
    var_to_num = {var: idx + 1 for idx, var in enumerate(variables)}

    # Agregar las aristas al grafo
    for origen, destino, actividad, sol in resultado:
        dot.edge(str(origen), str(destino), label=actividad, style=sol)

    # Imprimir las relaciones finales
    print("Relaciones finales:", resultado)

    # Guardar el grafo en archivo de salida
    dot.render(archivo_salida, format="png", cleanup=True)


# Ejemplo de relaciones
relaciones = [('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'E')]

variables = ['A', 'B', 'C', 'D', 'E']
# Dibujar el diagrama
draw_pert_diagram(relaciones, variables)


