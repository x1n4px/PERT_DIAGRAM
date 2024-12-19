import pandas as pd
from graphviz import Digraph

def generar_grafo_pert(relaciones, variables, archivo_salida="graph"):
    """
    Genera un grafo PERT a partir de las relaciones de precedencia usando Graphviz.

    :param relaciones: Lista de relaciones de precedencia como tuplas (origen, destino).
    :param variables: Lista de nombres de las variables.
    :param archivo_salida: Nombre del archivo de salida sin extensión.
    :return: Lista de las relaciones creadas en el formato [(origen, destino, label, estilo)].
    """
    
    print(relaciones)
    print(variables)
    
    # Crear un objeto Digraph
    dot = Digraph(comment="Grafo PERT")

    # Configurar el grafo
    dot.body.extend(
        [
            "rankdir=LR;",  # Dirección de izquierda a derecha
            "node [shape=circle];",  # Los nodos tienen forma circular
            "edge [splines=line];",  # Las aristas son líneas
        ]
    )

    # Lista para registrar las relaciones creadas
    relaciones_creadas = []

    # Diccionario para llevar un control de las flechas hacia cada nodo
    precedencias = {var: [] for var in variables}
    # Mapeo de variables a números
    var_to_num = {var: idx + 1 for idx, var in enumerate(variables)}

    # Contador para las flechas dashed
    dashed_counter = {var: 1 for var in variables}

    # Crear un diccionario para verificar si un nodo está precedido por otro
    nodos_entrantes = {var: False for var in variables}
    
    
    for origen, destino in relaciones:
        nodos_entrantes[destino] = True

    # Nodo 1 será el primer nodo sin flechas entrantes o el primero de la lista
    nodos_sin_precedencia = [
        var for var, tiene_entrantes in nodos_entrantes.items() if not tiene_entrantes
    ]
    nodo_inicio = nodos_sin_precedencia[0] if nodos_sin_precedencia else variables[0]
    
    # Crear nodo de inicio (nodo "1")
    dot.node(str(var_to_num[nodo_inicio]), label=str(var_to_num[nodo_inicio]))
    # Conectar los nodos sin precedencia al nodo "1"
    for var in nodos_sin_precedencia:
        if var != nodo_inicio:  # Evitar que el nodo inicial se conecte a sí mismo
            dot.edge(str(var_to_num[nodo_inicio]), str(var_to_num[var]), label=var)
            relaciones_creadas.append((var_to_num[nodo_inicio], var_to_num[var], var, "solid"))  # Registrar
    
    # Agregar las relaciones entre nodos
    for origen, destino in relaciones:
        # Convertir los nombres de las variables a números
        origen_num = var_to_num[origen]
        destino_num = var_to_num[destino]
        # Determinar si la flecha debe ser dashed
        style = "dashed" if len(precedencias[destino]) > 0 else "solid"

        # Asignar un nombre especial para las flechas dashed
        if style == "dashed":
            label = f"F{dashed_counter[destino]}"
            dashed_counter[destino] += 1
        else:
            for X,Y,B,C in relaciones_creadas:
                if B == origen :
                    label = destino
                    break
                else:
                    label = origen
        
        # Agregar la arista con el label apropiado
        #print(f"Origen {origen_num} -> destino {destino_num} :: label {label}")
        dot.edge(str(origen_num), str(destino_num), label=label, style=style)
        relaciones_creadas.append((origen_num, destino_num, label, style))  # Registrar la relación creada

        # Registrar la relación en el diccionario de precedencias
        precedencias[destino].append(origen)


    # Filtrar nodos que no preceden a otros (es decir, sin flechas salientes)
    nodos_salientes = {var: False for var in variables}

    for origen, destino in relaciones:
        nodos_salientes[origen] = True

    # Filtrar solo los nodos que tienen flechas salientes
    nodos_con_salientes = [
        var for var, tiene_salientes in nodos_salientes.items() if tiene_salientes
    ]

    # Eliminar los nodos sin salientes
    for var in nodos_con_salientes:
        dot.node(str(var_to_num[var]), label=str(var_to_num[var]))

    # Detectar el nodo final (último nodo sin salida)
    nodos_finales = [
        var for var, tiene_salientes in nodos_salientes.items() if not tiene_salientes
    ]
  
    # Crear el nodo final con el número siguiente al último nodo
    if nodos_finales:
        ultimo_numero = max(var_to_num.values())  # Número más alto existente
        nodo_final_num = ultimo_numero + 1  # Nodo final será el siguiente número
        dot.node(str(nodo_final_num), label=str(nodo_final_num))  # Crear nodo final

        # Conectar las variables finales al nuevo nodo final sin bucles
        for var in nodos_finales:
            dot.edge(
                str(var_to_num[var]), str(nodo_final_num), label=var
            )  # Conectar sin bucles
            relaciones_creadas.append((var_to_num[var], nodo_final_num, var, "solid"))  # Registrar

    # Renderizar el grafo
    dot.render(archivo_salida, format="png", cleanup=True)

    # Devolver la lista de relaciones creadas
    return relaciones_creadas




relaciones_precedencia = [('A', 'C'), ('B', 'E'), ('C', 'E'), ('D', 'F')]
variables_filtradas = ['A', 'B', 'C', 'D', 'E', 'F']
dir_name = "output"
generar_grafo_pert(
        relaciones_precedencia, variables_filtradas, archivo_salida=f"{dir_name}/pert_grafo"
    )