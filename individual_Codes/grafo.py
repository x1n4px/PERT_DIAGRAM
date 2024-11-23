import pandas as pd
from graphviz import Digraph

def leer_matriz_dependencias(archivo_excel):
    """
    Lee la matriz de dependencias desde un archivo Excel y extrae las relaciones de precedencia.
    
    :param archivo_excel: Ruta del archivo Excel.
    :return: Lista de relaciones de precedencia.
    """
    # Leer el archivo Excel sin encabezados
    df = pd.read_excel(archivo_excel, header=None)
    
    # Leer la matriz de dependencias (de A5 a P20)
    matriz = df.iloc[4:20, 0:16]  # Rango de A5:P20 (indexado desde 0 en pandas)
    
    # La primera fila y columna contienen los nombres de las variables
    variables = matriz.iloc[0, 1:].tolist()  # Variables de la primera fila (B5:P5)
    matriz = matriz.iloc[1:, 1:].values      # Valores numéricos de la matriz (sin encabezados)

    # Crear una lista para almacenar las relaciones de precedencia
    relaciones = []

    # Recorrer la matriz y capturar solo valores por encima de la diagonal principal
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            if matriz[i][j] == 1:  # Hay una relación de precedencia
                relaciones.append((variables[i], variables[j]))  # Relación como tupla

    return relaciones, variables

def generar_grafo_pert(relaciones, variables, archivo_salida='graph'):
    """
    Genera un grafo PERT a partir de las relaciones de precedencia usando Graphviz.
    
    :param relaciones: Lista de relaciones de precedencia como tuplas (origen, destino).
    :param variables: Lista de nombres de las variables.
    :param archivo_salida: Nombre del archivo de salida sin extensión.
    """
    # Crear un objeto Digraph
    dot = Digraph(comment='Grafo PERT')
    
    # Configurar el grafo
    dot.body.extend([
        'rankdir=LR;',  # Dirección de izquierda a derecha
        'node [shape=circle];',  # Los nodos tienen forma circular
        'edge [splines=line];'   # Las aristas son líneas
    ])
    
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
    nodos_sin_precedencia = [var for var, tiene_entrantes in nodos_entrantes.items() if not tiene_entrantes]
    nodo_inicio = nodos_sin_precedencia[0] if nodos_sin_precedencia else variables[0]

    # Crear nodo de inicio (nodo "1")
    dot.node(str(var_to_num[nodo_inicio]), label=str(var_to_num[nodo_inicio]))
    
    # Agregar las relaciones entre nodos
    for origen, destino in relaciones:
        # Convertir los nombres de las variables a números
        origen_num = var_to_num[origen]
        destino_num = var_to_num[destino]
        
        # Determinar si la flecha debe ser dashed
        style = 'dashed' if len(precedencias[destino]) > 0 else 'solid'
        
        # Asignar un nombre especial para las flechas dashed
        if style == 'dashed':
            label = f"F{dashed_counter[destino]}"
            dashed_counter[destino] += 1
        else:
            label = origen
        
        # Agregar la arista con el label apropiado
        dot.edge(str(origen_num), str(destino_num), label=label, style=style)


        # Registrar la relación en el diccionario de precedencias
        precedencias[destino].append(origen)
    
    # Conectar los nodos sin precedencia al nodo "1"
    for var in nodos_sin_precedencia:
        if var != nodo_inicio:  # Evitar que el nodo inicial se conecte a sí mismo
            dot.edge(str(var_to_num[nodo_inicio]), str(var_to_num[var]), label=var)
    
    # Filtrar nodos que no preceden a otros (es decir, sin flechas salientes)
    nodos_salientes = {var: False for var in variables}
    
    for origen, destino in relaciones:
        nodos_salientes[origen] = True
    
    # Filtrar solo los nodos que tienen flechas salientes
    nodos_con_salientes = [var for var, tiene_salientes in nodos_salientes.items() if tiene_salientes]
    
    # Eliminar nodos sin precedencia de la lista
    nodos_con_salientes_num = [var_to_num[var] for var in nodos_con_salientes]
    
    # Eliminar los nodos sin salientes
    for var in nodos_con_salientes:
        dot.node(str(var_to_num[var]), label=str(var_to_num[var]))
    
    # Detectar el nodo final (último nodo sin salida)
    nodos_finales = [var for var, tiene_salientes in nodos_salientes.items() if not tiene_salientes]

    # Crear el nodo final con el número siguiente al último nodo
    if nodos_finales:
        ultimo_numero = max(var_to_num.values())  # Número más alto existente
        nodo_final_num = ultimo_numero + 1  # Nodo final será el siguiente número
        dot.node(str(nodo_final_num), label=str(nodo_final_num))  # Crear nodo final

        # Conectar las variables finales al nuevo nodo final sin bucles
        for var in nodos_finales:
            dot.edge(str(var_to_num[var]), str(nodo_final_num), label=var)  # Conectar sin bucles

    # Renderizar el grafo
    dot.render(archivo_salida, format='png', cleanup=True)
    print(f"Grafo generado y guardado como {archivo_salida}.png")

# Uso del programa
archivo_excel = "input2.xlsx"  # Cambia esto por la ruta de tu archivo Excel

# Leer las relaciones de precedencia
relaciones_precedencia, variables = leer_matriz_dependencias(archivo_excel)
# Crear un conjunto con todos los elementos únicos de las tuplas en lista_B
elementos_en_B = set(item for tupla in relaciones_precedencia for item in tupla)

# Filtrar la lista A para mantener solo los elementos que están en el conjunto
resultado = [elemento for elemento in variables if elemento in elementos_en_B]

# Generar el grafo PERT
generar_grafo_pert(relaciones_precedencia, resultado, archivo_salida='pert_grafo')
