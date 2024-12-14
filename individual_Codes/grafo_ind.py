def grap(relaciones):
    # Imprimir las relaciones para verlas
    print("Relaciones:", relaciones)

    # Variables para almacenar actividades que no preceden a otras y que no son precedidas por nadie
    no_preceden = set()  # Actividades que no son origen
    no_son_precedidas = set()  # Actividades que no son destino

    # Crear conjuntos para determinar origen y destino
    origenes = set(r[0] for r in relaciones)
    destinos = set(r[1] for r in relaciones)

    # Identificar actividades que no preceden y no son precedidas
    no_preceden = destinos - origenes  # Actividades que no son origen en ninguna relación
    no_son_precedidas = origenes - destinos  # Actividades que no son destino en ninguna relación

    print("No preceden (no son origen):", no_preceden)
    print("No son precedidas (no son destino):", no_son_precedidas)

    # Crear una lista de datos iniciales para la estructura (origen, destino, tag, 'solid')
    datos = []
    nodos_intermedios = {}  # Para manejar convergencias a un mismo destino

    # Agregar las actividades que no son precedidas con origen 1 y destino incremental
    index_counter = 2  # Variable para controlar el índice incremental
    for actividad in no_son_precedidas:
        datos.append((1, index_counter, actividad, 'solid'))
        index_counter += 1

    print("Datos iniciales:", datos)

    # Recorrer las relaciones para expandir la lista de datos
    for origen, destino in relaciones:
        # Buscar en datos el elemento cuyo tag coincide con el origen de la relación
        for item in datos:
            if item[2] == origen:
                nuevo_origen = item[1]  # El destino del item actual será el origen del nuevo
                # Si el destino ya tiene un nodo intermedio, usarlo
                if destino in nodos_intermedios:
                    nodo_intermedio = nodos_intermedios[destino]
                else:
                    # Crear un nodo intermedio si hay más de un origen para el mismo destino
                    nodo_intermedio = index_counter
                    nodos_intermedios[destino] = nodo_intermedio
                    index_counter += 1
                
                datos.append((nuevo_origen, nodo_intermedio, destino, 'solid'))
                break

    # Crear un nodo final único
    nodo_final = index_counter
    index_counter += 1

    # Conectar todos los nodos que no preceden a nada al nodo final
    nodos_sin_salida = {destino for _, destino, _, _ in datos} - {origen for origen, _, _, _ in datos}
    for nodo in nodos_sin_salida:
        datos.append((nodo, nodo_final, "", 'solid'))

    # Mostrar cada caso de la lista en el formato específico
    for nuevo_origen, destino, label, _ in datos:
        print(f"{nuevo_origen} -> {destino} [label=\"{label}\"];")

# Relaciones de ejemplo
relaciones = [('A', 'C'), ('B', 'E'), ('C', 'E'), ('D', 'F')]

grap(relaciones)
