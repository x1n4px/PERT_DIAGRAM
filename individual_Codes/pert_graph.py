def construir_grafico_pert(entrada):
    # 1. Crear conjuntos de nodos y actividades
    nodos = set()
    actividades = []
    
    for (origen, destino, tag, tipo) in entrada:
        nodos.add(origen)
        nodos.add(destino)
        actividades.append((origen, destino, tag, tipo))
    
    # Identificar actividades sin precedencia
    nodos_sin_precedencia = {destino for _, destino, _, _ in actividades if destino not in nodos}
    
    # Nodo inicial común para actividades sin precedencias
    nodo_inicial_comun = (1, 2, 'Start', 'solid')
    
    # Nodo final común para actividades que no son precedidas por ninguna
    nodo_final_comun = (len(nodos) + 1, len(nodos) + 2, 'End', 'solid')
    
    # 2. Construir las relaciones totales
    relaciones_totales = []

    # Relaciona actividades con sus precedencias
    for origen, destino, tag, tipo in actividades:
        relaciones_totales.append((origen, destino, tag, tipo))
    
    # Agregar el nodo inicial común a actividades sin precedencias
    for nodo in nodos_sin_precedencia:
        relaciones_totales.append(nodo_inicial_comun[:2] + (nodo, nodo_inicial_comun[2], nodo_inicial_comun[3]))

    # Agregar el nodo final común a actividades que no son precedidas por ninguna
    relaciones_totales.append(nodo_final_comun)
    
    return relaciones_totales

# Ejemplo de uso
entrada = [('A', 'B', 'Activity 1', 'solid'), ('A', 'D', 'Activity 2', 'solid'), ('C', 'D', 'Activity 3', 'solid')]
relaciones_totales = construir_grafico_pert(entrada)
print(relaciones_totales)
