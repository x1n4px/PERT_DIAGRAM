from graphviz import Digraph

def create_pert_list(relaciones, variables, archivo_salida="graph"):
    """
    Crea una lista PERT a partir de una relación de precedencia.

    Args:
        relations: Lista de tuplas (origen, destino) representando las relaciones.

    Returns:
        Lista de tuplas (origen, destino, tarea, estilo) representando los nodos.
    """
    dot = Digraph(comment="Grafo PERT")
    # Configurar el grafo
    dot.body.extend([
        "rankdir=LR;",  # Dirección de izquierda a derecha
        "node [shape=circle];",  # Los nodos tienen forma circular
        "edge [splines=line];",  # Las aristas son líneas
    ])

    letras_en_tuplas = set(elemento for tupla in relaciones for elemento in tupla)
    letras_faltantes = set(variables) - letras_en_tuplas

    #print("Letras: ", letras_faltantes)
     # Crear un conjunto con todos los nodos destino
    destinos = {destino for _, destino in relaciones}

    preceding = set()
    non_preceding = set()

    for origin, destination in relaciones:
        preceding.add(destination)
        non_preceding.add(origin)
    
    nodos_iniciales = sorted(list(set([origen for origen, _ in relaciones if origen not in destinos])))
    nodos_finales = sorted(list(set(preceding - non_preceding)))
    #print(nodos_iniciales)
    #print(nodos_finales)
    
    i = 2
    resultado = []
    
    for x in nodos_iniciales:
        resultado.append((1, i, x, 'solid'))
        i += 1
        
    repetidos_y = []
    vistos_y = set()

    for x, y in relaciones:
        if y in vistos_y:
            repetidos_y.append((x, y))
        vistos_y.add(y)
        
    # Convertimos 'aux' en un conjunto para búsquedas más eficientes
    conjunto_aux = set(repetidos_y)

    # Creamos una nueva lista para almacenar los elementos a conservar
    nueva_relaciones = []

    for tupla in relaciones:
        if tupla not in conjunto_aux:
            nueva_relaciones.append(tupla)
    relaciones = nueva_relaciones

   
    
    
    
    for X,Y in relaciones:
        for orig, dest, nam, sol in resultado[:]:
            if(nam == X):
                resultado.append((dest, i, Y, 'solid'))
                i += 1
   
        
    
    for X,Y in repetidos_y:
        dest_X = 0
        orig_Y = 0
        for orig, dest, nam, sol in resultado[:]:
            if(nam == X):
                dest_X = dest
            elif nam == Y :
                orig_Y = orig
        resultado.append((dest_X, orig_Y, 'F1', 'dashed'))
        
    # nodos sin relaciones
    for a in letras_faltantes:
        resultado.append((1, i, a, 'solid'))
        nodos_finales.append(a)
    
    
    resultado_filtrado = [tupla for tupla in resultado if tupla[2] in nodos_finales]
    #print(resultado_filtrado)

    # todos los ndos finales, apunten al mismo
    #print(nodos_finales)
    nodo_final = i
    for X,Y,M,S in resultado[:]:
        if(M in nodos_finales):
            if(Y < nodo_final):
                nodo_final = Y
        
    #print("Valor ndo final: ", nodo_final)
    for i, nodo in enumerate(resultado):
        if nodo[2] in nodos_finales:
            #print(nodo[2])
            resultado[i] = (nodo[0], nodo_final, nodo[2], nodo[3])
            #print(resultado[i])
            
    
    for origen, destino, actividad, sol in resultado:
        dot.edge(str(origen), str(destino), label=actividad, style=sol)
        
    dot.render(archivo_salida, format="png", cleanup=True)
    return resultado

# Ejemplo de uso
#relations = [('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'E')]
#variables = ['A','B','C','D','E','F']
#pert_list = create_pert_list(relations, variables)
#print(pert_list)