import pandas as pd
from graphviz import Digraph
import matplotlib.pyplot as plt
import argparse
import os
from datetime import datetime
import individual_Codes.grp


def leer_y_procesar_excel(archivo_excel):
    """
    Lee un archivo Excel, filtra las variables con valores completos y calcula De y Var para cada variable.

    :param archivo_excel: Ruta del archivo Excel.
    :return: Un DataFrame con las variables, sus valores y los cálculos de De y Var.
    """
    # Leer el archivo Excel sin encabezados
    df = pd.read_excel(archivo_excel, header=None)

    # Extraer los nombres de las variables y sus valores
    variables = df.iloc[
        0, 1:
    ].tolist()  # Fila 1 (nombres de variables), columna B en adelante
    t0 = df.iloc[1, 1:].tolist()  # Fila 2 (t0)
    tm = df.iloc[2, 1:].tolist()  # Fila 3 (tm)
    tp = df.iloc[3, 1:].tolist()  # Fila 4 (tp)

    # Crear un DataFrame con las variables y valores
    datos = pd.DataFrame({"Variable": variables, "t0": t0, "tm": tm, "tp": tp})

    # Filtrar solo las variables con valores completos
    datos = datos.dropna()

    # Calcular De y Var para cada fila
    datos["De"] = (datos["t0"] + 4 * datos["tm"] + datos["tp"]) / 6
    datos["Var"] = ((datos["tp"] - datos["t0"]) / 6) ** 2

    return datos


def exportar_a_latex(datos, archivo_latex):
    """
    Exporta un DataFrame a un archivo LaTeX.

    :param datos: DataFrame con los datos a exportar.
    :param archivo_latex: Ruta del archivo de salida en formato LaTeX.
    """
    # Generar la tabla en formato LaTeX
    tabla_latex = datos.to_latex(index=False, float_format="%.3f")

    # Guardar en un archivo
    with open(archivo_latex, "w") as f:
        f.write(tabla_latex)


########################## GRAFOS ################################
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
    matriz = matriz.iloc[
        1:, 1:
    ].values  # Valores numéricos de la matriz (sin encabezados)
    # Crear una lista para almacenar las relaciones de precedencia
    relaciones = []

    # Recorrer la matriz y capturar solo valores por encima de la diagonal principal
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            if matriz[i][j] == 1:  # Hay una relación de precedencia
                relaciones.append((variables[i], variables[j]))  # Relación como tupla

    return relaciones, variables


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


######################## CALCULO TABLAS #########################


def calcular_late_times(relaciones, duraciones, valores_variables, maxEarlyValue):
    """
    Calcula los tiempos más tempranos (Ei) para cada nodo en un grafo dado.

    :param relaciones: Lista de relaciones [(origen, destino, nombre_variable, _)].
    :param duraciones: Diccionario con las duraciones de cada variable.
    :return: Diccionario con los tiempos más tempranos Ei y lista de cálculos detallados.
    """
    # Inicializar tiempos más tempranos (Ei)
    nodos = set([rel[0] for rel in relaciones] + [rel[1] for rel in relaciones])  # Obtener nodos únicos
    Ei = {nodo: 0 for nodo in nodos}

    valores_variables_dic = {chr(65 + i): round(float(valores_variables[i]),2) for i in range(len(valores_variables))}
    agrupados = {}

    for rel in relaciones:
        i = rel[0]  # El segundo valor de cada relación
        # Si el grupo para este i aún no existe, crear una lista vacía
        if i not in agrupados:
            agrupados[i] = []
        
        # Agregar la relación completa a la lista correspondiente al índice i
        agrupados[i].append(rel)
    
    agrupados = sorted(list(agrupados.items()), key=lambda x:x[0])
 
    # Listas para almacenar el cálculo y el resultado
    calculos_texto = []
    valores_finales_previos = [0]*len(nodos)
    valores_finales_previos[-1] = maxEarlyValue
    valores_finales_previos
    
    calculos_texto.append(f"L{len(nodos)} = {maxEarlyValue}")
    agrupados.reverse()
    
    for i, relaciones in agrupados:
        if len(relaciones) == 1:
            (X1, Y1, variable1, _) = relaciones[0]
            M1 = valores_variables_dic.get(variable1, 0)
            VX1 = valores_finales_previos[Y1-1]
            resultado = VX1 - M1
            valores_finales_previos[X1-1] = resultado;
            calculo = f"L{X1} = L{Y1} - {variable1} = {VX1} - {M1} = {VX1-M1}"

    # Iterar sobre cada valor de la lista Valores
    for i, relaciones in agrupados:
        if len(relaciones) == 1:
            (X1, Y1, variable1, _) = relaciones[0]
            M1 = valores_variables_dic.get(variable1, 0)
            VX1 = valores_finales_previos[Y1-1]
            resultado = VX1 - M1
            valores_finales_previos[X1-1] = resultado;
            calculo = f"L{X1} = L{Y1} - {variable1} = {VX1} - {M1} = {VX1-M1}"
        else:
            (X1, Y1, variable1, _) = relaciones[0]
            (X2, Y2, variable2, _) = relaciones[1]
            M1 = valores_variables_dic.get(variable1, 0)
            M2 = valores_variables_dic.get(variable2, 0)
            VA = valores_finales_previos[Y1-1]
            VB = valores_finales_previos[Y2-1]
            resultado = min(VA - M1, VB - M2)
            valores_finales_previos[X1-1] = resultado;
            calculo = f"L{X1} = MIN(L{Y1} - {variable1}, L{Y2}-{variable2}) = MIN({VA} - {M1},{VB}-{M2}) = {min(VA-M1,VB-M2)}  "

        calculos_texto.append(calculo)
        
    
   

    calculos_txt_final = []
    # Mostrar los resultados
    for texto, resultado in zip(calculos_texto, valores_finales_previos):
        calculos_txt_final.append(f"{texto} ")
    
    
    Ei = {i+1: valor for i, valor in enumerate(valores_finales_previos)}
    return Ei, calculos_txt_final


def calcular_early_times(relaciones, duraciones, valores_variables):
    """
    Calcula los tiempos más tempranos (Ei) para cada nodo en un grafo dado.

    :param relaciones: Lista de relaciones [(origen, destino, nombre_variable, _)].
    :param duraciones: Diccionario con las duraciones de cada variable.
    :return: Diccionario con los tiempos más tempranos Ei y lista de cálculos detallados.
    """
    # Inicializar tiempos más tempranos (Ei)
    nodos = set([rel[0] for rel in relaciones] + [rel[1] for rel in relaciones])  # Obtener nodos únicos
    Ei = {nodo: 0 for nodo in nodos}

    valores_variables_dic = {chr(65 + i): round(float(valores_variables[i]),2) for i in range(len(valores_variables))}
    agrupados = {}

    for rel in relaciones:
        i = rel[1]  # El segundo valor de cada relación
        # Si el grupo para este i aún no existe, crear una lista vacía
        if i not in agrupados:
            agrupados[i] = []
        
        # Agregar la relación completa a la lista correspondiente al índice i
        agrupados[i].append(rel)
    
    agrupados = sorted(list(agrupados.items()), key=lambda x:x[0])
 
    # Listas para almacenar el cálculo y el resultado
    calculos_texto = []
    valores_finales_previos = [0]*len(nodos)
    valores_finales = []
    valores_finales_previos
    valores_finales.insert(0,0)


    calculos_texto.append(f"E1 = 0")

    
    
    for i, relaciones in agrupados:
        if len(relaciones) == 1:
            # Nodo simple
            (X1, Y1, variable1, _) = relaciones[0]
            M1 = valores_variables_dic.get(variable1, 0)
            VX1 = valores_finales_previos[X1-1]
            resultado = VX1 + M1
            valores_finales_previos[Y1-1] = resultado;
            calculo = f"E{Y1} = E{X1} + {variable1} = {VX1} + {M1} = {VX1+M1}"
 
     # Iterar sobre cada valor de la lista Valores
    for i, relaciones in agrupados:
        if len(relaciones) == 1:
            # Nodo simple
            (X1, Y1, variable1, _) = relaciones[0]
            M1 = valores_variables_dic.get(variable1, 0)
            VX1 = valores_finales_previos[X1-1]
            resultado = VX1 + M1
            valores_finales_previos[Y1-1] = resultado;
            calculo = f"E{Y1} = E{X1} + {variable1} = {VX1} + {M1} = {VX1+M1}"
        else:
            # Nodo compuesto
            (X1, Y1, variable1, _) = relaciones[0]
            (X2, Y2, variable2, _) = relaciones[1]
            M1 = valores_variables_dic.get(variable1, 0)
            M2 = valores_variables_dic.get(variable2, 0)
            VA = valores_finales_previos[X1-1]
            VB = valores_finales_previos[X2-1]
            resultado = max(VA + M1, VB + M2)
            valores_finales_previos[Y1-1] = resultado;
            calculo = f"E{Y1} = MAX(E{X1} + {variable1}, E{X2}+{variable2}) = MAX({VA} + {M1},{VB}+{M2}) = {max(VA+M1,VB+M2)}"
        
        calculos_texto.append(calculo)
        valores_finales.append(resultado)
    
    
     

    calculos_txt_final = []
    # Mostrar los resultados
    for texto, resultado in zip(calculos_texto, valores_finales):
        calculos_txt_final.append(f"{texto} ")
    
    
    Ei = {i+1: valor for i, valor in enumerate(valores_finales)}
    
    return Ei, calculos_txt_final


def generar_tabla_tareas_con_detalles(nodos, relaciones, duraciones, Ei, Li):
    """
    Genera la tabla de tareas con la información de la ruta, duración, Ei, Lj, Hij y estado crítico,
    incluyendo los detalles de los cálculos realizados.

    :param nodos: Lista de nodos.
    :param relaciones: Diccionario de relaciones entre nodos.
    :param duraciones: Diccionario con las duraciones de las actividades.
    :param Ei: Diccionario con los tiempos de inicio temprano (Ei) de cada nodo.
    :param Li: Diccionario con los tiempos de inicio tardío (Li) de cada nodo.
    :return: DataFrame con la tabla de tareas y lista de detalles de los cálculos.
    """
    relaciones = [rel for rel in relaciones if rel[2] != 'F1']

    
    valores_variables_dic = {chr(65 + i): round(float(duraciones[i]),2) for i in range(len(duraciones))}
    
    # Lista de resultados para la tabla
    tareas_info = []
    # Lista para los detalles de los cálculos
    detalles_tareas = []

    # Recorremos todas las actividades para llenar la tabla

    for X,Y,M,_ in relaciones:
                # Obtener la duración de la tarea
                Di = valores_variables_dic.get(M, 0)
                # Obtener Ei para el nodo origen y Lj para el nodo destino
                Ei_val = Ei[X]
                Lj_val = Li[Y]
                # Calcular Hij (holgura)
                Hij = Lj_val - Di - Ei_val
                # Verificar si la tarea es crítica
                critico = "Sí" if Hij == 0 else "No"
                
                # Añadir la fila correspondiente a la tabla
                tareas_info.append(
                    [
                        M,
                        f"{X} -> {Y}",
                        Di,
                        Ei_val,
                        Lj_val,
                        Hij,
                        critico,
                    ]
                )
                # Crear el detalle del cálculo
                detalle = (
                    f"Tarea {M}: Hij = Lj - Di - Ei = {Lj_val} - {Di} - {Ei_val} = {Hij}, "
                    f"Crítico: {critico}"
                )
                detalles_tareas.append(detalle)

    # Crear un DataFrame con la información recopilada
    tabla_tareas = pd.DataFrame(
        tareas_info, columns=["Tarea", "Ruta(i->j)", "Di", "Ei", "Lj", "Hij", "Critico"]
    )
     
    return tabla_tareas, detalles_tareas



######################## GANTT #############################


def create_gantt_chart(tabla_tareas, dir_name,  output_file="gantt_chart.png"):
    """
    Procesa una tabla de tareas y genera un gráfico de Gantt y lo guarda como un archivo PNG.

    :param tabla_tareas: DataFrame con columnas ['Tarea', 'Ruta(i->j)', 'Di']
                         - 'Tarea': nombre de la tarea
                         - 'Ruta(i->j)': cadena con formato 'start -> end'
                         - 'Di': fracción completada de la tarea (como número)
    :param output_file: Nombre del archivo de salida (con extensión .png)
    """
    # Crear el DataFrame con las columnas necesarias
    df = pd.DataFrame(
        {
            "task": tabla_tareas["Tarea"],  # Columna con nombres de tareas
            "start": pd.to_numeric(
                [ruta.split(" -> ")[0] for ruta in tabla_tareas["Ruta(i->j)"]]
            ),  # Convertir inicio a numérico
            "end": pd.to_numeric(
                [ruta.split(" -> ")[1] for ruta in tabla_tareas["Ruta(i->j)"]]
            ),  # Convertir fin a numérico
            "completion_frac": pd.to_numeric(
                tabla_tareas["Di"]
            ),  # Convertir fracción completada a numérico
        }
    )

    # Calcular días hasta el inicio y fin, duración de tareas y días completados
    df["days_to_start"] = df["start"] - df["start"].min()
    df["days_to_end"] = df["end"] - df["start"].min()
    df["task_duration"] = (
        df["days_to_end"] - df["days_to_start"] + 1
    )  # Duración en días incluyendo el final
    df["completion_days"] = df["completion_frac"] * df["task_duration"]

    # Generar gráfico de Gantt
    plt.barh(y=df["task"], width=df["task_duration"], left=df["days_to_start"])
    plt.xlabel("Días")
    plt.ylabel("Tareas")
    plt.title("Gráfico de Gantt")

    # Guardar el gráfico como archivo PNG
    plt.savefig(f"./{dir_name}/output_gantt.png", format="png", bbox_inches="tight")
    plt.close()  # Cerrar la figura para liberar memoria



def generate_global(archivo_excel, archivo_latex): 
    var_name = ""
    if "input/" in archivo_excel:
        # Extraer la parte de la cadena después de "input/"
        file_name_with_ext = archivo_excel[archivo_excel.find("input/") + len("input/"):].strip()
    
    # Obtener solo el nombre del archivo sin la extensión
    var_name = os.path.splitext(file_name_with_ext)[0]
    dir_name = f"output/{var_name}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    datos_procesados = leer_y_procesar_excel(archivo_excel)
    # Supongamos que 'datos' es el DataFrame que devuelve la función
    columna_de = datos_procesados["De"]

    # Leer las relaciones de precedencia
    relaciones_precedencia, variables = leer_matriz_dependencias(archivo_excel)


    # Crear un conjunto con todos los elementos únicos de las tuplas en lista_B
    elementos_en_B = set(item for tupla in relaciones_precedencia for item in tupla)

    # Filtrar la lista A para mantener solo los elementos que están en el conjunto
    variables_filtradas = [elemento for elemento in variables if elemento in elementos_en_B]
    
    # Generar el grafo PERT
    relations = create_pert_list(relaciones_precedencia, variables_filtradas, archivo_salida=f"{dir_name}/pert_grafo")
    
    # Paso 1: Encontrar vértices iniciales
    destinos = {b for _, b in relaciones_precedencia}
    origenes = {a for a, _ in relaciones_precedencia}
    iniciales = origenes - destinos  # Vértices que no son precedidos por nadie

    # Paso 2: Crear un mapeo de identificadores de nodos
    todos_los_nodos = origenes | destinos
    nodo_ids = {nodo: i + 2 for i, nodo in enumerate(todos_los_nodos)}
    nodo_ids["1"] = 1  # Nodo inicial



        

    # Nodos en orden progresivo de cálculo
    # Extraer los números de los nodos
    nodos = set()  # Claves del diccionario
    for origen, destino, _, _ in relations:
        nodos.add(origen)
        nodos.add(destino)

    nodos = sorted(list(nodos))

    
    # Duraciones de las actividades (t0, tm, tp calculado previamente como De)
    # Crear el diccionario 'duraciones' con las claves de la columna 'Variable' y los valores de la columna 'De'
    duraciones = {}

    # Recorrer las filas del DataFrame y agregar los valores correspondientes al diccionario
    # Crear el diccionario con redondeo
    duraciones = {
        row["Variable"]: round(row["De"], 2) for _, row in datos_procesados.iterrows()
    }

    # Agregar 'F1': 0 al diccionario
    duraciones["F1"] = 0

    # Calcular Early Times (Ei)
    tabla_early, detalles_early = calcular_early_times(relations, duraciones, columna_de)


    # Calcular Late Times (Li)
    tabla_late, detalles_late = calcular_late_times(relations, duraciones, columna_de, list(tabla_early.values())[-1])





    # Combinar ambos detalles
    detalles_combinados = {
        "Early Times": detalles_early,
        "Late Times": detalles_late,
    }


    df_tabla_early = pd.DataFrame(list(tabla_early.items()), columns = ['Nodo', 'Ei'])
    df_tabla_late = pd.DataFrame(list(tabla_late.items()), columns=['Nodo', 'Li'])


    # Unir las dos tablas en una sola
    tabla_completa = pd.merge(df_tabla_early, df_tabla_late, on="Nodo")



    # Generar la tabla de tareas
    tabla_tareas, detalles_tareas_criticos = generar_tabla_tareas_con_detalles(nodos, relations, columna_de, tabla_early, tabla_late)

    create_gantt_chart(tabla_tareas, dir_name)


    # Filtrar las tareas críticas
    tareas_criticas = tabla_tareas[tabla_tareas["Critico"] == "Sí"]

    # Mostrar el camino crítico
    camino_critico = tareas_criticas[["Tarea", "Ruta(i->j)", "Di"]]
    camino_critico_lista = '-'.join(camino_critico['Tarea'])
    # Calcular la duración total
    duracion_total_camino_critico = sum(camino_critico['Di'])

    # Crear la expresión escrita
    duraciones_camino_critico = ' + '.join(map(lambda x: f"{x:.2f}", camino_critico['Di']))
    calculo_duracion_camino_critico_latex = f"Duración crítica = {camino_critico_lista} = {duraciones_camino_critico} = {duracion_total_camino_critico:.2f}"

    # Calcular la suma de las duraciones de las tareas críticas
    suma_criticas = tareas_criticas["Di"].sum()
    
    # Exportar 'tabla_completa' a LaTeX como una cadena
    tabla_completa_latex = tabla_completa.to_latex(index=False, float_format="%.3f")

    # Exportar 'tabla_tareas' a LaTeX como una cadena
    tabla_tareas_latex = tabla_tareas.to_latex(index=False, float_format="%.3f")

    tabla_datos_procesado = datos_procesados.to_latex(index=False, float_format="%.3f")



    detalles_combinados_latex = "\n\n% Detalles de los cálculos\n"
    for tipo, detalles in detalles_combinados.items():
        detalles_combinados_latex += f"\n% {tipo}\n"  # Agregar un encabezado para cada tipo
        for detalle in detalles:
            detalles_combinados_latex += f"{detalle}\n"

    detalles_combinados_criticos_latex = "\n\n% Detalles de los cálculo críticos\n"
    for detalle in detalles_tareas_criticos:
        detalles_combinados_criticos_latex += f"{detalle}\n"
        

    # Especificar el archivo LaTeX de salida
    archivo_latex_completo = f"{dir_name}/output_pert.txt"

    # Escribir ambas tablas en el mismo archivo
    with open(archivo_latex_completo, "w") as f:
        f.write("% Datos procesados\n")
        f.write(tabla_datos_procesado)
        f.write("% Tabla Completa\n")
        f.write(tabla_completa_latex)  # Escribir la tabla completa
        f.write(detalles_combinados_latex)  # Agregar los detalles combinados
        f.write("\n\n% Tabla Tareas\n")  # Separador entre las tablas
        f.write(tabla_tareas_latex)  # Escribir la tabla de tareas
        f.write(detalles_combinados_criticos_latex)
        f.write("\nCamino crítico: ")
        f.write(camino_critico_lista)
        f.write("\n")
        f.write(calculo_duracion_camino_critico_latex)
        


####################### LLAMADA A FUNCIONES #####################
excel = ["input/ej3.xlsx", "input/ej4.xlsx", "input/ej5.xlsx"]
latex = "output/tabla.tex"  

 
for item in excel:
    generate_global(item, latex)