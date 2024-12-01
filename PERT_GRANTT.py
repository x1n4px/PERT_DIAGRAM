import pandas as pd
from graphviz import Digraph
import matplotlib.pyplot as plt


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


from graphviz import Digraph

def generar_grafo_pert(relaciones, variables, archivo_salida="graph"):
    """
    Genera un grafo PERT a partir de las relaciones de precedencia usando Graphviz.

    :param relaciones: Lista de relaciones de precedencia como tuplas (origen, destino).
    :param variables: Lista de nombres de las variables.
    :param archivo_salida: Nombre del archivo de salida sin extensión.
    :return: Lista de las relaciones creadas en el formato [(origen, destino, label, estilo)].
    """
    
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
            label = origen
        # Agregar la arista con el label apropiado
        dot.edge(str(origen_num), str(destino_num), label=label, style=style)
        relaciones_creadas.append((origen_num, destino_num, label, style))  # Registrar la relación creada

        # Registrar la relación en el diccionario de precedencias
        precedencias[destino].append(origen)
  
    # Conectar los nodos sin precedencia al nodo "1"
    for var in nodos_sin_precedencia:
        if var != nodo_inicio:  # Evitar que el nodo inicial se conecte a sí mismo
            dot.edge(str(var_to_num[nodo_inicio]), str(var_to_num[var]), label=var)
            relaciones_creadas.append((var_to_num[nodo_inicio], var_to_num[var], var, "solid"))  # Registrar

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


######################## CALCULO TABLAS #########################


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
    valores_finales_previos = []
    valores_finales = []
    valores_finales_previos.insert(0,0)
    valores_finales.insert(0,0)

    # Iterar sobre cada valor de la lista Valores
    for i, relaciones in agrupados:
        if len(relaciones) == 1:
            # Solo un item, hacer X1 + M1
            (X1, Y1, variable1, _) = relaciones[0]
            M1 = valores_variables_dic.get(variable1, 0)
            resultado = X1 + M1
        else:
            resultado = -1
        
        valores_finales_previos.append(resultado)
    
     
     # Iterar sobre cada valor de la lista de relaciones agrupadas
    for i, grupo_relaciones in agrupados:
        if len(grupo_relaciones) == 1:
            # Solo un item, hacer X1 + M1
            (X1, Y1, variable1, _) = grupo_relaciones[0]
            M1 = valores_variables_dic.get(variable1, 0)
            VA = valores_finales_previos[X1-1]
            resultado = VA + M1
            calculo = f"E{Y1} = E{X1} + {variable1} = ({VA} + {M1}) = {resultado}"
            if valores_finales_previos[Y1-1] == -1:
                valores_finales_previos[Y1-1] = resultado
        else:
            # Dos items, hacer MAX(X1 + M1, X2 + M2)
            (X1, Y1, variable1, _) = grupo_relaciones[0]
            (X2, Y2, variable2, _) = grupo_relaciones[1]
            M1 = valores_variables_dic.get(variable1, 0)
            M2 = valores_variables_dic.get(variable2, 0)
            VA = valores_finales_previos[X1-1]
            VB = valores_finales_previos[X2-1]
            resultado = max(VA + M1, VB + M2)
            calculo = f"E{Y1} = MAX(E{X1} + {variable1}, E{X2} + {variable2}) = ({VA} + {M1}, {VB} + {M2}) = {resultado}"

            valores_finales_previos[Y1-1] = resultado 
            
        
        # Almacenar el cálculo y el resultado
        calculos_texto.append(calculo)
        valores_finales.append(resultado)
        
    calculos_txt_final = []
    # Mostrar los resultados
    for texto, resultado in zip(calculos_texto, valores_finales):
        calculos_txt_final.append(f"{texto} ")
    
    
    Ei = {i+1: valor for i, valor in enumerate(valores_finales)}
    
    return Ei, calculos_txt_final

   
'''
def calcular_late_times_con_detalles(nodos, relaciones, duraciones, lastValue):
    """
    Calcula Late Times (Li) para una red de nodos, registrando cálculos detallados en un array separado.

    :param nodos: Lista de nodos (ordenados según dependencias para asegurar cálculos progresivos).
    :param relaciones: Lista de relaciones [(origen, destino, nombre_variable, tag)].
    :param duraciones: Diccionario con las duraciones de las actividades.
    :param Ei: Diccionario con los Early Times calculados previamente.
    :return: Diccionario con los tiempos Li, un DataFrame con los resultados y una lista de cálculos detallados.
    """
    # Inicializar Late Times (Li) con valores infinitos
    Li = {nodo: float("inf") for nodo in nodos}
    detalles_calculos = []

    # El último nodo tiene Li igual a Ei
    Li[nodos[-1]] = lastValue

    # Registrar cálculo del último nodo
    detalles_calculos.append(f"L{nodos[-1]} = E{nodos[-1]} = {lastValue}")

    # Crear un diccionario para las relaciones por nodo de origen
    relaciones_dict = {}
    for origen, destino, actividad, _ in relaciones:
        if origen not in relaciones_dict:
            relaciones_dict[origen] = []
        relaciones_dict[origen].append((actividad, destino))

    # Calcular Late Times (Li) de forma inversa
    for nodo in reversed(nodos[:-1]):
        if nodo in relaciones_dict:  # Si el nodo tiene conexiones salientes
            for actividad, destino in relaciones_dict[nodo]:
                nuevo_valor = Li[destino] - duraciones.get(actividad, 0)
                if nuevo_valor < Li[nodo]:
                    Li[nodo] = nuevo_valor
                    # Registrar el cálculo en detalle
                    detalles_calculos.append(
                        f"L{nodo} = min(L{nodo}, L{destino} - {actividad}) = min({Li[nodo]}, {Li[destino]} - {duraciones[actividad]}) = {nuevo_valor}"
                    )

    # Crear un DataFrame con los resultados
    tabla_late = pd.DataFrame(
        {
            "Nodo": nodos,
            "Li": [Li[nodo] for nodo in nodos],
        }
    )
    return Li, tabla_late, detalles_calculos
'''
def calcular_late_times(relaciones, duraciones, valores_variables, maxEarlyValue):
    """
    Calcula los tiempos más tardios (Li) para cada nodo en un grafo dado.

    :param relaciones: Lista de relaciones [(origen, destino, nombre_variable, _)].
    :param duraciones: Diccionario con las duraciones de cada variable.
    :return: Diccionario con los tiempos más tempranos Ei y lista de cálculos detallados.
    """
    # Inicializar tiempos más tempranos (Ei)
    nodos = set([rel[0] for rel in relaciones] + [rel[1] for rel in relaciones])  # Obtener nodos únicos
    Li = {nodo: 0 for nodo in nodos}
    
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
    valores_finales_previos = [0] * len(nodos)
    valores_finales = [0] * len(nodos)
    valores_finales_previos[-1] = maxEarlyValue
    valores_finales[-1] = maxEarlyValue
    

    print(relaciones)
    # Iterar sobre cada valor de la lista Valores
    for i, relaciones in agrupados:
        if len(relaciones) == 1:
            # Solo un item, hacer X1 + M1
            (X1, Y1, variable1, _) = relaciones[0]
            M1 = valores_variables_dic.get(variable1, 0)
            resultado = X1 + M1
        else:
            resultado = -1
        
        valores_finales_previos.append(resultado)
    
     
     # Iterar sobre cada valor de la lista de relaciones agrupadas
    for i, grupo_relaciones in agrupados:
        if len(grupo_relaciones) == 1:
            # Solo un item, hacer X1 + M1
            (X1, Y1, variable1, _) = grupo_relaciones[0]
            M1 = valores_variables_dic.get(variable1, 0)
            VA = valores_finales_previos[X1-1]
            resultado = VA + M1
            calculo = f"E{Y1} = E{X1} + {variable1} = ({VA} + {M1}) = {resultado}"
            if valores_finales_previos[Y1-1] == -1:
                valores_finales_previos[Y1-1] = resultado
        else:
            # Dos items, hacer MAX(X1 + M1, X2 + M2)
            (X1, Y1, variable1, _) = grupo_relaciones[0]
            (X2, Y2, variable2, _) = grupo_relaciones[1]
            M1 = valores_variables_dic.get(variable1, 0)
            M2 = valores_variables_dic.get(variable2, 0)
            VA = valores_finales_previos[X1-1]
            VB = valores_finales_previos[X2-1]
            resultado = max(VA + M1, VB + M2)
            calculo = f"E{Y1} = MAX(E{X1} + {variable1}, E{X2} + {variable2}) = ({VA} + {M1}, {VB} + {M2}) = {resultado}"

            valores_finales_previos[Y1-1] = resultado 
            
        
        # Almacenar el cálculo y el resultado
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
    # Lista de resultados para la tabla
    tareas_info = []
    # Lista para los detalles de los cálculos
    detalles_tareas = []

    # Recorremos todas las actividades para llenar la tabla
    for nodo in nodos:
        if nodo in relaciones:  # Si el nodo tiene conexiones salientes
            for actividad, destino in relaciones[nodo]:
                # Obtener la duración de la tarea
                Di = duraciones[actividad]
                # Obtener Ei para el nodo origen y Lj para el nodo destino
                Ei_val = Ei[nodo]
                Lj_val = Li[destino]
                # Calcular Hij (holgura)
                Hij = Lj_val - Di - Ei_val
                # Verificar si la tarea es crítica
                critico = "Sí" if Hij == 0 else "No"
                # Añadir la fila correspondiente a la tabla
                tareas_info.append(
                    [
                        actividad,
                        f"{nodo} -> {destino}",
                        Di,
                        Ei_val,
                        Lj_val,
                        Hij,
                        critico,
                    ]
                )
                # Crear el detalle del cálculo
                detalle = (
                    f"Tarea {actividad}: Hij = Lj - Di - Ei = {Lj_val} - {Di} - {Ei_val} = {Hij}, "
                    f"Crítico: {critico}"
                )
                detalles_tareas.append(detalle)

    # Crear un DataFrame con la información recopilada
    tabla_tareas = pd.DataFrame(
        tareas_info, columns=["Tarea", "Ruta(i->j)", "Di", "Ei", "Lj", "Hij", "Critico"]
    )

    return tabla_tareas, detalles_tareas



######################## GANTT #############################


def create_gantt_chart(tabla_tareas, output_file="gantt_chart.png"):
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
    plt.savefig("./output/output_gantt.png", format="png", bbox_inches="tight")
    plt.close()  # Cerrar la figura para liberar memoria



####################### LLAMADA A FUNCIONES #####################
archivo_excel = "input/input.xlsx"
archivo_latex = "output/tabla.tex"  

# Leer, procesar y exportar
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
relations = generar_grafo_pert(
    relaciones_precedencia, variables_filtradas, archivo_salida="output/pert_grafo"
)


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

 
'''
# Unir las dos tablas en una sola
tabla_completa = pd.merge(df_tabla_early, tabla_late, on="Nodo")

    

# Generar la tabla de tareas
tabla_tareas, detalles_tareas_criticos = generar_tabla_tareas_con_detalles(nodos, relations, duraciones, tabla_early, Li)


create_gantt_chart(tabla_tareas)


# Filtrar las tareas críticas
tareas_criticas = tabla_tareas[tabla_tareas["Critico"] == "Sí"]

# Mostrar el camino crítico
camino_critico = tareas_criticas[["Tarea", "Ruta(i->j)", "Di"]]

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
archivo_latex_completo = "output/output_pert.txt"

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
    f.write("% Camino crítico")

    f.write(
        f"\n\n Suma de las duraciones de las tareas críticas: {suma_criticas:.3f}\n"
    )
'''