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


def generar_grafo_pert(relaciones, variables, archivo_salida="graph"):
    """
    Genera un grafo PERT a partir de las relaciones de precedencia usando Graphviz.

    :param relaciones: Lista de relaciones de precedencia como tuplas (origen, destino).
    :param variables: Lista de nombres de las variables.
    :param archivo_salida: Nombre del archivo de salida sin extensión.
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
    nodos_con_salientes = [
        var for var, tiene_salientes in nodos_salientes.items() if tiene_salientes
    ]

    # Eliminar nodos sin precedencia de la lista
    nodos_con_salientes_num = [var_to_num[var] for var in nodos_con_salientes]

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

    # Renderizar el grafo
    dot.render(archivo_salida, format="png", cleanup=True)
    ##print(f"Grafo generado y guardado como {archivo_salida}.png")


######################## CALCULO TABLAS #########################
def calcular_early_times(nodos, relaciones, duraciones):
    """
    Calcula Early Times (Ei) para una red de nodos.

    :param nodos: Lista de nodos (ordenados según dependencias para asegurar cálculos progresivos).
    :param relaciones: Diccionario donde las claves son nodos y los valores son listas de tuplas (actividad, destino).
    :param duraciones: Diccionario con las duraciones de las actividades.
    :return: Diccionario con los tiempos Ei y un DataFrame con los resultados.
    """
    # Inicializar Early Times (Ei) en 0 para todos los nodos
    Ei = {nodo: 0 for nodo in nodos}

    # Calcular Early Times (Ei) nodo por nodo
    for nodo in nodos:
        if nodo in relaciones:  # Si el nodo tiene conexiones salientes
            for actividad, destino in relaciones[nodo]:
                # Actualizar Ei del destino como el máximo entre los valores actuales y el nuevo cálculo
                Ei[destino] = max(Ei[destino], Ei[nodo] + duraciones[actividad])

    # Crear un DataFrame con los resultados
    tabla_early = pd.DataFrame(
        {
            "Ti": nodos,
            "Ei": [Ei[nodo] for nodo in nodos],
        }
    )

    return Ei, tabla_early


def calcular_late_times(nodos, relaciones, duraciones, Ei):
    """
    Calcula Late Times (Li) para una red de nodos, usando la lógica inversa de los Early Times.

    :param nodos: Lista de nodos (ordenados según dependencias para asegurar cálculos progresivos).
    :param relaciones: Diccionario donde las claves son nodos y los valores son listas de tuplas (actividad, destino).
    :param duraciones: Diccionario con las duraciones de las actividades.
    :param Ei: Diccionario con los Early Times calculados previamente.
    :return: Diccionario con los tiempos Li y un DataFrame con los resultados.
    """
    # Inicializar Late Times (Li) con valores grandes (un valor suficientemente grande para empezar)
    Li = {nodo: float("inf") for nodo in nodos}

    # El último nodo tiene Li igual a Ei
    Li[nodos[-1]] = Ei[nodos[-1]]

    # Calcular Late Times (Li) de forma inversa
    for nodo in reversed(nodos[:-1]):
        if nodo in relaciones:  # Si el nodo tiene conexiones salientes
            for actividad, destino in relaciones[nodo]:
                # Actualizar Li del nodo como el mínimo entre los valores actuales y el cálculo usando la actividad
                Li[nodo] = min(Li[nodo], Li[destino] - duraciones[actividad])

    # Crear un DataFrame con los resultados
    tabla_late = pd.DataFrame(
        {
            "Ti": nodos,
            "Li": [Li[nodo] for nodo in nodos],
        }
    )

    return Li, tabla_late


def generar_tabla_tareas(nodos, relaciones, duraciones, Ei, Li):
    """
    Genera la tabla de tareas con la información de la ruta, duración, Ei, Lj, Hij y estado crítico.

    :param nodos: Lista de nodos.
    :param relaciones: Diccionario de relaciones entre nodos.
    :param duraciones: Diccionario con las duraciones de las actividades.
    :param Ei: Diccionario con los tiempos de inicio temprano (Ei) de cada nodo.
    :param Li: Diccionario con los tiempos de inicio tardío (Li) de cada nodo.
    :return: DataFrame con la tabla de tareas.
    """
    # Lista de resultados para la tabla
    tareas_info = []

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

    # Crear un DataFrame con la información recopilada
    tabla_tareas = pd.DataFrame(
        tareas_info, columns=["Tarea", "Ruta(i->j)", "Di", "Ei", "Lj", "Hij", "Critico"]
    )

    return tabla_tareas


######################## GANTT #############################
def create_gantt_chart(tabla_tareas):
    """
    Procesa una tabla de tareas y genera un gráfico de Gantt.

    :param tabla_tareas: DataFrame con columnas ['Tarea', 'Ruta(i->j)', 'Di']
                         - 'Tarea': nombre de la tarea
                         - 'Ruta(i->j)': cadena con formato 'start -> end'
                         - 'Di': fracción completada de la tarea (como número)
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
    plt.show()


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
resultado = [elemento for elemento in variables if elemento in elementos_en_B]

# Generar el grafo PERT
generar_grafo_pert(
    relaciones_precedencia, resultado, archivo_salida="output/pert_grafo"
)


# Paso 1: Encontrar vértices iniciales
destinos = {b for _, b in relaciones_precedencia}
origenes = {a for a, _ in relaciones_precedencia}
iniciales = origenes - destinos  # Vértices que no son precedidos por nadie

# Paso 2: Crear un mapeo de identificadores de nodos
todos_los_nodos = origenes | destinos
nodo_ids = {nodo: i + 2 for i, nodo in enumerate(todos_los_nodos)}
nodo_ids["1"] = 1  # Nodo inicial

# Paso 3: Crear las relaciones
relaciones = {1: [(nodo, nodo_ids[nodo]) for nodo in iniciales]}  # Nodo inicial
for origen, destino in relaciones_precedencia:
    nodo_origen = nodo_ids[origen]
    nodo_destino = nodo_ids[destino]
    if nodo_origen not in relaciones:
        relaciones[nodo_origen] = []
    relaciones[nodo_origen].append((origen, nodo_destino))


# Nodos en orden progresivo de cálculo
# Extraer los números de los nodos
nodos = set(relaciones.keys())  # Claves del diccionario
for conexiones in relaciones.values():
    for _, destino in conexiones:
        nodos.add(destino)  # Agregar los nodos destino

# Convertir a lista y ordenar
nodos = sorted(nodos)


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
Ei, tabla_early = calcular_early_times(nodos, relaciones, duraciones)

# Calcular Late Times (Li)
Li, tabla_late = calcular_late_times(nodos, relaciones, duraciones, Ei)

# Unir las dos tablas en una sola
tabla_completa = pd.merge(tabla_early, tabla_late, on="Ti")

# Generar la tabla de tareas
tabla_tareas = generar_tabla_tareas(nodos, relaciones, duraciones, Ei, Li)


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
camino_critico_latex = camino_critico.to_latex(index=False, float_format="%.3f")

# Especificar el archivo LaTeX de salida
archivo_latex_completo = "output/output_pert.txt"

# Escribir ambas tablas en el mismo archivo
with open(archivo_latex_completo, "w") as f:
    f.write("% Datos procesados\n")
    f.write(tabla_datos_procesado)
    f.write("% Tabla Completa\n")
    f.write(tabla_completa_latex)  # Escribir la tabla completa
    f.write("\n\n% Tabla Tareas\n")  # Separador entre las tablas
    f.write(tabla_tareas_latex)  # Escribir la tabla de tareas
    f.write("% Camino crítico")
    f.write(camino_critico_latex)
    f.write(
        f"\n\n% Suma de las duraciones de las tareas críticas: {suma_criticas:.3f}\n"
    )
