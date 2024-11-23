import pandas as pd

def leer_y_procesar_excel(archivo_excel):
    """
    Lee un archivo Excel, filtra las variables con valores completos y calcula De y Var para cada variable.
    
    :param archivo_excel: Ruta del archivo Excel.
    :return: Un DataFrame con las variables, sus valores y los cálculos de De y Var.
    """
    # Leer el archivo Excel sin encabezados
    df = pd.read_excel(archivo_excel, header=None)
    
    # Extraer los nombres de las variables y sus valores
    variables = df.iloc[0, 1:].tolist()  # Fila 1 (nombres de variables), columna B en adelante
    t0 = df.iloc[1, 1:].tolist()         # Fila 2 (t0)
    tm = df.iloc[2, 1:].tolist()         # Fila 3 (tm)
    tp = df.iloc[3, 1:].tolist()         # Fila 4 (tp)

    # Crear un DataFrame con las variables y valores
    datos = pd.DataFrame({
        'Variable': variables,
        't0': t0,
        'tm': tm,
        'tp': tp
    })

    # Filtrar solo las variables con valores completos
    datos = datos.dropna()

    # Calcular De y Var para cada fila
    datos['De'] = (datos['t0'] + 4 * datos['tm'] + datos['tp']) / 6
    datos['Var'] = ((datos['tp'] - datos['t0']) / 6) ** 2

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
    with open(archivo_latex, 'w') as f:
        f.write(tabla_latex)

# Uso del programa
archivo_excel = "input2.xlsx"  # Cambia esto por la ruta de tu archivo Excel
archivo_latex = "tabla.tex"  # Ruta del archivo de salida en LaTeX

# Leer, procesar y exportar
datos_procesados = leer_y_procesar_excel(archivo_excel)
exportar_a_latex(datos_procesados, archivo_latex)
# Mostrar la tabla en consola
print("Tabla en formato LaTeX:")
print(datos_procesados.to_latex(index=False, float_format="%.3f"))
