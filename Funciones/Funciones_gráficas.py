import numpy as np
import matplotlib.pyplot as plt
import Funciones.Funciones_agrupadas as fn
import os

def calcular_ktn_y_graficar(H_nominal, d_nominal):

    '''
    Permite graficar el valor de Ktn y graficar la relación entre Ktn y r/d
    '''

    d_nominal_values = [d_nominal, 88.46, 76.67, 57.5]
    colores = ['blue', 'orange', 'green', 'red']

    plt.figure(figsize=(10, 6))

    for d_nom, color in zip(d_nominal_values, colores):
        r_coeficientes = np.linspace(1e-10, d_nom, 100)
        r_d_coeficientes = r_coeficientes / d_nom
        t_H_coeficientes = -(r_coeficientes / r_d_coeficientes) / H_nominal

        Ktn_results = [fn.calculo_Ktn(H_nominal - d_nom, r, t_H_coeficientes[i]) for i, r in enumerate(r_coeficientes)]

        plt.plot(r_d_coeficientes, Ktn_results, label=f'H/d = {(H_nominal / d_nom) :.2f}', color=color)

    y_coeficientes = fn.calculo_Ktn(H_nominal - d_nominal, 0.15 * d_nominal, -((0.15 * d_nominal) / 0.15) / H_nominal)
    plt.scatter(0.15, y_coeficientes, marker='o', color='red', label=f'Ktn={y_coeficientes:.2f} (r/d = 0.15), H/d= {H_nominal/d_nominal:.2f}')

    plt.xlim(0, 0.3)
    plt.ylim(1, 3)
    plt.xlabel('r/d')
    plt.ylabel(f"K$_{{tn}}$")
    plt.title(f"Curva K$_{{tn}}$ obtenida mediante coeficientes y ecuaciones")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # y_coeficientes = fn.calculo_Ktn(H_nominal - d_nominal, 0.15 * d_nominal, -((0.15 * d_nominal) / 0.15) / H_nominal)

    # plt.scatter(0.15, y_coeficientes, marker='o', color='red', label=f'Ktn={y_coeficientes:.2f} (r/d = 0.15)')
    # plt.xlim(0, 0.3)
    # plt.ylim(1, 3)
    # plt.xlabel('r/d')
    # plt.ylabel('Ktn')
    # plt.title('Relación entre Ktn y r/d')
    # plt.legend(loc='best')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


def plot_ktn_curves(datos_curvas, r_d_coeficientes, Ktn_results, H_nominal, d_nominal):
    '''
    Permite graficar las curvas de Ktn mediante los datos de las curvas de Ktn
    '''
    # Inicializar listas para datos y etiquetas
    data = []
    labels = []
    colores = ['blue', 'orange', 'green', 'red']
    r_d_fijo = 0.15
    y_coeficientes = fn.calculo_Ktn(H_nominal - d_nominal, r_d_fijo * d_nominal, -((r_d_fijo * d_nominal) / r_d_fijo) / H_nominal)

    # Recorrer los datos de las curvas y almacenarlos en las listas
    for i in range(len(datos_curvas)):
        x = datos_curvas[i][0]
        y_ajustados = datos_curvas[i][2]
        y_interpolado_akima = datos_curvas[i][3]
        y_interpolado_makima = datos_curvas[i][4]
        nombre_archivo = datos_curvas[i][5]

        # data.append((x, y_ajustados))
        # labels.append(f'{nombre_archivo} - Ajustados')

        # data.append((x, y_interpolado_akima))
        # labels.append(f'{nombre_archivo} - Interpolado Akima')

        # data.append((x, y_interpolado_makima))
        # labels.append(f'{nombre_archivo} - Interpolado Makima')


        data.append((x, y_ajustados))
        labels.append('Ajustados mediante poly1d')

        data.append((x, y_interpolado_akima))
        labels.append('Interpolado Akima')

        data.append((x, y_interpolado_makima))
        labels.append('Interpolado Makima')

    # Graficar las curvas de Ktn
    plt.figure(figsize=(10, 6))
    for i, (x, y) in enumerate(data):
        color_index = (i // 3) % len(colores)
        plt.plot(x, y, label=labels[i], color=colores[color_index])

    plt.xlabel('H/d')
    plt.ylabel(f"K$_{{tn}}$")
    plt.title(f"Curvas de K$_{{tn}}$ extraidas de archivos de texto")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
    '''
    ........................................Combinación de coeficientes + archivos de puntos........................................
    '''

    # Crear una nueva figura para la combinación de Ktn_results y datos_curvas
    plt.figure(figsize=(10, 6))

    d_nominal_values = [d_nominal, 88.46, 76.67, 57.5]
    colores = ['blue', 'orange', 'green', 'red']


    for d_nom, color in zip(d_nominal_values, colores):
        r_coeficientes = np.linspace(1e-10, d_nom, 100)
        r_d_coeficientes = r_coeficientes / d_nom
        t_H_coeficientes = -(r_coeficientes / r_d_coeficientes) / H_nominal

        Ktn_results = [fn.calculo_Ktn(H_nominal - d_nom, r, t_H_coeficientes[i]) for i, r in enumerate(r_coeficientes)]

        plt.plot(r_d_coeficientes, Ktn_results, label=f'Mediante coeficientes. H/d = {(H_nominal / d_nom) :.2f}', color=color)


    # # Graficar Ktn_results
    # plt.plot(r_d_coeficientes, Ktn_results, label=f'Ktn vs r/d, H/d = {(H_nominal / d_nominal):.2f}', color='blue')

    # Graficar datos_curvas
    for i, (x, y) in enumerate(data):
        plt.plot(x, y, label=f"Mediante archivos. {labels[i]}")

    # Añadir puntos específicos
    plt.scatter(r_d_fijo, y_coeficientes, marker='o', color='red', label=f'Curva comparación frente a las ajustadas. Ktn={y_coeficientes:.2f} (r/d = {r_d_fijo}), H/d= {H_nominal/d_nominal:.2f}')

    # Configurar la gráfica
    plt.xlim(0, 0.3)
    plt.ylim(1, 3)
    plt.xlabel('r/d')
    plt.ylabel(f"K$_{{tn}}$")
    plt.title(f"Combinación de curvas K$_{{tn}}$ por coeficientes frente a los archivos")
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()





def graficar_ecuacion_y_archivos_Nf(resultados_Ktn_nominal, d_nominal, h, M, R):

    '''
    Permite graficar la ecuación de fatiga y los archivos de puntos de las curvas S/N
    '''
    
    # Parámetros del problema
    s_max_ksi = np.linspace(0.1, 60, 100)  # Evitar s_eq cerca de 0 # Esfuerzo máximo en ksi
    r_values = [-1, 0.0, 0.1, 0.5]  # Diferentes valores de r

        # Convertir s_max_ksi a Pa usando la función kasi_to_pa
    s_max_pa = fn.ksi_to_pa(s_max_ksi)
    # Asegurarse de que los valores en s_max_pa sean al menos 5e6
    s_max_pa = np.where(s_max_pa < 5e6, 5e6, s_max_pa)
    s_max_ksi = fn.pa_to_ksi(s_max_pa)

    # Generación de gráficos
    plt.figure(figsize=(10, 6))
    plt.grid(True, which="both", ls="-")
    plt.xscale('log')
    # Graficar para cada valor de r
    for r in r_values:
        nf_values = fn.calcular_nf(s_max_ksi, r)

        # Gráfica
        plt.plot(nf_values[0], s_max_ksi, linestyle='-', label=f"r={r}")

    plt.axhline(y=fn.pa_to_ksi(5e6), color='blue', linestyle='-', label = f'Linea de 5 MPa, en KSI: {fn.pa_to_ksi(5e6):.2f}')
    plt.ylabel('Esfuerzo en KSI')
    plt.xlabel('Ciclos de Vida por Fatiga (Nf)')
    plt.title('Ecuaciones S-N para diferentes valores de radio')
    plt.legend()
 
    plt.show()

    '''
    _______________________________________________Graficado de los archivos de puntos___________________________________________________________
    '''

    # Archivos y parámetros para las curvas S/N
    carpeta_archivos = 'Archivos_de_texto_Curvas'
    archivos = ['Curva0_5_SN.txt', 'Curva0_1_SN.txt', 'Curva0_SN.txt', 'Curva_1_SN.txt']
    grados = [3, 3, 3, 3]
    colores = ['blue', 'orange', 'green', 'red']
    archivos = [os.path.join(carpeta_archivos, archivo) for archivo in archivos]

    # Procesar archivos para obtener polinomios e interpolaciones
    _, polinomios_akima_SN, polinomios_makima_SN, datos_SN = fn.process_files(archivos, grados)

    # Calcular Ktn final usando el método Makima
    Ktn_makima_final_nominal = resultados_Ktn_nominal["valor_Ktn_final_makima"]

    _, s_eq_ksi, Ciclos_de_vida = fn.fatigue_life(R, h, M, d_nominal, Ktn_makima_final_nominal)

    # Graficar curvas S/N ajustadas con el esfuerzo máximo
    plt.figure(figsize=(10, 6))
    for (x, y, y_ajustados, y_interpolado_akima, y_interpolado_makima, nombre_archivo), color in zip(datos_SN, colores):
        plt.plot(x, y_interpolado_makima, ':', label=f'Interpolación Makima {nombre_archivo}', color=color)

    plt.axhline(y=fn.pa_to_ksi(5e6), color='blue', linestyle='-', label = f'Linea de 5 MPa, en KSI: {fn.pa_to_ksi(5e6):.2f}')
    plt.xscale('log')
    plt.xlim(1e3, 1e8)
    plt.ylim(0, 100)
    plt.xlabel('Ciclos de vida a fatiga (Nf)')
    plt.ylabel('Esfuerzo (KSI)')
    plt.title('Curvas S-N ajustadas para diferentes relaciones de esfuerzo')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()

    '''
    _______________________________________________Combinación de ecuación y archivos de puntos___________________________________________________________
    '''

    # Crear una nueva figura para la combinación de ecuación y archivos de puntos
    plt.figure(figsize=(10, 6))

    # Graficar ecuación de fatiga
    for r in r_values:
        nf_values = fn.calcular_nf(s_max_ksi, r)
        plt.plot(nf_values[0], s_max_ksi, linestyle='-', label=f"Ecuación r={r}")

    # Graficar curvas S/N ajustadas con el esfuerzo máximo
    for (x, y, y_ajustados, y_interpolado_akima, y_interpolado_makima, nombre_archivo), color in zip(datos_SN, colores):
        plt.plot(x, y_interpolado_makima, ':', label=f'Interpolación Makima {nombre_archivo}', color=color)

    plt.axhline(y=fn.pa_to_ksi(5e6), color='blue', linestyle='-', label = f'Linea de 5 MPa, en KSI: {fn.pa_to_ksi(5e6):.2f}')
    plt.xscale('log')
    plt.xlim(1e3, 1e8)
    plt.ylim(0, 100)
    plt.xlabel('Ciclos de vida a fatiga (Nf)')
    plt.ylabel('Esfuerzo (KSI)')
    plt.title('Combinación de curvas S-N usando ecuación y mediante archivos')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
