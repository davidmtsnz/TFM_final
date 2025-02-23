import os
import Funciones.Funciones_agrupadas as fn
import Funciones.Funciones_gráficas as fn_g
import matplotlib.pyplot as plt
import numpy as np


# fn.verificar_requisitos()

# Datos de entrada
H_nominal = 115
d_nominal = 75
t_nominal = H_nominal - d_nominal
r_nominal = 20

'''
_______________________________________________Calculo mediante coeficientes___________________________________________________________
'''

# Estudio inicial de las curvas de Ktn mediante coeficientes
r_coeficientes = np.linspace(1e-10, d_nominal, 100)
r_d_coeficientes = r_coeficientes / d_nominal
t_H_coeficientes = -(r_coeficientes / r_d_coeficientes) / H_nominal

# Calcular Ktn para cada valor de r_coeficientes
Ktn_results = [fn.calculo_Ktn(H_nominal-d_nominal, r, t_H_coeficientes[i]) for i, r in enumerate(r_coeficientes)]

fn_g.calcular_ktn_y_graficar(H_nominal, d_nominal)


'''
_______________________________________Calculo mediante archivos de puntos___________________________________________________________
'''

# Calcular Ktn nominal
resultados_Ktn_nominal = fn.calculate_ktn(H_nominal, d_nominal, r_nominal)

# Acceder a los datos del diccionario
datos_curvas = resultados_Ktn_nominal["datos"]

# Llamar a la función con los datos de las curvas
fn_g.plot_ktn_curves(datos_curvas, r_d_coeficientes, Ktn_results, H_nominal, d_nominal)



'''
_______________________________________Calculo de Ktn mediante archivos___________________________________________________________
'''

# Valores de HD para interpolación
HD_values = [1.15, 1.3, 1.5, 2]

# Obtener Ktn para diferentes interpolaciones
Ktn_para_r_d_nominal = resultados_Ktn_nominal["Ktn_para_r_d"]
Ktn_para_r_d_nominal_akima = resultados_Ktn_nominal["Ktn_para_r_d_akima"]
Ktn_para_r_d_nominal_makima = resultados_Ktn_nominal["Ktn_para_r_d_makima"]

# Interpoladores usando la función ajuste_polinomico
_, interpolador_Ktn_r_d, _ , _ = fn.ajuste_polinomico(HD_values, Ktn_para_r_d_nominal)
_, _, interpolador_akima_Ktn_r_d , _ = fn.ajuste_polinomico(HD_values, Ktn_para_r_d_nominal_akima)
_, _, _ , interpolador_makima_Ktn_r_d = fn.ajuste_polinomico(HD_values, Ktn_para_r_d_nominal_makima)

# Calcular valores interpolados para HD_values
Ktn_calculados_poly = interpolador_Ktn_r_d(resultados_Ktn_nominal["H_d"])
Ktn_calculados_akima = interpolador_akima_Ktn_r_d(resultados_Ktn_nominal["H_d"])
Ktn_calculados_makima = interpolador_makima_Ktn_r_d(resultados_Ktn_nominal["H_d"])

# Crear un linspace para representar las funciones
HD_linspace = np.linspace(min(HD_values), max(HD_values), 500)

# Calcular los valores interpolados usando los interpoladores
Ktn_interpolados_poly = interpolador_Ktn_r_d(HD_linspace)
Ktn_interpolados_akima = interpolador_akima_Ktn_r_d(HD_linspace)
Ktn_interpolados_makima = interpolador_makima_Ktn_r_d(HD_linspace)

# Datos y etiquetas para la interpolación
data_interpolated = [
    (HD_linspace, Ktn_interpolados_poly),
    (HD_linspace, Ktn_interpolados_akima),
    (HD_linspace, Ktn_interpolados_makima)
]

labels_interpolated = [
    'Interpolación Polyfit',
    'Interpolación Akima',
    'Interpolación Makima'
]

# Graficar interpolaciones
plt.figure(figsize=(10, 6))
for i, (x, y) in enumerate(data_interpolated):
    plt.plot(x, y, label=labels_interpolated[i])

# Añadir scatter de puntos para HD_values en función del Ktn obtenido
HD_points = HD_values + [H_nominal / d_nominal]
Ktn_points_poly = list(Ktn_para_r_d_nominal) + [Ktn_calculados_poly]
Ktn_points_akima = list(Ktn_para_r_d_nominal_akima) + [Ktn_calculados_akima]
Ktn_points_makima = list(Ktn_para_r_d_nominal_makima) + [Ktn_calculados_makima]

# Puntos para HD_values
plt.scatter(HD_values, Ktn_para_r_d_nominal, marker='o', color='blue', label='Puntos Polyfit HD_values')
plt.scatter(HD_values, Ktn_para_r_d_nominal_akima, marker='x', color='orange', label='Puntos Akima HD_values')
plt.scatter(HD_values, Ktn_para_r_d_nominal_makima, marker='s', color='green', label='Puntos Makima HD_values')

# Puntos calculados
plt.scatter([H_nominal / d_nominal], [Ktn_calculados_poly], marker='o', color='cyan', label=f'Punto Calculado Polyfit Ktn={Ktn_calculados_poly:.2f}')
plt.scatter([H_nominal / d_nominal], [Ktn_calculados_akima], marker='x', color='red', label=f'Punto Calculado Akima Ktn={Ktn_calculados_akima:.2f}')
plt.scatter([H_nominal / d_nominal], [Ktn_calculados_makima], marker='s', color='magenta', label=f'Punto Calculado Makima Ktn={Ktn_calculados_makima:.2f}')

plt.xlabel('HD')
plt.ylabel('Ktn')
plt.title('Interpolación de Ktn para r_d')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()


'''
_______________________________________________Ciclo de vida para una pieza distinta de la nominal____________________________________________________
'''

# Datos para el cálculo de esfuerzo
h = 3  # Espesor de la placa en mm
M = 10e11  # Momento flector en Nmm (10^6 Nmm = 10^3 Nm = 1 kNm)
R = 0.1  # Ratio de estres


H = 115
d = 70
r = 20

print(fn.nominal_stress(h, M, d)) # MPa
print(fn.pa_to_ksi(fn.nominal_stress(h, M, d)))

# Calcular Ktn cualquiera
resultados_Ktn = fn.calculate_ktn(H, d, r)

# Acceder a los datos del diccionario
datos_curvas_cualquiera = resultados_Ktn["datos"]
Ktn_para_r_d_makima = resultados_Ktn["Ktn_para_r_d_makima"]
H_d = resultados_Ktn["H_d"]

# Valores de HD para interpolación
HD_values = [1.15, 1.3, 1.5, 2]

_, _, _ , interpolador_makima_Ktn_r_d = fn.ajuste_polinomico(HD_values, Ktn_para_r_d_makima)

# Calcular valores interpolados para HD_values
Ktn_calculado_makima = interpolador_makima_Ktn_r_d(H_d)


# Graficar datos_curvas_cualquiera
plt.figure(figsize=(10, 6))
for (x, _, _, _, y_interpolado_makima, nombre_archivo) in datos_curvas_cualquiera:
    plt.plot(x, y_interpolado_makima, label=f'Interpolación Makima {nombre_archivo}')

# Añadir punto específico
plt.scatter(resultados_Ktn["r_d"], Ktn_calculado_makima, marker='o', color='red', label=f'Punto específico Ktn={Ktn_calculado_makima:.2f} (r/d = {resultados_Ktn["r_d"]:.2f}, H/d = {resultados_Ktn["H_d"]:.2f})')

# Configurar la gráfica
plt.xlim(0, 0.3)
plt.ylim(1, 3)
plt.xlabel('r/d')
plt.ylabel('Ktn')
plt.title('Curvas de Ktn Interpoladas Makima')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()


'''
........................................Combinación de coeficientes + archivos de puntos........................................
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

# Parámetros del problema
s_max_ksi = np.linspace(0.1, 60, 100)  # Evitar s_eq cerca de 0 # Esfuerzo máximo en ksi
r_values = [-1, 0.0, 0.1, 0.5]  # Diferentes valores de r


fn_g.graficar_ecuacion_y_archivos_Nf(resultados_Ktn_nominal, d_nominal, h, M, R)


_, s_eq_ksi, Ciclos_de_vida = fn.fatigue_life(R, h, M, d_nominal, Ktn_makima_final_nominal)



# Calcular Ktn final usando el método Makima
Ktn_makima_final = resultados_Ktn_nominal["valor_Ktn_final_makima"]

_, s_eq_ksi, Ciclos_de_vida = fn.fatigue_life(R, h, M, d_nominal, Ktn_makima_final_nominal)
_, s_eq_ksi_cualquiera, Ciclos_de_vida_cualquiera = fn.fatigue_life(R, h, M, d, Ktn_makima_final)

# Generación de gráficos
plt.figure(figsize=(10, 6))

# Graficar para cada valor de r
for r in r_values:
    nf_values = fn.calcular_nf(s_max_ksi, r)

    # Gráfica
    plt.plot(nf_values[0], s_max_ksi, linestyle='-', label=f"r={r}")


# Añadir scatter y axvline
plt.scatter(Ciclos_de_vida, s_eq_ksi, marker='o', label=f'{s_eq_ksi} - Esfuerzo máximo, Ciclos de vida: {Ciclos_de_vida:.2f}, Ratio {R}')
plt.scatter(Ciclos_de_vida_cualquiera, s_eq_ksi_cualquiera, marker='x', label=f'{s_eq_ksi_cualquiera} - Esfuerzo de la pieza fabricada, Ciclos de vida: {Ciclos_de_vida_cualquiera:.2f}, Ratio {R}')
plt.axvline(x=Ciclos_de_vida, color='gray', linestyle='--')

plt.xscale('log')
plt.xlim(1e3, 1e8)
plt.ylim(0, 100)
plt.xlabel('Ciclos de Vida por Fatiga (Nf)')
plt.ylabel('Esfuerzo (KSI)')
plt.title('Curvas S/N Ajustadas para Diferentes Relaciones de Esfuerzo')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()


'''
____________________________________________Mediante estadisticos________________________________________________
'''

H = 115
d = 75
t = H - d

# # Archivos y parámetros para las curvas S/N
# carpeta_archivos = 'Archivos_de_texto_Curvas'
# archivos = ['Curva0_5_SN.txt', 'Curva0_1_SN.txt', 'Curva0_SN.txt', 'Curva_1_SN.txt']
# grados = [3, 3, 3, 3]
# colores = ['blue', 'orange', 'green', 'red']
# archivos = [os.path.join(carpeta_archivos, archivo) for archivo in archivos]

# # Procesar archivos para obtener polinomios e interpolaciones
# _, polinomios_akima_SN, polinomios_makima_SN, datos_SN = fn.process_files(archivos, grados)

# radio_mecanizado, d = fn_e.mecanizar_pieza(H, d, tolerancia=0.1)

# radios_mecanizados = fn_e.simular_variacion_mecanizado(H, d, tolerancia=0.1, iteraciones=10)


def distribucion_normal(media, desviacion_estandar, puntos_por_sigma=100):
    """
    Genera valores distribuidos normalmente que abarcan el 99.7% (3σ a cada lado de la media).
    """
    np.random.seed(42)
    # Genera una cantidad razonable de puntos
    num_puntos = int(6 * desviacion_estandar * puntos_por_sigma)
    
    # Distribuye los valores normalmente
    valores_normales = np.random.normal(media, desviacion_estandar, num_puntos)
    valores_normales = np.sort(valores_normales)
    return valores_normales



# valores_estadistico = distribucion_normal(t-5, (t * 0.05) , (t - 0) / 6) # Asumiendo que el 99.7% de los valores caen dentro del rango
valores_estadistico = distribucion_normal(t-5, (t * 0.05) , 100)
valores_estadistico = [v for v in valores_estadistico if 0 <= v / d <= 0.3] # Filtrar valores válidos de r/d
valores_estadistico_r_d = [v/d for v in valores_estadistico ]


resultados_Ktn_estadistico = [fn.calculate_ktn(H, d, radio) for radio in valores_estadistico]

Ktn_estadisticos = [resultado["valor_Ktn_final_makima"] for resultado in resultados_Ktn_estadistico]

r_d_estadistico = [resultado["r_d"] for resultado in resultados_Ktn_estadistico]

# Plot histogram of Ktn values
plt.figure(figsize=(10, 6))
plt.hist(Ktn_estadisticos, bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel('Ktn')
plt.ylabel('Frecuencia')
plt.title('Frecuencia y Dispersión de los Valores de Ktn')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot histogram of Ktn values
plt.figure(figsize=(10, 6))
plt.hist(r_d_estadistico, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel('Ktn')
plt.ylabel('Frecuencia')
plt.title('Frecuencia y Dispersión de los Valores de Ktn')
plt.grid(True)
plt.tight_layout()
plt.show()


# Graficar los resultados estadísticos de Ktn frente a r/d
plt.figure(figsize=(10, 6))
plt.scatter(r_d_estadistico, Ktn_estadisticos, marker='o', color='blue', label='Resultados Estadísticos')
plt.xlim(0, 0.3)
plt.ylim(0, 3)
plt.xlabel('r/d')
plt.ylabel('Ktn')
plt.title('Resultados Estadísticos de Ktn frente a r/d')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()



# Calcular vida a fatiga para cada radio y Ktn final
vida_a_fatiga = [
    fn.fatigue_life(r_d, h, M, d, Ktn_final)
    for r_d, Ktn_final in zip(valores_estadistico_r_d, Ktn_estadisticos)
]

# Extraer los valores de esfuerzo equivalente y vida a fatiga
resultados_fatiga = [
    {"s_eq_pa": resultado[0], "s_eq_ksi": resultado[1], "nf": resultado[2]}
    for resultado in vida_a_fatiga
    if not np.isnan(resultado[0]) and not np.isnan(resultado[1]) and not np.isnan(resultado[2])
]


# Graficar curvas S/N ajustadas con el esfuerzo máximo
plt.figure(figsize=(10, 6))

# Graficar los puntos de resultados_fatiga
for resultado in resultados_fatiga:
    plt.scatter(resultado["nf"], resultado["s_eq_ksi"], marker='o', label=f'{resultado["s_eq_ksi"]:.2f} KSI, Ciclos de vida: {resultado["nf"]:.2f}')

plt.xscale('log')
plt.xlim(1e3, 1e8)
plt.ylim(0, 100)
plt.xlabel('Ciclos de Vida por Fatiga (Nf)')
plt.ylabel('Esfuerzo (KSI)')
plt.title('Curvas S/N Ajustadas para Diferentes Relaciones de Esfuerzo')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()


# Plot histogram of ciclos de vida
plt.figure(figsize=(10, 6))
nf_values = [resultado["nf"] for resultado in resultados_fatiga]
plt.hist(nf_values, bins=30, color='green', edgecolor='black', alpha=0.7)
plt.xlabel('Ciclos de Vida (Nf)')
plt.ylabel('Frecuencia')
plt.title('Frecuencia y Dispersión de los Ciclos de Vida')
plt.grid(True)
plt.tight_layout()
plt.show()
