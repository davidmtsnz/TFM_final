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
    f'Interpolación Polyfit',
    'Interpolación Akima',
    'Interpolación Makima'
]


# Graficar interpolaciones
plt.figure(figsize=(10, 6))
for i, (x, y) in enumerate(data_interpolated):
    plt.plot(x, y, label=labels_interpolated[i])

# Puntos para HD_values
plt.scatter(HD_values, Ktn_para_r_d_nominal, marker='o', color='blue', label='Puntos Polyfit de los H/d')
plt.scatter(HD_values, Ktn_para_r_d_nominal_akima, marker='x', color='orange', label='Puntos Akima de los H/d')
plt.scatter(HD_values, Ktn_para_r_d_nominal_makima, marker='s', color='green', label='Puntos Makima de los H/d')


# Puntos calculados
plt.scatter([H_nominal / d_nominal], [Ktn_calculados_poly], marker='o', color='cyan', label=f'Punto Polyfit Ktn={Ktn_calculados_poly:.2f}')
plt.scatter([H_nominal / d_nominal], [Ktn_calculados_akima], marker='x', color='red', label=f'Punto Akima Ktn={Ktn_calculados_akima:.2f}')
plt.scatter([H_nominal / d_nominal], [Ktn_calculados_makima], marker='s', color='magenta', label=f'Punto Makima Ktn={Ktn_calculados_makima:.2f}')

plt.xlabel('H/d')
plt.ylabel(f"K$_{{tn}}$")
plt.title(f'Interpolación de K$_{{tn}}$ para r/d')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()