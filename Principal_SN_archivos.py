import Funciones.Funciones_agrupadas as fn
import Funciones.Funciones_gráficas as fn_g
import matplotlib.pyplot as plt
import numpy as np



'''
........................................Combinación de coeficientes + archivos de puntos........................................
'''

# Datos de entrada
H_nominal = 115
d_nominal = 75
t_nominal = H_nominal - d_nominal
r_nominal = 20

# Datos para el cálculo de esfuerzo
h = 3  # Espesor de la placa en mm
M = 10e10  # Momento flector en Nmm (10^6 Nmm = 10^3 Nm = 1 kNm)
R = 0.1  # Ratio de estres

# Calcular Ktn nominal
resultados_Ktn_nominal = fn.calculate_ktn(H_nominal, d_nominal, r_nominal)

# Calcular Ktn final usando el método Makima
Ktn_makima_final_nominal = resultados_Ktn_nominal["valor_Ktn_final_makima"]

# Parámetros del problema
s_max_ksi = np.linspace(0.1, 100, 100)  # Esfuerzo máximo en ksi

# Convertir s_max_ksi a Pa usando la función kasi_to_pa
s_max_pa = fn.ksi_to_pa(s_max_ksi)
# Asegurarse de que los valores en s_max_pa sean al menos 5e6
s_max_pa = np.where(s_max_pa < 5e6, 5e6, s_max_pa)
s_max_ksi = fn.pa_to_ksi(s_max_pa)


r_values = [-1, 0.0, 0.1, 0.5]  # Diferentes valores de r

fn_g.graficar_ecuacion_y_archivos_Nf(resultados_Ktn_nominal, d_nominal, h, M, R)

s_max_ksi_temp, s_eq_ksi, Ciclos_de_vida  = fn.fatigue_life(R, h, M, d_nominal, Ktn_makima_final_nominal)

print(s_max_ksi_temp)
print(s_eq_ksi)
print(Ciclos_de_vida)


# Generación de gráficos
plt.figure(figsize=(10, 6))

# Graficar para cada valor de r
for r in r_values:
    nf_values = fn.calcular_nf(s_max_ksi, r)

    # Gráfica
    plt.plot(nf_values[0], s_max_ksi, linestyle='-', label=f"r={r}")


# Añadir scatter y axvline
plt.scatter(Ciclos_de_vida, s_max_ksi_temp, marker='o', label=f'Esfuerzo equivalente nominal: {s_eq_ksi:.2f}, Ciclos de vida: {Ciclos_de_vida:.2f}, Ratio {R}')
plt.axvline(x=Ciclos_de_vida, color='black', linestyle='--')
plt.axhline(y=fn.pa_to_ksi(5e6), color='blue', linestyle='-', label = f'Linea de 5 MPa, en KSI: {fn.pa_to_ksi(5e6):.2f}')

plt.xscale('log')
plt.xlim(1e3, 1e8)
plt.ylim(0, 100)
plt.xlabel('Ciclos de vida por fatiga (Nf)')
plt.ylabel('Esfuerzo (KSI)')
plt.title('Curvas S-N ajustadas para pieza nominal')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
