import Funciones.Funciones_agrupadas as fn
import matplotlib.pyplot as plt
import numpy as np

# Datos de entrada
H = 115
d = 75
t = H - d
r = 11.25

# Datos para el cálculo de esfuerzo
h = 3  # Espesor de la placa en mm
M = 10e10  # Momento flector en Nmm (10^6 Nmm = 10^3 Nm = 1 kNm)
R = 0.1  # Ratio de estres

# Calcular Ktn cualquiera
resultados_Ktn = fn.calculate_ktn(H, d, r)

# Acceder a los datos del diccionario
datos_curvas = resultados_Ktn["datos"]
Ktn_para_r_d_makima = resultados_Ktn["Ktn_para_r_d_makima"]
H_d = resultados_Ktn["H_d"]

# Valores de HD para interpolación
HD_values = [1.15, 1.3, 1.5, 2]

# Ajuste polinómico
_, _, _, interpolador_makima_Ktn_r_d = fn.ajuste_polinomico(HD_values, Ktn_para_r_d_makima)

# Calcular valores interpolados para HD_values
Ktn_calculado_makima = interpolador_makima_Ktn_r_d(H_d)

# Graficar datos_curvas
plt.figure(figsize=(10, 6))
for (x, _, _, _, y_interpolado_makima, nombre_archivo), hd in zip(datos_curvas, HD_values):
    plt.plot(x, y_interpolado_makima, label=f'Interpolación Makima para {hd}')

# Añadir punto específico
plt.scatter(resultados_Ktn["r_d"], Ktn_calculado_makima, marker='o', color='red', label=f'K$_{{tn}}$={Ktn_calculado_makima:.2f} (r/d = {resultados_Ktn["r_d"]:.2f}, H/d = {resultados_Ktn["H_d"]:.2f})')

# Configurar la gráfica
plt.xlim(0, 0.3)
plt.ylim(1, 3)
plt.xlabel('r/d')
plt.ylabel('K$_{tn}$')
plt.title(f'Curvas de K$_{{tn}}$ por ajuste makima para pieza con radio {r} mm')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# Crear un linspace para representar las funciones
HD_linspace = np.linspace(min(HD_values), max(HD_values), 500)

# Calcular los valores interpolados usando los interpoladores
Ktn_interpolados_makima = interpolador_makima_Ktn_r_d(HD_linspace)

# Datos y etiquetas para la interpolación
data_interpolated = [
    (HD_linspace, Ktn_interpolados_makima)
]

labels_interpolated = [
    'Interpolación Makima'
]

# Graficar interpolaciones
plt.figure(figsize=(10, 6))
for i, (x, y) in enumerate(data_interpolated):
    plt.plot(x, y, label=labels_interpolated[i])

# Puntos para HD_values
plt.scatter(HD_values, Ktn_para_r_d_makima, marker='s', color='green', label='Puntos Makima para los H/d (1.15, 1.3, 1.5, 2)')

# Puntos calculados
plt.scatter([H / d], [Ktn_calculado_makima], marker='s', color='magenta', label=f'Punto calculado con Makima: K$_{{tn}}$={Ktn_calculado_makima:.2f}')

plt.xlabel('H/d')
plt.ylabel('K$_{tn}$')
plt.title(f'Interpolación de K$_{{tn}}$ por makima para una pieza con radio {r} mm')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calcular Ktn final usando el método Makima
Ktn_makima_final = resultados_Ktn["valor_Ktn_final_makima"]

# Parámetros del problema
s_max_ksi = np.linspace(0.1, 100, 100)  # Esfuerzo máximo en ksi

# Convertir s_max_ksi a Pa usando la función kasi_to_pa
s_max_pa = fn.ksi_to_pa(s_max_ksi)
# Asegurarse de que los valores en s_max_pa sean al menos 5e6
s_max_pa = np.where(s_max_pa < 5e6, 5e6, s_max_pa)
s_max_ksi = fn.pa_to_ksi(s_max_pa)

r_values = [-1, 0.0, 0.1, 0.5]  # Diferentes valores de r

s_max_ksi_temp, s_eq_ksi, Ciclos_de_vida  = fn.fatigue_life(R, h, M, d, Ktn_makima_final)

print(s_max_ksi_temp)
print(s_eq_ksi)
print(Ciclos_de_vida)

# Generación de gráficos
plt.figure(figsize=(10, 6))

# Graficar para cada valor de r
for r in r_values:
    nf_values = fn.calcular_nf(s_max_ksi, r)

    # Gráfica
    plt.plot(nf_values[0], s_max_ksi, linestyle='-', label=f"Para una R = {r}")

# Añadir scatter y axvline
plt.scatter(Ciclos_de_vida, s_max_ksi_temp, marker='o', label=f'Esfuerzo equivalente: {s_eq_ksi:.2f}, Ciclos de vida: {Ciclos_de_vida:.2f}, Ratio {R}')
#plt.axvline(x=Ciclos_de_vida, color='black', linestyle='--')
plt.axhline(y=fn.pa_to_ksi(5e6), color='blue', linestyle='-', label = f'Linea de 5 MPa, en KSI: {fn.pa_to_ksi(5e6):.2f}')

plt.xscale('log')
plt.xlim(1e3, 1e8)
plt.ylim(0, 100)
plt.xlabel('Ciclos de vida a fatiga (Nf)')
plt.ylabel('Esfuerzo (KSI)')
plt.title(f'Curvas S-N ajustadas por makima para una pieza con radio {r} mm')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
