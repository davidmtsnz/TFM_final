import Funciones.Funciones_agrupadas as fn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


'''
____________________________________________Mediante estadisticos________________________________________________
'''

H = 115
d = 75
t = H - d

# Datos para el cálculo de esfuerzo
h = 3  # Espesor de la placa en mm
M = 10e10  # Momento flector en Nmm (10^6 Nmm = 10^3 Nm = 1 kNm)
R = 0.1  # Ratio de estres

FACTOR_LIMITANTE = 2

valores_estadistico = fn.distribucion_normal(t-29, (t * 0.05) , 100)
valores_estadistico_r_d = [v/d for v in valores_estadistico ]

# # Plot histogram of Ktn values
# plt.figure(figsize=(10, 6))
# plt.hist(valores_estadistico_r_d, color='blue', edgecolor='black', alpha=0.7)
# plt.xlabel('r/d')
# plt.ylabel('Frecuencia')
# plt.title('Frecuencia y Dispersión de los Valores de r/d')
# plt.grid(False)
# plt.tight_layout()
# plt.show()

'''
_______________________________Representacion de lo que sería desde 0 hasta t_________________________________________
'''

# num_plots = 10
# num_figures = (t // num_plots) + 1

# for fig_num in range(num_figures):
#     fig, axs = plt.subplots(2, 5, figsize=(25, 12), sharey=True)
    
#     for i in range(num_plots):
#         index = fig_num * num_plots + i
#         if index > t:
#             break
#         valores_estadistico = fn.distribucion_normal(t - index, (t * 0.05), 100)
#         valores_estadistico_r_d = [v / d for v in valores_estadistico]

#         row = i // 5
#         col = i % 5
#         axs[row, col].hist(valores_estadistico_r_d, color='blue', edgecolor='black', alpha=0.7)
#         axs[row, col].set_xlabel('r/d')
#         axs[row, col].set_title(f't-{index}')
#         axs[row, col].grid(False)

#     for ax in axs.flat:
#         ax.set_ylabel('Frecuencia')

#     fig.suptitle(f'Frecuencia y Dispersión de los Valores de r/d para diferentes t (Figura {fig_num + 1})')
#     plt.tight_layout()
#     plt.show()


'''
_________________________________________________________________________________________________________________________
'''

resultados_Ktn_estadistico = [fn.calculate_ktn(H, d, radio) for radio in valores_estadistico]

Ktn_estadisticos = [resultado["valor_Ktn_final_makima"] for resultado in resultados_Ktn_estadistico]

r_d_estadistico = [resultado["r_d"] for resultado in resultados_Ktn_estadistico]


plt.figure()

# Graficar los resultados de r/d
plt.subplot(2, 2, (1, 2))
plt.scatter(r_d_estadistico, Ktn_estadisticos, marker='o', color='blue', label='Resultados Estadísticos')
plt.xlim(0, 0.3)
# plt.ylim(0, 3)
plt.xlabel('r/d')
plt.ylabel('Ktn')
plt.title('Resultados Estadísticos de Ktn frente a r/d')
plt.legend(loc='best')
plt.grid(True, which="both", ls="--")
plt.tight_layout()


# Graficar los resultados de r/d
plt.subplot(2, 2, 3)
plt.hist(valores_estadistico_r_d, bins=30, color='blue', edgecolor='black', alpha=0.7)
sns.kdeplot(valores_estadistico_r_d, bw_adjust=1.5, color="r", label="Curva de valores")
plt.xlabel('r/d')
plt.ylabel('Frecuencia')
plt.title('Frecuencia y Dispersión de los Valores de r/d')
plt.legend(loc='best')
plt.grid(True, which="both", ls="--")
plt.tight_layout()


plt.subplot(2, 2, 4)
plt.hist(Ktn_estadisticos, bins=50, color='blue', edgecolor='black', alpha=0.7)
Ktn_estadisticos = np.array(Ktn_estadisticos, dtype=float)
sns.kdeplot(Ktn_estadisticos, bw_adjust=1.5, color="r", label="Curva de valores")
plt.xlabel('Ktn')
plt.ylabel('Frecuencia')
plt.title('Frecuencia y Dispersión de los Valores de Ktn')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()


# Datos de entrada
H_nominal = 115
d_nominal = 75
t_nominal = H_nominal - d_nominal
r_nominal = 20

# Calcular Ktn cualquiera
resultados_Ktn_nominal = fn.calculate_ktn(H_nominal, d_nominal, r_nominal)

# Calcular Ktn final usando el método Makima
Ktn_makima_final_nominal = resultados_Ktn_nominal["valor_Ktn_final_makima"]

s_max_ksi_temp, s_eq_ksi, Ciclos_de_vida  = fn.fatigue_life(R, h, M, d_nominal, Ktn_makima_final_nominal)


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
plt.figure()
plt.subplot(2, 1, 1)

# Graficar los puntos de resultados_fatiga
for resultado in resultados_fatiga:
    plt.scatter(resultado["nf"], resultado["s_eq_ksi"], marker='o', color='blue')

# Añadir una sola entrada en la leyenda para los puntos
plt.scatter([], [], marker='o', color='blue', label='Resultados de fatiga')

plt.axvline(x=Ciclos_de_vida, color='black', linestyle='--', label='Ciclos de vida de la pieza nominal')
plt.axvline(x=Ciclos_de_vida / FACTOR_LIMITANTE, color='red', linestyle='--', label='Ciclos de vida con factor de seguridad')
plt.xscale('log')
plt.xlim(1e3, 1e8)
# plt.ylim(0, 100)
plt.xlabel('Ciclos de Vida por Fatiga (Nf)')
plt.ylabel('Esfuerzo (KSI)')
plt.title('Curvas S/N Ajustadas para Diferentes Relaciones de Esfuerzo')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
# plt.show()

# Plot histogram of ciclos de vida
plt.subplot(2, 1, 2)
nf_values = [resultado["nf"] for resultado in resultados_fatiga]


plt.hist(nf_values, bins=50, color='green', edgecolor='black', alpha=0.7)
sns.kdeplot(nf_values, bw_adjust=1.5, color="r", label="Curva de valores", log_scale=True)
plt.xlabel('Ciclos de Vida (Nf)')
plt.xscale('log')
plt.axvline(x=Ciclos_de_vida, color='black', linestyle='--', label='Ciclos de vida de la pieza nominal')
plt.axvline(x=Ciclos_de_vida / FACTOR_LIMITANTE, color='red', linestyle='--', label='Ciclos de vida con factor de seguridad')
plt.legend()
plt.ylabel('Frecuencia')
plt.title('Frecuencia y Dispersión de los Ciclos de Vida')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
