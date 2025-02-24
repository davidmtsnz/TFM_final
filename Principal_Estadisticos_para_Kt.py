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

valores_estadistico = fn.distribucion_normal(0.15 * d, (t * 0.05) , 100)
valores_estadistico_r_d = [v/d for v in valores_estadistico]

# # Plot histogram of Ktn values
# plt.figure(figsize=(10, 6))
# plt.hist(valores_estadistico_r_d, color='blue', edgecolor='black', alpha=0.7)
# plt.xlabel('r/d')
# plt.ylabel('Frecuencia')
# plt.title('Frecuencia y Dispersión de los Valores de r/d')
# plt.grid(False)
# plt.tight_layout()
# plt.show()


resultados_Ktn_estadistico = [fn.calculate_ktn(H, d, radio) for radio in valores_estadistico]

Ktn_estadisticos = [resultado["valor_Ktn_final_makima"] for resultado in resultados_Ktn_estadistico]

r_d_estadistico = [resultado["r_d"] for resultado in resultados_Ktn_estadistico]



porcentaje_de_subida = 1.5 #Porcentaje de incremento


# Datos de entrada
H_nominal = 115
d_nominal = 75
t_nominal = H_nominal - d_nominal
r_nominal = 11.25

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


plt.figure()

# Graficar los resultados de r/d
plt.subplot(2, 2, (1, 2))
plt.scatter(r_d_estadistico, Ktn_estadisticos, marker='o', color='blue', label='Resultados Estadísticos')
plt.axvline(Ktn_makima_final_nominal, color='black', linestyle='--', label=f'$K_{{tn}}$ nominal: {Ktn_makima_final_nominal:.2f}')
plt.xlim(0, 0.3)
# plt.ylim(0, 3)
plt.xlabel('r/d')
plt.ylabel(f'$K_{{tn}}$')
plt.title(f'Resultados Estadísticos de $K_{{tn}}$ frente a r/d')
plt.legend(loc='best')
plt.grid(True, which="both", ls="--")
plt.tight_layout()


# Graficar los resultados de r/d
plt.subplot(2, 2, 3)
plt.hist(valores_estadistico_r_d, bins=30, color='blue', edgecolor='black', alpha=0.7)
sns.kdeplot(valores_estadistico_r_d, bw_adjust=1.5, color="r", label="Curva de valores")
plt.axvline(r_nominal/d_nominal, color='black', linestyle='--', label=f'r/d nominal: {r_nominal/d_nominal:.2f}')
plt.xlabel('r/d')
plt.ylabel('Frecuencia')
plt.title('Frecuencia y Dispersión de los Valores de r/d')
plt.legend(loc='best')
plt.grid(True, which="both", ls="--")
plt.tight_layout()


plt.subplot(2, 2, 4)
plt.hist(Ktn_estadisticos, bins=50, color='blue', edgecolor='black', alpha=0.7)
Ktn_estadisticos = np.array(Ktn_estadisticos, dtype=float)

# sns.kdeplot(Ktn_estadisticos, bw_adjust=1.5, color="r", label="Curva de valores")

# Ajuste KDE y escalado
kde = sns.kdeplot(Ktn_estadisticos, bw_adjust=1.5, color="r", label="Curva de valores")

# Multiplica los valores de KDE por una constante para aumentar la altura
constante = 4  # Ajusta esta constante según necesites
kde_lines = kde.get_lines()[0]
kde_ydata = kde_lines.get_ydata()
kde_lines.set_ydata(kde_ydata * constante)

plt.axvline(x=Ktn_makima_final_nominal, color='black', linestyle='--', label=f'Ktn nominal: {Ktn_makima_final_nominal:.2f} y factor de seguridad {porcentaje_de_subida}')
plt.xlabel('$K_{tn}$')
plt.ylabel('Frecuencia')
plt.title('Frecuencia y Dispersión de los Valores de $K_{tn}$')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()



# Graficar curvas S/N ajustadas con el esfuerzo máximo
plt.figure()
plt.subplot(2, 1, 1)

# Graficar los puntos de resultados_fatiga
for resultado in resultados_fatiga:
    plt.scatter(resultado["nf"], resultado["s_eq_ksi"], marker='o', color='blue')

# Añadir una sola entrada en la leyenda para los puntos
plt.scatter([], [], marker='o', color='blue', label='Resultados de fatiga')

plt.axvline(x=Ciclos_de_vida, color='black', linestyle='--', label=f'Ciclos de vida de la pieza nominal: {Ciclos_de_vida:.2f}')
plt.axvline(x=Ciclos_de_vida * porcentaje_de_subida , color='red', linestyle='--', label=f'Ciclos de vida con factor de seguridad {porcentaje_de_subida}: {Ciclos_de_vida * porcentaje_de_subida:.2f}')
plt.xscale('log')
plt.xlim(1e3, 1e8)
# plt.ylim(0, 100)
plt.xlabel('Ciclos de vida a fatiga (Nf)')
plt.ylabel('Esfuerzo (KSI)')
plt.title('Curvas S-N para los resultados de ciclos a fatiga')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
# plt.show()



# Plot histogram of ciclos de vida
plt.subplot(2, 1, 2)
nf_values = [resultado["nf"] for resultado in resultados_fatiga]
nf_values = np.array(nf_values, dtype=float)  # Convertir a array de tipo float

plt.hist(nf_values, bins=50, color='green', edgecolor='black', alpha=0.7)

kde_2 = sns.kdeplot(nf_values, bw_adjust=1.2, color="r", label="Curva de valores", log_scale=True)

# Multiplica los valores de KDE por una constante para aumentar la altura
constante_kde_2 = 4  # Ajusta esta constante según necesites
kde_2_lines = kde_2.get_lines()[0]
kde_2_ydata = kde_2_lines.get_ydata()
kde_2_lines.set_ydata(kde_2_ydata * constante)


plt.xlabel('Ciclos de vida (Nf)')
# plt.xscale('log')
plt.axvline(x=Ciclos_de_vida, color='black', linestyle='--', label='Ciclos de vida de la pieza nominal')
plt.axvline(x=Ciclos_de_vida * porcentaje_de_subida, color='red', linestyle='-.', label=f'Ciclos de vida con factor de seguridad {porcentaje_de_subida}')
#plt.axvline(x=Ciclos_de_vida * , color='orange', linestyle=':', label='Ciclos de vida con factor de seguridad')
plt.legend()
plt.xlabel('$K_{tn}$')
plt.ylabel('Frecuencia')
plt.title('Frecuencia y Dispersión de los Valores de $K_{tn}$')
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
# # __________________________________________________________________________________________________________________________

# media = np.mean(Ktn_estadisticos)

# # Dividir en dos grupos
# por_encima_de_la_media = Ktn_estadisticos[Ktn_estadisticos > media]
# por_debajo_de_la_media = Ktn_estadisticos[Ktn_estadisticos <= media]

# # Calcular intervalos y contar los elementos en cada intervalo
# intervalos = np.linspace(np.min(Ktn_estadisticos), np.max(Ktn_estadisticos), 50)
# conteo_por_encima = np.histogram(por_encima_de_la_media, bins=intervalos)[0]
# conteo_por_debajo = np.histogram(por_debajo_de_la_media, bins=intervalos)[0]

# # Obtener el 60% de los datos por encima de la media
# percentil_arriba = np.percentile(por_encima_de_la_media, 50)
# por_ciento_arriba = por_encima_de_la_media[por_encima_de_la_media >= percentil_arriba]

# # Obtener el 40% de los datos por debajo de la media
# percentil_abajo = np.percentile(por_debajo_de_la_media, 50)
# por_ciento_abajo = por_debajo_de_la_media[por_debajo_de_la_media < percentil_abajo]

# # Calcular el conteo de los datos obtenidos
# conteo_arriba = len(por_ciento_arriba)
# conteo_abajo = len(por_ciento_abajo)

# print(f"El 50% de los datos por encima de la media son: {por_ciento_arriba} (Cantidad: {conteo_arriba})")
# print(f"El 50% de los datos por debajo de la media son: {por_ciento_abajo} (Cantidad: {conteo_abajo})")

# print(f"Media de los datos por encima de la media son: {np.mean(por_ciento_arriba)}")
# print(f"Media de los datos por debajo de la media son: {np.mean(por_ciento_abajo)}")

# # Visualizar los resultados
# plt.figure(figsize=(12, 6))

# plt.hist(por_encima_de_la_media, bins=intervalos, alpha=0.7, label='Por encima de la media', color='blue', edgecolor='black')
# plt.hist(por_debajo_de_la_media, bins=intervalos, alpha=0.7, label='Por debajo de la media', color='red', edgecolor='black')

# plt.axvline(media, color='black', linestyle='dashed', linewidth=2, label=f'Media: {media:.2f}')

# plt.xlabel('Valores')
# plt.ylabel('Frecuencia')
# plt.legend()
# plt.title('Distribución de Valores por Encima y por Debajo de la Media')
# plt.show()

# # Mostrar resultados en texto
# print(f'Media: {media:.2f}')
# print(f'Conteo por encima de la media en cada intervalo: {conteo_por_encima}')
# print(f'Conteo por debajo de la media en cada intervalo: {conteo_por_debajo}')


# # Calcular la media
# # media_nf_values = np.mean(nf_values)
# media_nf_values = Ciclos_de_vida * porcentaje_de_subida

# # Dividir en dos grupos
# por_encima_de_la_media_nf_values = nf_values[nf_values > media_nf_values]
# por_debajo_de_la_media_nf_values = nf_values[nf_values <= media_nf_values]

# # Calcular intervalos y contar los elementos en cada intervalo
# intervalos_nf_values = np.linspace(np.min(nf_values), np.max(nf_values), 50)
# conteo_por_encima_nf_values = np.histogram(por_encima_de_la_media_nf_values, bins=intervalos_nf_values)[0]
# conteo_por_debajo_nf_values = np.histogram(por_debajo_de_la_media_nf_values, bins=intervalos_nf_values)[0]

# # Obtener el 50% de los datos por encima de la media
# percentil_arriba = np.percentile(por_encima_de_la_media_nf_values, 50)
# por_ciento_arriba = por_encima_de_la_media_nf_values[por_encima_de_la_media_nf_values >= percentil_arriba]

# # Obtener el 50% de los datos por debajo de la media
# percentil_abajo = np.percentile(por_debajo_de_la_media_nf_values, 50)
# por_ciento_abajo = por_debajo_de_la_media_nf_values[por_debajo_de_la_media_nf_values < percentil_abajo]

# # Calcular el conteo de los datos obtenidos
# conteo_arriba = len(por_ciento_arriba)
# conteo_abajo = len(por_ciento_abajo)

# print(f"El 50% de los datos por encima de la media son: {por_ciento_arriba} (Cantidad: {conteo_arriba})")
# print(f"El 50% de los datos por debajo de la media son: {por_ciento_abajo} (Cantidad: {conteo_abajo})")

# print(f"Media de los datos por encima de la media son: {np.mean(por_ciento_arriba)}")
# print(f"Media de los datos por debajo de la media son: {np.mean(por_ciento_abajo)}")


# # Visualizar los resultados
# plt.figure(figsize=(12, 6))

# plt.hist(por_encima_de_la_media_nf_values, bins=intervalos_nf_values, alpha=0.7, label='Por encima de la media', color='blue', edgecolor='black')
# plt.hist(por_debajo_de_la_media_nf_values, bins=intervalos_nf_values, alpha=0.7, label='Por debajo de la media', color='red', edgecolor='black')

# plt.axvline(media_nf_values, color='black', linestyle='dashed', linewidth=2, label=f'Media: {media_nf_values:.2f}')

# plt.xlabel('Valores')
# plt.ylabel('Frecuencia')
# plt.legend()
# plt.title('Distribución de Valores por Encima y por Debajo de la Media')
# plt.show()

# # Mostrar resultados en texto
# print(f'Media: {media_nf_values:.2f}')
# # print(f'Conteo por encima de la media en cada intervalo: {conteo_por_encima_nf_values}')
# # print(f'Conteo por debajo de la media en cada intervalo: {conteo_por_debajo_nf_values}')


