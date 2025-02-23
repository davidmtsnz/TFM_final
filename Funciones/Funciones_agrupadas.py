import os
import numpy as np
import pkg_resources # Requiere hacer: pip install setuptools
import subprocess
from scipy.interpolate import Akima1DInterpolator

def verificar_requisitos(archivo='requirements.txt'):
    """
    Verifica si los paquetes del archivo requirements.txt están instalados y pregunta al usuario si desea instalarlos.
    """
    with open(archivo, 'r') as f:
        paquetes = f.read().splitlines()
    for paquete in paquetes:
        try:
            pkg_resources.require(paquete)
        except pkg_resources.DistributionNotFound:
            respuesta = input(f"El paquete '{paquete}' no está instalado. ¿Deseas instalarlo? (s/n): ").strip().lower()
            if respuesta == 's':
                subprocess.check_call(["pip", "install", paquete])

'''
___________________________________Funciones de archivos_____________________________________
'''

# Ajuste polinómico
def ajuste_polinomico( x, y, grado = 2):
    '''
    :param x: Valores de x
    :param y: Valores de y
    :return: polinomio(x), polinomio, interpolador_akima, interpolador_makima
    '''
    coeficientes = np.polyfit(x, y, grado)
    polinomio = np.poly1d(coeficientes)
    interpolador_akima = Akima1DInterpolator(x, y, method='akima')
    interpolador_makima = Akima1DInterpolator(x, y, method='makima')
    return polinomio(x), polinomio, interpolador_akima, interpolador_makima



def leer_datos(filename):
    data = np.loadtxt(filename, delimiter=';', skiprows=1)
    x = data[:, 0]  # Primera columna: valores de x
    y = data[:, 1]  # Segunda columna: valores de y
    return x, y

def process_files(archivos, grados):
    """
    Procesa una lista de archivos aplicando ajustes polinómicos e interpolaciones.
    :param archivos: Lista de archivos a procesar.
    :param grados: Lista de grados de ajuste polinómico para cada archivo.
    :return: Una tupla con cuatro listas:
        - polinomios: Lista de tuplas (x, y_ajustados, polinomio) para cada archivo.
        - polinomios_akima: Lista de tuplas (x, y_interpolado_akima, interpolado_akima) para cada archivo.
        - polinomios_makima: Lista de tuplas (x, y_interpolado_makima, interpolado_makima) para cada archivo.
        - datos: Lista de tuplas (x, y, y_ajustados, y_interpolado_akima, y_interpolado_makima, nombre_archivo) para cada archivo.
    """
    
    polinomios, polinomios_akima, polinomios_makima, datos = [], [], [], []

    for nombre_archivo, grados in zip(archivos, grados):
        
        #Guarda los datos en x e y de cada archivo
        x, y = leer_datos(nombre_archivo)
        
        #Realiza un ajuste polinomico del cual se obtienen los valores de y ajustados a cada aproximación
        y_ajustados, polinomio, interpolado_akima, interpolado_makima = ajuste_polinomico(x, y)
        y_interpolado_akima = interpolado_akima(x)
        y_interpolado_makima = interpolado_makima(x)

        #Se reunen todos los valores en una lista según la aproximación
        polinomios.append((x, y_ajustados, polinomio))
        polinomios_akima.append((x, y_interpolado_akima, interpolado_akima))
        polinomios_makima.append((x, y_interpolado_makima, interpolado_makima))
        datos.append((x, y, y_ajustados, y_interpolado_akima, y_interpolado_makima, nombre_archivo))

    return polinomios, polinomios_akima, polinomios_makima, datos


'''
___________________________________Cálculo de Ktn_____________________________________
'''


'''
Opción 1: Según el Petersons, se pueden utilizar las ecuaciones que vienen en él, pero realmente
no se ajustan a lo que debe salir.
'''
# Definición de la función obtener_constantes con manejo de rango
def obtener_constantes(t, r):
    t_r = t / r
    if 0.5 <= t_r < 2.0:
        C1 = 1.795 + 1.481 * t_r - 0.211 * (t_r)**2
        C2 = -3.544 - 3.677 * t_r + 0.578 * (t_r)**2
        C3 = 5.459 + 3.691 * t_r - 0.565 * (t_r)**2
        C4 = -2.678 - 1.531 * t_r + 0.205 * (t_r)**2
    elif 2.0 <= t_r < 20.0:
        C1 = 2.966 + 0.502 * t_r - 0.009 * (t_r)**2
        C2 = -6.475 - 1.126 * t_r + 0.019 * (t_r)**2
        C3 = 8.023 + 1.253 * t_r - 0.020 * (t_r)**2
        C4 = -3.572 - 0.634 * t_r + 0.010 * (t_r)**2
    else:
        # Si t/r está fuera del rango, devolvemos None
        return None
    return C1, C2, C3, C4


def calculo_Ktn(t, r, t_H):
    '''
    :param t: Valores de t
    :param r: Valores de r
    :param t_H: Valores de y
    :return: Ktn
    '''
    constantes = obtener_constantes(t, r)
    if constantes is None:
        return None  # Devuelve None si obtener_constantes no proporciona coeficientes
    C1, C2, C3, C4 = constantes
    Ktn = C1 + C2 * (t_H) + C3 * (t_H)**2 + C4 * (t_H)**3
    return Ktn /10


'''
Opción 2 de cálculo de Ktn mediante valores x e y obtenidos mediante archivos
'''


def calculate_ktn(H, d, r):
    '''
    Calculo del Ktn utilizando H, d y r.
 
    :return: Lo que saca calculate_ktn:
    {   "Ktn_para_r_d": Ktn_para_r_d,
        "Ktn_para_r_d_akima": Ktn_para_r_d_akima,
        "Ktn_para_r_d_makima": Ktn_para_r_d_makima,
        "polinomios": polinomios, [polinomios : (x, y_ajustados, polinomio)]
        "polinomios_akima": polinomios_akima, [polinomios_akima : (x, y_interpolado_akima, interpolado_akima)]
        "polinomios_makima": polinomios_makima, [polinomios_makima : (x, y_interpolado_makima, interpolado_makima)]
        "datos": datos, [datos : (x, y, y_ajustados, y_interpolado_akima, y_interpolado_makima, nombre_archivo)]
        "valor_Ktn_final_poly1d": valor_Ktn_final_poly1d,
        "valor_Ktn_final_akima": valor_Ktn_final_akima,
        "valor_Ktn_final_makima": valor_Ktn_final_makima,
        "H_d": H_d,
        "r_d": r_d
    }
    '''
    # Ruta a la carpeta de archivos
    carpeta_archivos = 'Archivos_de_texto_Curvas'
    archivos = ['CurvaHD1_15.txt', 'CurvaHD1_3.txt', 'CurvaHD1_5.txt', 'CurvaHD2.txt']
    archivos = [os.path.join(carpeta_archivos, archivo) for archivo in archivos]
    grados = [3, 3, 3, 3]
    # colores = ['blue', 'orange', 'green', 'red']

    # Valores de la pieza
    t = H - d
    H_d = H / d
    r_d = (H_d - 1) / (t / r)

    polinomios, polinomios_akima, polinomios_makima, datos = process_files(archivos, grados)

    # Tener en cuenta lo que devuelve process_files:
    #   polinomios : (x, y_ajustados, polinomio)
    #   polinomios_akima : (x, y_interpolado_akima, interpolado_akima)
    #   polinomios_makima : (x, y_interpolado_makima, interpolado_makima)
    #   datos : (x, y, y_ajustados, y_interpolado_akima, y_interpolado_makima, nombre_archivo)

    Ktn_para_r_d = [interpolador(r_d) for _, _, interpolador in polinomios]
    Ktn_para_r_d_akima = [interpolador(r_d) for _, _, interpolador in polinomios_akima]
    Ktn_para_r_d_makima = [interpolador(r_d) for _, _, interpolador in polinomios_makima]

    HD_values = [1.15, 1.3, 1.5, 2]

    _, interpolado_r_d, interpolado_akima_r_d, interpolado_makima_r_d = ajuste_polinomico(HD_values, Ktn_para_r_d)

    valor_Ktn_final_poly1d = interpolado_r_d(H_d)
    valor_Ktn_final_akima = interpolado_akima_r_d(H_d)
    valor_Ktn_final_makima = interpolado_makima_r_d(H_d)

    return {
        "Ktn_para_r_d": Ktn_para_r_d,
        "Ktn_para_r_d_akima": Ktn_para_r_d_akima,
        "Ktn_para_r_d_makima": Ktn_para_r_d_makima,
        "polinomios": polinomios,
        "polinomios_akima": polinomios_akima,
        "polinomios_makima": polinomios_makima,
        "datos": datos,
        "valor_Ktn_final_poly1d": valor_Ktn_final_poly1d,
        "valor_Ktn_final_akima": valor_Ktn_final_akima,
        "valor_Ktn_final_makima": valor_Ktn_final_makima,
        "H_d": H_d,
        "r_d": r_d
    }


'''
_______________________________Calculo del esfuerzo____________________________________
'''

# Función para calcular Nf dada una r y s_max
def calcular_nf(s_max_ksi, r):
    # Cálculo de s_eq según la ecuación dada
    s_eq_ksi = s_max_ksi * (1 - r) ** 0.64
    factor = (1 - r) ** 0.64
    # Ecuación para calcular Nf
    log_nf = 10.0 - 3.96 * np.log10(s_eq_ksi)
    nf = 10 ** log_nf
    return nf, s_eq_ksi, factor


# Función para calcular el esfuerzo máximo
def nominal_stress(h, M, d):
    """
    Calcula el esfuerzo nominal en una placa.

    :param h: Espesor de la placa (en metros)
    :param M: Momento flector (en Newton-metros)
    :param d: Distancia de placa que queda con material (en metros)
    :return: Esfuerzo nominal (en Pascales)
    """
    return ((6 * M) / (h * (d ** 2)))

# Función para convertir de Pa a KSI
def pa_to_ksi(pa):
    # 1 KSI = 6.89476 MPa = 6.89476e6 Pa
    ksi = pa / 6.89476e6
    return ksi

# Función para convertir de KSI a Pa
def ksi_to_pa(ksi):
    # 1 KSI = 6.89476 MPa = 6.89476e6 Pa
    pa = ksi * 6.89476e6
    return pa

# Función para calcular la vida por fatiga en ciclos (Nf)
def fatigue_life(R, h, M, d_nominal, Ktn_final):
    """
    Calcula la vida por fatiga en ciclos (Nf) para un esfuerzo máximo y un ratio.

    Parámetros:
    s_max_pa (float): Esfuerzo máximo en Pascales.
    r (float): Ratio de esfuerzo (esfuerzo mínimo / esfuerzo máximo).
    Ktn_makima_final (float): Valor final de Ktn usando interpolación makima.
    nominal_stress_ksi (float): Esfuerzo nominal en ksi (kilo libras por pulgada cuadrada).

    Retorna:
    return: Una tupla que contiene:
        - s_eq_pa (float): Esfuerzo equivalente en Pascales.
        - s_eq_ksi (float): Esfuerzo equivalente en ksi.
        - nf (float): Vida por fatiga en ciclos.
    """
    Kt_gráfica_SN = 3
    Tratamiento_superficial = 1.3
    nominal_stress_pa = nominal_stress(h, M, d_nominal)
    nominal_stress_ksi = pa_to_ksi(nominal_stress_pa)


    SAF_previo = 1
    SAF = (Ktn_final / Kt_gráfica_SN) * Tratamiento_superficial * SAF_previo
    
    s_max_ksi = Kt_gráfica_SN * SAF * nominal_stress_ksi 

    nf, s_eq_ksi, factor = calcular_nf(s_max_ksi, R)
    print(nominal_stress_ksi)
    return s_max_ksi, s_eq_ksi, nf 


def find_closest_x(interpolator, target_y, x_range, tol=1e-5, max_iter=100):
    """
    Encuentra el valor de x que produce el valor de y más cercano al objetivo usando un interpolador.

    :param interpolator: Función interpoladora (interp1d u otra función de interpolación)
    :param target_y: Valor objetivo de y
    :param x_range: Tupla (x_min, x_max) que define el rango de búsqueda para x
    :param tol: Tolerancia aceptable para la diferencia entre y estimado y target_y
    :param max_iter: Número máximo de iteraciones
    :return: Mejor valor encontrado de x
    """
    x_min, x_max = x_range
    for _ in range(max_iter):
        x_mid = (x_min + x_max) / 2
        y_mid = interpolator(x_mid)

        if abs(y_mid - target_y) < tol:
            return x_mid
        elif y_mid < target_y:
            x_min = x_mid
        else:
            x_max = x_mid
    
    return x_mid  # Retorna el mejor valor encontrado tras las iteraciones


def distribucion_normal(media, desviacion_estandar, puntos_por_sigma=100):
    """
    Genera valores distribuidos normalmente que abarcan el 95% (2σ a cada lado de la media).
    """
    np.random.seed(42)
    
    # Distribuye los valores normalmente
    valores_normales = np.random.normal(media, desviacion_estandar, puntos_por_sigma * 2)
    # Ese 2 genera aproximadamente 200 puntos distribuidos normalmente, 
    # lo que refleja el 95% de la distribución normal. 
    # Esto asegura que los valores generados estarán dentro de dos desviaciones estándar 
    # de la media, abarcando así el 95% de la distribución.

    valores_normales = np.sort(valores_normales)
    return valores_normales