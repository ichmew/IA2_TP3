import pandas as pd
import numpy as np
import random
import math
from matplotlib import pyplot

# Definición de parámetros de la red
NEURONAS_ENTRADA = 2
NEURONAS_CAPA_OCULTA = 20
NEURONAS_SALIDA = 1
PORCEN_EJ_VAL = 0.1
EPSILON = 0.001
EJEMPLOS_CANT = 1000
EJEMPLOS_VAL = int(PORCEN_EJ_VAL * EJEMPLOS_CANT)
EPOCHS = 20

cant_ej_training = EJEMPLOS_CANT - EJEMPLOS_VAL


def genera_pendulo():
    # Se encarga de generar un dataset de 3 columnas (posición, velocidad y
    # label de fuerza), de los cuales se irán modificando las etiquetas con la
    # tabla de verdades de la lógica difusa

    dataset_pendulo = np.random.rand(EJEMPLOS_CANT, 3)
    dataset_pendulo_norm = np.zeros([EJEMPLOS_CANT, 3])

    # Definición de reglas (0,0) = (pos:NF, vel:NF)
    REGLAS = [['PF', 'PF', 'PF', 'PP', 'Z'],
              ['PF', 'PP', 'PP', 'Z', 'Z'],
              ['PF', 'PP', 'Z', 'NP', 'NF'],
              ['Z', 'Z', 'NP', 'NP', 'NF'],
              ['Z', 'NP', 'NF', 'NF', 'NF']]

    Valores_Fuerza = {
        'NF': 15,
        'NP': 3,
        'Z': 0,
        'PP': -3,
        'PF': -15
    }

    # Definicion de umbrales para los conjutos borrosos
    uNF = - math.pi / 3
    uNP = - math.pi / 6
    uPP = math.pi / 6
    uPF = math.pi / 3

    mNF = - math.pi / 2
    mNP = - math.pi / 4
    mPP = math.pi / 4
    mPF = math.pi / 2

    for mu in range(0, EJEMPLOS_CANT):
        # Generación de valores random
        pos = random.uniform(- math.pi, math.pi)
        vel = random.uniform(- 0.75 * math.pi, 0.75 * math.pi)
        # Asignacion de los valores a los conjuntos borrosos
        # Posición
        if pos <= uNF:
            Pnf = 1
            pos1 = [0, Pnf]
            pos2 = [0, 0]
        if pos > uNF and pos <= uNP:
            Pnp = abs((pos - uNF) / (uNF - uNP))
            Pnf = 1 - Pnp
            if Pnp >= Pnf:
                pos1 = [1, Pnp]
                pos2 = [0, Pnf]
            else:
                pos1 = [0, Pnf]
                pos2 = [1, Pnp]
        if pos > uNP and pos <= 0:
            Pze = abs((pos - uNP) / uNP - 0)
            Pnp = 1 - Pze
            if Pze >= Pnp:
                pos1 = [2, Pze]
                pos2 = [1, Pnp]
            else:
                pos1 = [1, Pnp]
                pos2 = [2, Pze]
        if pos > 0 and pos <= uPP:
            Ppp = abs(pos / uPP)
            Pze = 1 - Ppp
            if Ppp >= Pze:
                pos1 = [3, Ppp]
                pos2 = [2, Pze]
            else:
                pos1 = [2, Pze]
                pos2 = [3, Ppp]
        if pos > uPP and pos <= uPF:
            Ppf = abs((pos - uPP) / (uPF - uPP))
            Ppp = 1 - Ppf
            if Ppf >= Ppp:
                pos1 = [4, Ppf]
                pos2 = [3, Ppp]
            else:
                pos1 = [3, Ppp]
                pos2 = [4, Ppf]
        if pos >= uPF:
            Ppf = 1
            pos1 = [4, Ppf]
            pos2 = [4, 0]

        # Velocidad
        if vel <= mNF:
            Vnf = 1
            vel1 = [0, Vnf]
            vel2 = [0, 0]
        if vel > mNF and vel <= mNP:
            Vnp = abs((vel - mNF) / (mNF - mNP))
            Vnf = 1 - Vnp
            if Vnp >= Vnf:
                vel1 = [1, Vnp]
                vel2 = [0, Vnf]
            else:
                vel1 = [0, Vnf]
                vel2 = [1, Vnp]
        if vel > mNP and vel <= 0:
            Vze = abs((vel - mNP) / mNP - 0)
            Vnp = 1 - Vze
            if Vze >= Vnp:
                vel1 = [2, Vze]
                vel2 = [1, Vnp]
            else:
                vel1 = [1, Vnp]
                vel2 = [2, Vze]
        if vel > 0 and vel <= mPP:
            Vpp = abs(vel / mPP)
            Vze = 1 - Vpp
            if Vpp >= Vze:
                vel1 = [3, Vpp]
                vel2 = [2, Vze]
            else:
                vel1 = [2, Vze]
                vel2 = [3, Vpp]
        if vel > mPP and vel <= mPF:
            Vpf = abs((vel - mPP) / (mPF - mPP))
            Vpp = 1 - Vpf
            if Vpf >= Vpp:
                vel1 = [4, Vpf]
                vel2 = [3, Vpp]
            else:
                vel1 = [3, Vpp]
                vel2 = [4, Vpf]
        if vel >= mPF:
            Vpf = 1
            vel1 = [4, Vpf]
            vel2 = [4, 0]

        # Cálculo de salidas borrosas
        F1 = REGLAS[pos1[0]][vel1[0]]
        F2 = REGLAS[pos2[0]][vel2[0]]

        F1 = Valores_Fuerza.get(F1)
        F2 = Valores_Fuerza.get(F2)

        # Cálculo de antecedentes por norma T
        peso1 = min(pos1[1], vel1[1])
        peso2 = min(pos2[1], vel2[1])

        # Desborrosificación por media de centros (weighted average)
        Fsal = (F1 * peso1 + F2 * peso2) / (peso1 + peso2)

        dataset_pendulo[mu][0] = float(Fsal)
        dataset_pendulo[mu][1] = float(pos)
        dataset_pendulo[mu][2] = float(vel)

    # Normalizado del dataset
    fmaxima = 0
    for mu in range(0, EJEMPLOS_CANT):
        if dataset_pendulo[mu][0] > fmaxima:
            fmaxima = dataset_pendulo[mu][0]

    for mu in range(0, EJEMPLOS_CANT):
        dataset_pendulo_norm[mu][0] = dataset_pendulo[mu][0] / fmaxima
        dataset_pendulo_norm[mu][1] = dataset_pendulo[mu][1]
        dataset_pendulo_norm[mu][2] = dataset_pendulo[mu][2]

    df = pd.DataFrame(dataset_pendulo)
    df.to_csv('prueba_dataset_pendulo.csv')
    dfn = pd.DataFrame(dataset_pendulo_norm)
    dfn.to_csv('prueba_dataset_pendulo_norm.csv')
    return dataset_pendulo, dataset_pendulo_norm


def training(dataset_pendulo, dataset_pendulo_norm):
    # Se encarga de actualizar los pesos de la red y de calcular la desviación
    # de los valores obtenidos respecto a los labels

    # Wji = np.zeros([NEURONAS_ENTRADA, NEURONAS_CAPA_OCULTA])
    # Wkj = np.zeros([NEURONAS_CAPA_OCULTA, NEURONAS_SALIDA])
    Wji = np.random.rand(NEURONAS_ENTRADA, NEURONAS_CAPA_OCULTA)
    Wkj = np.random.rand(NEURONAS_CAPA_OCULTA, NEURONAS_SALIDA)
    x = np.zeros(NEURONAS_ENTRADA)
    y = np.zeros(NEURONAS_CAPA_OCULTA)
    z = np.zeros(NEURONAS_SALIDA)

    for e in range(0, EPOCHS):
        # Actualización de pesos con back propagation
        for mu in range(0, cant_ej_training):
            x = dataset_pendulo[mu][1:]
            t = dataset_pendulo[mu][0]
            x, y, z = calculo_salidas(Wji, Wkj, x, y, z)
            Wji, Wkj = bp(Wji, Wkj, x, y, z, t)
        # Verificación de errores
        plot_z = np.zeros(cant_ej_training)
        plot_t = np.zeros(cant_ej_training)
        plot_error = np.arange(cant_ej_training)
        error_abs = np.zeros(cant_ej_training)
        error_promedio = 0
        for mu in range(0, cant_ej_training):
            x = dataset_pendulo[mu][1:]
            t = dataset_pendulo[mu][0]
            x, y, z = calculo_salidas(Wji, Wkj, x, y, z)
            plot_z[mu] = z
            plot_t[mu] = t
            error_abs[mu] = abs(z - t)
            error_promedio += error_abs[mu]
        error_promedio = error_promedio / cant_ej_training
        print('Epoch ', e, ': ', error_promedio, '\n')

        '''
        max_z = np.amax(plot_z)
        # Normalización de z
        for mu in range(0, cant_ej_training):
            plot_z[mu] = plot_z[mu] / max_z
            error_abs[mu] = abs(plot_t[mu] - plot_z[mu])
            error_promedio += error_abs[mu]
        error_promedio = error_promedio / cant_ej_training
        print('Epoch ', e, ': ', error_promedio, '\n')
        '''

        pyplot.figure(e + 1)
        pyplot.plot(plot_error, error_abs)
        pyplot.plot(error_promedio)
        pyplot.ylabel('Error absoluto (Newton)')
        pyplot.xlabel('Ejemplos de entrenamiento')

    pyplot.show()
    return Wji, Wkj


def validation(dataset_pendulo, dataset_pendulo_norm, Wji, Wkj):
    # Funge como prueba test. Calcula las desviaciones de los valores
    # obtenidos para ejemplos de entrenamiento que no han pasado por la red.

    x = np.zeros(NEURONAS_ENTRADA)
    y = np.zeros(NEURONAS_CAPA_OCULTA)
    z = np.zeros(NEURONAS_SALIDA)
    plot_z = np.zeros(EJEMPLOS_VAL)
    plot_t = np.zeros(EJEMPLOS_VAL)
    plot_error = np.arange(EJEMPLOS_VAL)
    error_abs = np.zeros(EJEMPLOS_VAL)
    error_promedio = 0

    for mu in range(cant_ej_training, EJEMPLOS_CANT):
        x = dataset_pendulo[mu][1:]
        t = dataset_pendulo[mu][0]
        x, y, z = calculo_salidas(Wji, Wkj, x, y, z)
        plot_z[mu - cant_ej_training] = z
        plot_t[mu - cant_ej_training] = t
        error_abs[mu - cant_ej_training] = abs(z - t)
        error_promedio += error_abs[mu - cant_ej_training]
    error_promedio = error_promedio / EJEMPLOS_VAL
    print('Error absoluto promedio (Newton): ', error_promedio, '\n')

    '''
    max_z = np.amax(plot_z)
    # Normalización de z
    for mu in range(cant_ej_training, EJEMPLOS_CANT):
        plot_z[mu - cant_ej_training] = plot_z[mu - cant_ej_training] / max_z
        error_abs[mu - cant_ej_training] = abs(plot_t[mu - cant_ej_training] -
        plot_z[mu - cant_ej_training])
        error_promedio += error_abs[mu - cant_ej_training]
    error_promedio = error_promedio / EJEMPLOS_VAL
    print('Error absoluto promedio: ', error_promedio, '\n')
    '''

    pyplot.figure(1)
    pyplot.plot(plot_error, error_abs)
    pyplot.plot(error_promedio)
    pyplot.ylabel('Error absoluto validación (Newton)')
    pyplot.xlabel('Ejemplos de entrenamiento validación')

    pyplot.figure(2)
    pyplot.plot(plot_error, plot_z)
    pyplot.plot(plot_error, plot_t)
    pyplot.ylabel('z y t')
    pyplot.xlabel('Ejemplos de entrenamiento validación')

    pyplot.show()


def test(pos, vel, Wji, Wkj):
    # Se encarga de obtener la Fuerza de salida del péndulo para unos valores
    # de posición y velocidad específicos

    x = np.zeros(NEURONAS_ENTRADA)
    y = np.zeros(NEURONAS_CAPA_OCULTA)
    z = np.zeros(NEURONAS_SALIDA)
    x[0] = pos
    x[1] = vel
    x, y, z = calculo_salidas(Wji, Wkj, x, y, z)
    return z  # z = Fsal


def calculo_salidas(Wji, Wkj, x, y, z):
    # Se encarga de calcular la salida z de cada neurona

    # Cálculo de salidas de la capa oculta
    for j in range(0, NEURONAS_CAPA_OCULTA):
        entrada_y = 0
        for i in range(0, NEURONAS_ENTRADA):
            entrada_y += Wji[i][j] * x[i]
        # Sesgo de las neuronas de la capa oculta
        entrada_y -= Wji[NEURONAS_ENTRADA - 1][j]
        # Valor de salida de la neurona j
        y[j] = f(entrada_y)

    # Cálculo de salidas de la capa de salida
    for k in range(0, NEURONAS_SALIDA):
        entrada_z = 0
        for j in range(0, NEURONAS_CAPA_OCULTA):
            entrada_z += Wkj[j][k] * y[j]
        # Sesgo de las neuronas de la cada de salida
        entrada_z -= Wkj[NEURONAS_CAPA_OCULTA - 1][k]
        # entrada_z -= 1
        # Valor de salidad de la neurona k
        z[k] = g(entrada_z)
    # print(z)
    return x, y, z


def bp(Wji, Wkj, x, y, z, t):
    # Se encarga de actualizar los pesos sinápticos

    delta_mu_k = np.zeros(NEURONAS_SALIDA)

    # Actualización pesos capa oculta-capa salida
    for k in range(0, NEURONAS_SALIDA):
        h_mu_k = 0
        for j in range(0, NEURONAS_CAPA_OCULTA):
            h_mu_k += Wkj[j][k] * y[j]
        h_mu_k -= Wkj[NEURONAS_CAPA_OCULTA - 1][k]
        delta_mu_k[k] = (t - g(h_mu_k)) * g_derivada(h_mu_k)
        for j in range(0, NEURONAS_CAPA_OCULTA):
            Wkj[j][k] += EPSILON * delta_mu_k[k] * y[j]
        Wkj[NEURONAS_CAPA_OCULTA - 1][k] += EPSILON * delta_mu_k[k] * - 1

    # Actualización pesos capa entrada-capa oculta
    for j in range(0, NEURONAS_CAPA_OCULTA):
        h_mu_j = 0
        for i in range(0, NEURONAS_ENTRADA):
            h_mu_j += Wji[i][j] * x[i]
        h_mu_j -= Wji[NEURONAS_ENTRADA - 1][j]
        delta_mu_j = 0
        for k in range(0, NEURONAS_SALIDA):
            delta_mu_j += delta_mu_k[k] * Wkj[j][k]
        delta_mu_j = delta_mu_j * f_derivada(h_mu_j)
        for i in range(0, NEURONAS_ENTRADA):
            Wji[i][j] += EPSILON * delta_mu_j * x[i]
        Wji[NEURONAS_ENTRADA - 1][j] += EPSILON * delta_mu_j * -1
    return Wji, Wkj


# Función de activación Heavyside (sin usar)
def Heavy(nox0):
    if nox0 >= 0:
        return 1
    else:
        return 0


# Derivada de la función de activación Heavyside (sin usar)
def H_derivada(nox1):
    return 1


# Función de activación sigmoide de la capa de entrada
def f(nox2):
    return 1 / (1 + math.exp(-nox2))


# Derivada de la función de activación de la capa de entrada
def f_derivada(nox3):
    return math.exp(-nox3) / ((1 + math.exp(-nox3)) * (1 + math.exp(-nox3)))


# Función de activación sigmoide de la capa de entrada (sin usar)
def f0(nox4):
    return math.tanh(nox4)


# Derivada de la función de activación de la capa de entrada (sin usar)
def f0_derivada(nox5):
    return 1 - math.tanh(nox5)


# Función de activación de la capa oculta
def g(nox6):
    return nox6


# Derivada de la función de activación de la capa oculta
def g_derivada(nox7):
    return 1
