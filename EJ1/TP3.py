#
# DESCRIPCIÓN:
# Implementación de un Multi-Layer Perceptron con 1 capa oculta utilizando
# función sigmoide para la activación de neuronas en la capa oculta y de
# salida y entrenamiento mediante el algoritmo de BackPropagation (BP)
#
# Creado el 04/06/2019
# Inteligencia Artificial II - Ingeniería en Mecatronica
# Facultad de Ingeniería - Universidad Nacional de Cuyo
# Autor: Sebastian Giunta, P8 capo, Ichmew
#

import math
from matplotlib import pyplot as plt
import numpy as np
from datataset import genera_data
from datataset import dataset_size

# Definición de hiperparámetros de la red
NEURONAS_ENTRADA = 4
NEURONAS_CAPA_OCULTA = 20
NEURONAS_SALIDA = 1
PORCEN_EJ_TEST = 0.1
PORCEN_EJ_VAL = 0.1
EPSILON = 0.01
EJEMPLOS_CANT = dataset_size(PORCEN_EJ_TEST)
EJEMPLOS_TEST = int(dataset_size(PORCEN_EJ_TEST) * PORCEN_EJ_TEST)
EJEMPLOS_VAL = int((EJEMPLOS_CANT - EJEMPLOS_TEST) * PORCEN_EJ_VAL)
EPOCHS = 100
CANTIDAD_ENTRADAS_SALIDAS = 1  # depende del dataset, es para el t
mostrar_e_s = 0


# Función de activación Heavyside de la capa de entrada
def Heavy(nox0):
    if nox0 >= 0:
        return 1
    else:
        return 0


# Derivada de la función de activación Heavyside de la capa de entrada
def H_derivada(nox4):
    return 1


# Función de activación sigmoide de la capa de entrada
def f(nox1):
    return 1 / (1 + math.exp(-nox1))


# Derivada de la función de activación de la capa de entrada
def f_derivada(nox2):
    return math.exp(-nox2) / ((1 + math.exp(-nox2)) * (1 + math.exp(-nox2)))


# Función de activación de la capa oculta
def g(nox3):
    return nox3


# Derivada de la función de activación de la capa oculta
def g_derivada(nox3):
    return 1


# Cálculo de las salidas de cada capa y de las salidas finales
Wji = np.zeros([NEURONAS_ENTRADA, NEURONAS_CAPA_OCULTA])
Wkj = np.zeros([NEURONAS_CAPA_OCULTA, NEURONAS_SALIDA])
x = np.zeros(NEURONAS_ENTRADA)
y = np.zeros(NEURONAS_CAPA_OCULTA)
z = np.zeros(NEURONAS_SALIDA)
cant_ej_training = EJEMPLOS_CANT - EJEMPLOS_TEST - EJEMPLOS_VAL
tasa_aciertos = 0


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
        entrada_z -= 1
        # entrada_z -= Wkj[NEURONAS_CAPA_OCULTA - 1][k]
        # Valor de salidad de la neurona k
        z[k] = g(entrada_z)


def bp(Wji, Wkj, x, y, z, t):
    # Se encarga de actualizar los pesos sinápticos

    delta_mu_k = np.zeros(NEURONAS_SALIDA)
    # Actualización pesos capa oculta-capa salida
    for k in range(0, NEURONAS_SALIDA):
        h_mu_k = 0
        for j in range(0, NEURONAS_CAPA_OCULTA):
            h_mu_k += Wkj[j][k] * y[j]
        h_mu_k -= Wkj[NEURONAS_CAPA_OCULTA - 1][k]
        delta_mu_k[k] = (t[k] - g(h_mu_k)) * g_derivada(h_mu_k)
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
        delta_mu_j *= f_derivada(h_mu_j)
        for i in range(0, NEURONAS_ENTRADA):
            Wji[i][j] += EPSILON * delta_mu_j * x[i]
        Wji[NEURONAS_ENTRADA - 1][j] += EPSILON * delta_mu_j * -1




def calcula_rendimiento_val(ejemplos, Wji, Wkj, mostrar_e_s):
    # Se encarga de calcular la tasa de aciertos del conjunto de validación
    # de cada epoch

    aciertos = 0
    ejemplo = 0
    y = np.zeros(NEURONAS_CAPA_OCULTA)
    z = np.zeros(NEURONAS_SALIDA)
    for mu in range(cant_ej_training, cant_ej_training + EJEMPLOS_VAL):
        t = dataset_t[mu][:]
        for i in range(0, NEURONAS_ENTRADA):
            x[i] = ejemplos[mu][i]
        calculo_salidas(Wji, Wkj, x, y, z)
        # Si usamos sólo una neurona, saltamos directamente (es necesario
        # comentar lo que están en el medio) a la verificación de errores.
        # Utilizamos para ello el z de calculo_salidas y el t del dataset.
        '''max = z[0]
        for k in range(1, NEURONAS_SALIDA):
            if max < z[k]:
                max = z[k]
        for k in range(0, NEURONAS_SALIDA):
            if max != z[k]:
                z[k] = 0
            else:
                z[k] = 1.0
        t = np.zeros(NEURONAS_SALIDA)
        t[int(dataset_t[mu - 1])] = 1'''
        # Verificación de aciertos
        error = 0
        if mostrar_e_s == 1:
            print(str((ejemplo + 1)), '. z=[')
        for k in range(0, NEURONAS_SALIDA):
            error += round(pow(pow(t[k] - z[k], 2), 0.5))
            if mostrar_e_s == 1:
                print(str(z[k]))
        if mostrar_e_s == 1:
            print('] -- t=[')
            for k in range(0, NEURONAS_SALIDA):
                print(str(t[k]))
            print(']\n')
            ejemplo = ejemplo + 1
        if error == 0:
            aciertos = aciertos + 1
    # Calculamos la tasa de aciertos
    tasa_aciertos = aciertos / EJEMPLOS_VAL
    return tasa_aciertos


def calcula_rendimiento(ejemplos, Wji, Wkj, mostrar_e_s):
    # Se encarga de calcular la tasa de aciertos de cada epoch

    aciertos = 0
    ejemplo = 0
    y = np.zeros(NEURONAS_CAPA_OCULTA)
    z = np.zeros(NEURONAS_SALIDA)
    for mu in range(0, cant_ej_training):
        t = dataset_t[mu][:]
        for i in range(0, NEURONAS_ENTRADA):
            x[i] = ejemplos[mu][i]
        calculo_salidas(Wji, Wkj, x, y, z)
        # Si usamos sólo una neurona, saltamos directamente (es necesario
        # comentar lo que están en el medio) a la verificación de errores.
        # Utilizamos para ello el z de calculo_salidas y el t del dataset.
        '''max = z[0]
        for k in range(1, NEURONAS_SALIDA):
            if max < z[k]:
                max = z[k]
        for k in range(0, NEURONAS_SALIDA):
            if max != z[k]:
                z[k] = 0
            else:
                z[k] = 1.0
        t = np.zeros(NEURONAS_SALIDA)
        t[int(dataset_t[mu - 1])] = 1'''
        # Verificación de aciertos
        error = 0
        if mostrar_e_s == 1:
            print(str((ejemplo + 1)), '. z = [')
        for k in range(0, NEURONAS_SALIDA):
            error += round(pow(pow(t[k] - z[k], 2), 0.5))
      
            if mostrar_e_s == 1:
                print(str(z[k]))
        if mostrar_e_s == 1:
            print('] -- t=[')

            for k in range(0, NEURONAS_SALIDA):
                print(str(t[k]))
            print(']\n')
            ejemplo = ejemplo + 1
        if error == 0:
            aciertos = aciertos + 1
    # Calculamos la tasa de aciertos
    tasa_aciertos = aciertos / cant_ej_training
    return tasa_aciertos



def calcula_final(ejemplos, Wji, Wkj, mostrar_e_s):
    # Prueba del rendimiento con TEST

    print('\nTEST INICIALIZADO')
    aciertos = 0
    ejemplo = 0
    y = np.zeros(NEURONAS_CAPA_OCULTA)
    z = np.zeros(NEURONAS_SALIDA)
    for mu in range(cant_ej_training + EJEMPLOS_VAL, EJEMPLOS_CANT):
        t = dataset_t[mu][:]
        for i in range(0, NEURONAS_ENTRADA):
            x[i] = ejemplos[mu][i]
        calculo_salidas(Wji, Wkj, x, y, z)
        # Si usamos sólo una neurona, saltamos directamente (es necesario
        # comentar lo que están en el medio) a la verificación de errores.
        # Utilizamos para ello el z de calculo_salidas y el t del dataset.
        '''max = z[0]
        for k in range(1, NEURONAS_SALIDA):
            if max < z[k]:
                max = z[k]
        for k in range(0, NEURONAS_SALIDA):
            if max != z[k]:
                z[k] = 0
            else:
                z[k] = 1.0
        t = np.zeros(NEURONAS_SALIDA)
        t[int(ejemplos[mu][NEURONAS_SALIDA - 1])] = 1'''
        # Verificación de aciertos
        error = 0
        if mostrar_e_s == 1:
            print(str((ejemplo + 1)), '. z=[')
        for k in range(0, NEURONAS_SALIDA):
            error += round(pow(pow(t[k] - z[k], 2), 0.5))
            if mostrar_e_s == 1:
                print(str(z[k]))
        if mostrar_e_s == 1:
            print('] -- t=[')
            for k in range(0, NEURONAS_SALIDA):
                print(str(t[k]))
            print(']\n')
            ejemplo = ejemplo + 1
        if error == 0:
            aciertos = aciertos + 1
    # Calculamos la tasa de aciertos
    tasa_aciertos = aciertos / EJEMPLOS_TEST
    return tasa_aciertos



# MAIN ------------------------------------------------------------------------
t = np.zeros([EJEMPLOS_CANT, NEURONAS_SALIDA])
Wji = np.random.rand(NEURONAS_ENTRADA, NEURONAS_CAPA_OCULTA)
Wkj = np.random.rand(NEURONAS_CAPA_OCULTA, NEURONAS_SALIDA)
tasa_aciertos= np.zeros(EPOCHS)
epocas= range(EPOCHS)
dataset_t, ejemplos = genera_data(PORCEN_EJ_TEST)
for e in range(0, EPOCHS):
    # TRAINING AND VALIDATION (descomentar la linea de abajo si se quiere usar la validation, es para el shufleo de datos cada epoca)
    # dataset_t, ejemplos = genera_data(PORCEN_EJ_TEST)
    for mu in range(0, cant_ej_training):
        x = ejemplos[mu][:]
        t = dataset_t[mu][:]
        calculo_salidas(Wji, Wkj, x, y, z)
        bp(Wji, Wkj, x, y, z, t)
   #Escoger si se quiere usar o no la validacion     
   # tasa_aciertos[e] = calcula_rendimiento_val(ejemplos, Wji, Wkj, mostrar_e_s)
    tasa_aciertos[e] = calcula_rendimiento(ejemplos, Wji, Wkj, mostrar_e_s)
    print('Epoch ', e, ': ', tasa_aciertos[e], '\n')
    dataset_t, ejemplos = genera_data(PORCEN_EJ_TEST)

# waiting = input()
# TEST / VALIDATION
error = 0
tasa_aciertos_test = calcula_final(ejemplos, Wji, Wkj, mostrar_e_s)
print('\nTasa de aciertos test = ', tasa_aciertos_test)
plt.plot(epocas,tasa_aciertos)
plt.ylabel('Tasa de Aciertos')
plt.xlabel('Epoca')
plt.show()
