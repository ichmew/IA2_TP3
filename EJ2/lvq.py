#
# 12/06/19
#
#
#
#
#

import math
import numpy as np
from matplotlib.pyplot import imshow
from dataset_abc import genera_data
from dataset_abc import dataset_size
from dataset_abc import genera_data_inicializacion


NEURONAS_ENTRADA = 28 * 28
MAPA_X = 10
MAPA_Y = 10
MAPA_Z = NEURONAS_ENTRADA
NEURONAS_MAPA = MAPA_X * MAPA_Y
PORC_EJ_TEST = 0.1
EPOCHS = 1
ALFA = 0.05

Wijk = np.zeros([MAPA_X, MAPA_Y, MAPA_Z])
Wentrada = np.zeros(MAPA_Z)
Wmapa = np.zeros(MAPA_Z)
mapa = np. zeros([MAPA_X, MAPA_X])

labels, dataset = genera_data()

CANT_EJ = dataset_size()
CANT_EJ_TEST = int(CANT_EJ * PORC_EJ_TEST)
CANT_EJ_TRAINING = CANT_EJ - CANT_EJ_TEST

# inicializacion aleatoria de los pesos
Wijk = np.random.rand([MAPA_X, MAPA_Y, MAPA_Z])


def calculo_metrica(Wentrada, Wmapa):
    # Calcula la distancia euclideana entre 2 vectores de pesos

    d_euclideana = 0
    dist = 0
    for k in range(0, MAPA_Z):
        dist += pow(Wentrada[k] - Wmapa[k], 2)
    d_euclideana = pow(dist, 0.5)
    return d_euclideana


def comparacion(Wentrada, Wijk):
    # Obtiene la neurona más próxima a la entrada

    min_x = 0
    min_y = 0
    distancias = np.zeros([MAPA_X, MAPA_Y])
    for i in range(0, MAPA_X):
        for j in range(0, MAPA_Y):
            Wmapa = Wijk[i][j][:]
            distancias[i][j] = calculo_metrica(Wentrada, Wmapa)
    min_distance = distancias[min_x][min_y]
    for i in range(0, MAPA_X):
        for j in range(0, MAPA_Y):
            if distancias[i][j] < min_distance:
                min_distance = distancias[i][j]
                min_x = i
                min_y = j
    return min_x, min_y


def aprendizaje(Wentrada, Wijk):
    # Modifica los pesos de la neurona más adecuada a la entrada y los de las
    # neuronas inmediatas a ella
    min_x, min_y = comparacion(Wentrada, Wijk)
    for i in range(min_x - 1, min_x + 2):
        for j in range(min_y - 1, min_y + 2):
            # Neuronas ganadoras
            if i >= 0 and i < MAPA_X and j >= 0 and j < MAPA_Y:
                for k in range(0, NEURONAS_ENTRADA):
                    Wijk[i][j][k] += ALFA * pow(pow(Wentrada[k] - Wijk[i][j][k], 2), 0.5)
    return Wijk, min_x, min_y


# KOHONEN:
def radio(mu):
    # Determinación del radio de vecindad con decreción lineal
    R_inicial = 2
    R_final = 0.5
    tr = 50
    r = R_inicial + (R_final - R_inicial) * mu / tr
    return r


def h(RADIO, i, j, min_x, min_y):
    # Determinación de función de vecindad
    dist_actual_ganadora = pow(pow(i - min_x, 2) + pow(j - min_y, 2), 0.5)
    if dist_actual_ganadora > RADIO:
        return 0
    else:
        return 1


def alfa(mu):
    # Determinación de la tasa de aprendizaje con decreción lineal
    A_inicial = 0.2
    A_final = 0.01
    ta = 50
    a = A_inicial + (A_final - A_inicial) * mu / ta
    return a


def aprendizaje_kohonen(Wentrada, Wijk, mu):
    # Modifica los pesos de la neurona más adecuada a la entrada y los de las
    # neuronas inmediatas a ella
    min_x, min_y = comparacion(Wentrada, Wijk)
    RADIO = radio(mu)
    ALFA = alfa(mu)
    for i in range(0, MAPA_X):
        for j in range(0, MAPA_Y):
            VECINDAD = h(RADIO, i, j, min_x, min_y)
            for k in range(0, MAPA_Z):
                DIST_EUCL = pow(pow(Wentrada[k] - Wijk[i][j][k], 2), 0.5)
                Wijk[i][j][k] += ALFA * VECINDAD * DIST_EUC
    return Wijk, min_x, min_y


# MAIN--------------------------------------------------------------------------

for e in range(0, EPOCHS):
    for mu in range(0, CANT_EJ_TRAINING):
        Wentrada = dataset[mu][:]
        Wijk, min_x, min_y = aprendizaje(Wentrada, Wijk)
        mapa[min_x, min_y] += 1

'''
# test
for mu in range(CANT_EJ_TRAINING, CANT_EJ_TEST):
    Wentrada = dataset[mu][:]
    wij'''

imshow(mapa)
