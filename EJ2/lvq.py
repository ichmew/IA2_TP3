#
# 12/06/19
#
#
#
#
#

import math
import numpy as np
import matplotlib.pylab as plt
from dataset_abc import genera_data
from dataset_abc import dataset_size
from dataset_abc import genera_data_inicializacion

h = 0.1
NEURONAS_ENTRADA = 28*28
MAPA_X = 25
MAPA_Y = 25
MAPA_Z = NEURONAS_ENTRADA
NEURONAS_MAPA = MAPA_X*MAPA_Y
PORC_EJ_TEST = 0.1
EPOCHS = 1
#Parece que lo de hacer un alfa variable en el tiempo es sumamente importante, 
#tengo que acordarme de agregar esa función



Wijk = np.zeros([MAPA_X, MAPA_Y, MAPA_Z])
Wijk = np.random.rand(MAPA_X, MAPA_Y, MAPA_Z)
Wentrada  = np.zeros(MAPA_Z)
Wmapa = np.zeros(MAPA_Z)
mapa = np. zeros([MAPA_X, MAPA_X])

print('Antes del genera_data')
labels, dataset = genera_data()
print('Pasó el genera_data')

CANT_EJ = dataset_size() - 27500
CANT_EJ_TEST = int(CANT_EJ * PORC_EJ_TEST)
CANT_EJ_CLAS = 100
CANT_EJ_TRAINING = CANT_EJ - CANT_EJ_TEST - CANT_EJ_CLAS

ALFAi = 10
ALFAf = 0.01
tALFA = CANT_EJ_TRAINING
def n_alfa(mu):
    ALFA = ALFAi*pow(ALFAf/ALFAi, mu/tALFA)
    return ALFA

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
            if distancias[i][j] <= min_distance:
                min_distance = distancias[i][j]
                min_x = i
                min_y = j
    return min_x, min_y

def aprendizaje(Wentrada, Wijk):
    # Modifica los pesos de la neurona más adecuada a la entrada y los de las neuronas
    # inmediatas a ella

    min_x, min_y = comparacion(Wentrada, Wijk)
    for i in range(min_x - 1, min_x + 2):
        for j in range(min_y - 1, min_y + 2):
            '''
            # Para las neuronas de los bordes:
            if min_x==0 and min_y>0 and min_y<MAPA_Y-1:
                if i == min_x and j == min_y:
                    for k in range(0, NEURONAS_ENTRADA):
                        Wijk[i][j][k] += ALFA*(Wentrada[k] - Wijk[i][j][k])
                elif i == -1:
                    for k in range(0, NEURONAS_ENTRADA):
                        Wijk[MAPA_X-1][j][k] += ALFA*(Wentrada[k] - Wijk[MAPA_X-1][j][k])*h
                else:
                    for k in range(0, NEURONAS_ENTRADA):
                        Wijk[i][j][k] += ALFA*(Wentrada[k] - Wijk[i][j][k])*h
            elif min_x==MAPA_X-1 and min_y>0 and min_y<MAPA_Y-1:
                if i == min_x and j == min_y:
                    for k in range(0, NEURONAS_ENTRADA):
                        Wijk[i][j][k] += ALFA*(Wentrada[k] - Wijk[i][j][k])
                elif i == MAPA_X:
                    for k in range(0, NEURONAS_ENTRADA):
                        Wijk[0][j][k] += ALFA*(Wentrada[k] - Wijk[0][j][k])*h
                else:
                    for k in range(0, NEURONAS_ENTRADA):
                        Wijk[i][j][k] += ALFA*(Wentrada[k] - Wijk[i][j][k])*h
            elif min_x>0 and min_x<MAPA_X-1 and min_y==0:
                if i == min_x and j == min_y:
                    for k in range(0, NEURONAS_ENTRADA):
                        Wijk[i][j][k] += ALFA*(Wentrada[k] - Wijk[i][j][k])
                elif j == -1:
                    for k in range(0, NEURONAS_ENTRADA):
                        Wijk[i][MAPA_Y-1][k] += ALFA*(Wentrada[k] - Wijk[i][MAPA_Y-1][k])*h
                else:
                    for k in range(0, NEURONAS_ENTRADA):
                        Wijk[i][j][k] += ALFA*(Wentrada[k] - Wijk[i][j][k])*h
            elif min_x>0 and min_x<MAPA_X-1 and min_y==MAPA_Y-1:
                if i == min_x and j == min_y:
                    for k in range(0, NEURONAS_ENTRADA):
                        Wijk[i][j][k] += ALFA*(Wentrada[k] - Wijk[i][j][k])
                elif j == MAPA_Y:
                    for k in range(0, NEURONAS_ENTRADA):
                        Wijk[i][0][k] += ALFA*(Wentrada[k] - Wijk[i][0][k])*h
                else:
                    for k in range(0, NEURONAS_ENTRADA):
                        Wijk[i][j][k] += ALFA*(Wentrada[k] - Wijk[i][j][k])*h
            else:'''
            if i>=0 and i<MAPA_X and j>=0 and j<MAPA_Y:
                for k in range(0, NEURONAS_ENTRADA):
                    Wijk[i][j][k] += ALFA*(Wentrada[k] - Wijk[i][j][k])

    return Wijk, min_x, min_y


#MAIN--------------------------------------------------------------------------------
print('Entrenando...')
for e in range(0, EPOCHS):
    for mu in range(0, CANT_EJ_TRAINING):
        ALFA = n_alfa(mu)
        Wentrada = dataset[mu][:]
        Wijk, min_x, min_y = aprendizaje(Wentrada, Wijk)
        mapa[min_x, min_y] += 1
        print(mu)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
plt.imshow(mapa, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()



# LVQ-----------------------------------------------------------------------------------
valor_na = 100 #Valor que va a indicar que no se ha asignado nada a esa neurona
print('Iniciando LVQ...')
labels_map = np.zeros([MAPA_X, MAPA_Y])
for i in range(0, MAPA_X):
    for j in range(0, MAPA_Y):
        labels_map[i][j] = valor_na
tasa_de_aciertos = 0
for mu in range(CANT_EJ_TRAINING, CANT_EJ_TRAINING + CANT_EJ_CLAS):
    ALFA = n_alfa(mu)
    Wentrada = dataset[mu][:]
    clase_patron = labels[mu]
    min_x, min_y = comparacion(Wentrada, Wijk)
    clase_ganadora = labels_map[min_x][min_y]
 
    if clase_ganadora == valor_na:
        labels_map[min_x][min_y] = clase_patron
        # print('Valor asignado.')
    elif clase_ganadora == clase_patron:
        for k in range(0, MAPA_Z):
            Wijk[min_x][min_y][k] = Wijk[min_x][min_y][k] + ALFA*(Wentrada[k] - Wijk[min_x][min_y][k])
        # print('Acierto: ', clase_patron, ' ', clase_ganadora, '<----')
    else:
        for k in range(0, MAPA_Z):
            Wijk[min_x][min_y][k] = Wijk[min_x][min_y][k] - ALFA*(Wentrada[k] - Wijk[min_x][min_y][k])
        # print('Error: ', clase_patron, ' ', clase_ganadora)
# print(labels_map)
# print(tasa_de_aciertos)

# TEST-----------------------------------------------------------------------------
tasa_de_aciertos = 0
tasa_de_errores = 0
for mu in range(CANT_EJ_TRAINING + CANT_EJ_CLAS, CANT_EJ_TRAINING + CANT_EJ_CLAS + CANT_EJ_TEST):
    Wentrada = dataset[mu][:]
    clase_patron = labels[mu]
    min_x, min_y = comparacion(Wentrada, Wijk)
    clase_ganadora = labels_map[min_x, min_y]
    if clase_ganadora == clase_patron:
        tasa_de_aciertos += 1
        print('Acierto: ', clase_patron, ' ', clase_ganadora, '<----')
    else:
        tasa_de_errores += 1
        print('Error: ', clase_patron, ' ', clase_ganadora)

print(labels_map)
print('Cantidad de aciertos: ', tasa_de_aciertos)
print('Cantidad de errores: ', tasa_de_errores)
