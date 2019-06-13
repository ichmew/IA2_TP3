import math
import numpy as np
from matplotlib import pyplot
from pendulo_dataset import genera_pendulo, training, test, validation

# Definición de parámetros iniciales
pos = - math.pi / 3
vel = math.pi / 4
ac = 0
Fsal = 0
dt = 0.0001
t = 0
l = 1
g = - 9.8
M = 1
m = 0.1
Mt = M + m

Wji = np.zeros([2, 15])
Wkj = np.zeros([15, 1])

plot_fuerza = []
plot_posicion = []
plot_velocidad = []
plot_aceleracion = []
plot_tiempo = []
plot_posicion.append(pos)
plot_tiempo.append(t)
plot_velocidad.append(vel)
plot_aceleracion.append(ac)
plot_fuerza.append(Fsal)

print('\nGENERANDO DATASET...\n')
dataset_pendulo = genera_pendulo()

print('ENTRENANDO RED...\n')
Wji, Wkj = training(dataset_pendulo)

print('VALIDANDO...\n')
validation(dataset_pendulo, Wji, Wkj)

# Funcionamiento del péndulo
print('EJECUTANDO PÉNDULO...\n')
while t <= 30:
    Fsal = test(pos, vel, Wji, Wkj)
    # Actualización de variables
    seno = math.sin(pos)
    coseno = math.cos(pos)
    num = g * seno + coseno * ((-Fsal - m * l * seno * vel ** 2) / Mt)
    den = l * ((4 / 3) - ((m * (coseno) ** 2) / Mt))
    new_ac = num / den
    new_vel = vel + ac * dt
    new_pos = pos + vel * dt + (ac * dt ** 2) / 2

    ac = new_ac
    vel = new_vel
    pos = new_pos
    if pos > math.pi:
        print(str(pos), "", str(vel), "", str(ac), "")
        pos = pos - 2 * math.pi
        print(str(pos), "", str(vel), "", str(ac))
        print("")
        # vel = - vel
        # ac = - ac
    elif pos < -math.pi:
        print(str(pos), "", str(vel), "", str(ac), "")
        pos = pos + 2 * math.pi
        print(str(pos), "", str(vel), "", str(ac))
        print("")
        # vel = - vel
        # ac = - ac
    t = t + dt

    plot_posicion.append(pos)
    plot_tiempo.append(t)
    plot_velocidad.append(vel)
    plot_aceleracion.append(ac)
    plot_fuerza.append(Fsal)

print('SIMULACIÓN TERMINADA')
pyplot.figure(1)
pyplot.plot(plot_tiempo, plot_posicion)
pyplot.ylabel('Posicion')
pyplot.xlabel('Tiempo')

pyplot.figure(2)
pyplot.plot(plot_tiempo, plot_velocidad)
pyplot.ylabel('Velocidad')
pyplot.xlabel('Tiempo')

pyplot.figure(3)
pyplot.plot(plot_tiempo, plot_aceleracion)
pyplot.ylabel('Aceleracion')
pyplot.xlabel('Tiempo')

pyplot.figure(4)
pyplot.plot(plot_tiempo, plot_fuerza)
pyplot.ylabel('Fuerza')
pyplot.xlabel('Tiempo')

pyplot.show()
