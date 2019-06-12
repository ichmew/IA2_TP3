import pandas as pd
import numpy as np


def genera_data(porcentaje_test):

    datos = pd.read_csv('titanic_train.csv', header=0, na_filter=False)

    datosfiltrados = [datos['Sex'], datos['Age'], datos['Pclass'], datos['Embarked'], datos['Survived']]
    datostranspuestos = np.transpose(datosfiltrados)

    datostranspuestosfiltrados = np.array([v for v in datostranspuestos if v[1] != ''])

    datostranspuestosfiltradoscasteados = np.zeros([len(datostranspuestosfiltrados), len(datostranspuestosfiltrados[0])])

    for i in range(0, len(datostranspuestosfiltrados)):
        for j in range(0, len(datostranspuestosfiltrados[0])):
            if datostranspuestosfiltrados[i][j] == 'male':
                datostranspuestosfiltradoscasteados[i][j] = float(0)
            elif datostranspuestosfiltrados[i][j] == 'female':
                datostranspuestosfiltradoscasteados[i][j] = float(1)
            elif datostranspuestosfiltrados[i][j] == 'S':
                datostranspuestosfiltradoscasteados[i][j] = float(1)
            elif datostranspuestosfiltrados[i][j] == 'C':
                datostranspuestosfiltradoscasteados[i][j] = float(2)
            elif datostranspuestosfiltrados[i][j] == 'Q':
                datostranspuestosfiltradoscasteados[i][j] = float(3)
            elif datostranspuestosfiltrados[i][j] == '0' or datostranspuestosfiltrados[i][j] == 0:
                datostranspuestosfiltradoscasteados[i][j] = float(0)
            elif datostranspuestosfiltrados[i][j] == '1' or datostranspuestosfiltrados[i][j] == 1:
                datostranspuestosfiltradoscasteados[i][j] = float(1)
            elif datostranspuestosfiltrados[i][j] == '2' or datostranspuestosfiltrados[i][j] == 2:
                datostranspuestosfiltradoscasteados[i][j] = float(2)
            elif datostranspuestosfiltrados[i][j] == '3' or datostranspuestosfiltrados[i][j] == 3:
                datostranspuestosfiltradoscasteados[i][j] = float(3)
        datostranspuestosfiltradoscasteados[i][1] = float(datostranspuestosfiltrados[i][1])

    datostranspuestosfiltradoscasteadosnormalizados = np.zeros([len(datostranspuestosfiltradoscasteados), len(datostranspuestosfiltradoscasteados[0])])
    edadmaxima = 0
    for i in range(0, len(datostranspuestosfiltradoscasteados)):
        if datostranspuestosfiltradoscasteados[i][1] > edadmaxima:
            edadmaxima = datostranspuestosfiltradoscasteados[i][1]

    for i in range(0, len(datostranspuestosfiltradoscasteados)):
        datostranspuestosfiltradoscasteadosnormalizados[i][0] = datostranspuestosfiltradoscasteados[i][0]
        datostranspuestosfiltradoscasteadosnormalizados[i][1] = datostranspuestosfiltradoscasteados[i][1] / edadmaxima
        datostranspuestosfiltradoscasteadosnormalizados[i][2] = datostranspuestosfiltradoscasteados[i][2] / 3
        datostranspuestosfiltradoscasteadosnormalizados[i][3] = datostranspuestosfiltradoscasteados[i][3] / 3
        datostranspuestosfiltradoscasteadosnormalizados[i][4] = datostranspuestosfiltradoscasteados[i][4]

    cantidad_training = int(len(datostranspuestosfiltradoscasteadosnormalizados) * (1 - porcentaje_test))
    aux = np.zeros([cantidad_training, len(datostranspuestosfiltradoscasteadosnormalizados[0])])
    datostranspuestosfiltradoscasteadosnormalizadosshuffleados = np.zeros([len(datostranspuestosfiltradoscasteadosnormalizados), len(datostranspuestosfiltradoscasteadosnormalizados[0])])

    for i in range(0, cantidad_training):
        aux[i][:] = datostranspuestosfiltradoscasteados[i][:]

    np.random.shuffle(aux)

    for i in range(0, cantidad_training):
        datostranspuestosfiltradoscasteadosnormalizadosshuffleados[i][:] = aux[i][:]

    for i in range(cantidad_training, len(datostranspuestosfiltradoscasteadosnormalizados)):
        datostranspuestosfiltradoscasteadosnormalizadosshuffleados[i][:] = datostranspuestosfiltradoscasteados[i][:]


    ejemplos = np.zeros([len(datostranspuestosfiltrados), len(datostranspuestosfiltrados[0]) - 1])
    dataset_t = np.zeros([len(datostranspuestosfiltrados), 1])

    for i in range(0, len(datostranspuestosfiltradoscasteadosnormalizadosshuffleados)):
        dataset_t[i] = datostranspuestosfiltradoscasteadosnormalizadosshuffleados[i][4]
        for j in range(0, len(datostranspuestosfiltradoscasteadosnormalizadosshuffleados[0]) - 1):
            ejemplos[i][j] = datostranspuestosfiltradoscasteadosnormalizadosshuffleados[i][j]

    df = pd.DataFrame(datostranspuestosfiltradoscasteadosnormalizados)
    df.to_csv('datos_filtrados_train.csv')
    df = pd.DataFrame(datostranspuestosfiltradoscasteadosnormalizadosshuffleados)
    df.to_csv('datos_filtrados_train_shuffle.csv')

    return dataset_t, ejemplos


def dataset_size(porcentaje_test):
        dataset_t, ejemplos = genera_data(porcentaje_test)
        cant_datos = len(ejemplos)
        return cant_datos


def ddataset_size():
    datos = pd.read_csv('dataset_reducido.csv', header=0, na_filter=False)
    cant_datos = datos.shape[1]  # shape[0] para filas, shape[1] para columnas
    return cant_datos
# print(ddataset_size())
