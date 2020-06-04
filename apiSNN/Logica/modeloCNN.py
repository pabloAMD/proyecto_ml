from keras.preprocessing import image
import numpy as np
from keras import backend as k
from tensorflow.keras.models import model_from_json


def predecir(url_imagen):
    imagen = image.load_img(path=url_imagen, target_size=(35, 35, 3))
    imagen = image.img_to_array(imagen)
    imagen /= 255
    imagen = np.expand_dims(imagen, axis= 0)

    modelo = cargarModelo(r'apiSNN/Logica/modelo', r'apiSNN/Logica/pesos')
    prediccion = modelo.predict(imagen)
    leo = None
    for i in prediccion:
        leo = i
    for l in leo:
        print(l*100)

    maximo = max(leo)
    round(maximo*100, 2)
    i, = np.where(np.isclose(leo, maximo))
    retorno = dict()

    if i[0] == 0:
        retorno['pred'] = 'MANZANA'
        retorno['porcentaje'] = round(maximo*100, 2)
    elif i[0] == 1:
        retorno['pred'] = 'MANDARINA'
        retorno['porcentaje'] = round(maximo*100, 2)
    elif i[0] == 2:
        retorno['pred'] = 'AGUACATE'
        retorno['porcentaje'] = round(maximo*100, 2)
    elif i[0] == 3:
        retorno['pred'] = 'BANANA'
        retorno['porcentaje'] = round(maximo*100, 2)
    elif i[0] == 4:
        retorno['pred'] = 'ARANDANO'
        retorno['porcentaje'] = round(maximo*100, 2)
    elif i[0] == 5:
        retorno['pred'] = 'MELON'
        retorno['porcentaje'] = round(maximo*100, 2)
    elif i[0] == 6:
        retorno['pred'] = 'CEREZA'
        retorno['porcentaje'] = round(maximo*100, 2)

    elif i[0] == 7:
        retorno['pred'] = 'DURAZNO'
        retorno['porcentaje'] = round(maximo*100, 2)
    elif i[0] == 8:
        retorno['pred'] = 'COCO'
        retorno['porcentaje'] = round(maximo*100, 2)

    elif i[0] == 9:
        retorno['pred'] = 'MAIZ'
        retorno['porcentaje'] = round(maximo*100, 2)

    elif i[0] == 10:
        retorno['pred'] = 'GRANADILLA'
        retorno['porcentaje'] = round(maximo*100, 2)

    elif i[0] == 11:
        retorno['pred'] = 'UVA'
        retorno['porcentaje'] = round(maximo*100, 2)

    elif i[0] == 12:
        retorno['pred'] = 'KIWI'
        retorno['porcentaje'] = round(maximo*100, 2)

    elif i[0] == 13:
        retorno['pred'] = 'MANGO'
        retorno['porcentaje'] = round(maximo*100, 2)

    elif i[0] == 14:
        retorno['pred'] = 'FRESA'
        retorno['porcentaje'] = round(maximo*100, 2)

    elif i[0] == 15:
        retorno['pred'] = 'SANDIA'
        retorno['porcentaje'] = round(maximo*100, 2)

    return retorno


def cargarModelo(url_modelo, url_pesos):
    k.reset_uids()
    with open(url_modelo + '.json', 'r') as f:
        modelo = model_from_json(f.read())

    modelo.load_weights(url_pesos + '.h5')
    return modelo

# [0] --> MANZANA
# [1] --> MANDARIANA
# [2] --> AGUACATE
# [3] --> BANANA
# [4] --> ARANDANO
# [5] --> MELON
# [6] -- CEREZA
# [7] --> DURAZNO
# [8] -- COCO
# [9] --> MAIZ
# [10] --> GRANADILLA
# [11] --> UVA
# [12] --> KIWI
# [13] --> MANGO
# [14] --> FRESA
# [15] --> SANDIA

