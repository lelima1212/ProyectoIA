import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()
data_entrenamiento='./data/entrenamiento'
data_validacion='./data/validacion'

##Parámetros red neuronal

##numero de veces que vamos a iterar sobre el set de datos en el entrenamiento
epocas=20
##tamano sobre el que cambiaremos las imagenes
altura, longitud= 100, 100
##numero de imagenes que mandaremos a procesar en cada paso
batch_size=32
##numero de veces que se va a procesar en cada epoca
pasos=1000
##al final de cada epoca se correran 200 pasos con el set de validacion
pasos_validacion=200
##numero de filtros a aplicar en cada convolucion
##profundidad de 32 en 1era
filtrosConv1=32
##profundidad de 64 en 2da
filtrosConv2=64
##tamano de filtro que usaremos en la convolucion
tamano_filtro1=(3,3)
tamano_filtro2=(2,2)
##tamano de filtro a usar en max pooling
tamano_pool=(2,2)
##cavendish, orito y rojo
clases=3
##learning rate, que tan grande seran los ajustes para acercarse a la solucion optima
lr=0.005

##pre procesamiento de imagenes

##generador como procesaremos la informacion
entrenamiento_datagen= ImageDataGenerator(
    rescale=1./255, ##La primera lo reescalara
    shear_range=0.3, ##esto es para las imagenes inclinadas
    zoom_range=0.3, ##a algunas les hara zoom
    horizontal_flip = True ##Tomar imagen e invertir
)

validacion_datagen= ImageDataGenerator(
    rescale=1./255 ##Solo la reescalaremos, no la inclinaremos ni haremos zoom ni las invertiremos
)

imagen_entrenamiento= entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size= (altura, longitud),
    batch_size=batch_size,
    class_mode='categorical' ##la clasificacion sera en base a las etiquetas de las carpetas
)

imagen_validacion=validacion_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

##crear red CNN

cnn= Sequential()

cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding='same', input_shape=(altura,longitud,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

## Hacer plana la imagen profunda
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))##A la capa densa durante el entrenamiento vamos a apagar el 50% de las neuronas en cada paso para aprender caminos alternos
cnn.add(Dense(clases, activation='softmax'))

cnn.compile(loss='categorical crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])
cnn.fit(imagen_entrenamiento, steps_per_epoch=pasos, epochs=epocas, validation_data=imagen_validacion, validation_steps=pasos_validacion)
dir='./modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')


