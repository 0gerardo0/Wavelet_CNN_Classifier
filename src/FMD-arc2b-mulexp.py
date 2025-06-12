import os
current_dir = os.getcwd()
print("Direccion actual:", current_dir)

os.chdir('/content/drive/MyDrive/Colab Notebooks/FDM_WCNN')
print("Nuevo directorio:", os.getcwd())

import pywt, cv2, time, random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras import backend as K
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing import image
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical, plot_model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, GlobalAveragePooling2D, Flatten, Dropout, Dense
#from keras.layers import Rescaling, RandomFlip, RandomBrightness, RandomContrast, RandomTranslation, RandomCrop

from utils.manager_checkpointers import get_checkpoints
from utils.manage_csv_logger import get_csv_logger
from utils import export_metrics_FMDexp as export_metrics
from utils.test_manual_model import show_test

# Configurar las semillas para la reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f'SEED: {SEED}')

# Definir una función de normalización
def normalize_data(data):
    # Inicializar el escalador
    scaler = MinMaxScaler()
    return scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

def export_model(wavelet, experiment, model):
    base_dir = 'saves/checkpointer'
    model_save_dir = os.path.join(base_dir, f'{wavelet}-experiment')
    subdirectories = ['model_save']
    sub_dirs = [os.path.join(model_save_dir, sub) for sub in subdirectories]
    export_metrics.create_directories(model_save_dir, *sub_dirs)

    model_name_h5 = os.path.join(sub_dirs[0], f'model_FDM-{wavelet}-{experiment}-(HDF5)).h5')
    model_name_keras = os.path.join(sub_dirs[0], f'model_FDM-{wavelet}-{experiment}-(keras)).keras')

    model.save(model_name_h5)
    model.save(model_name_keras)
    print('Saved trained model successfully')

#Parametros para iniciales de imagenes
image_size = (300, 300)
n_imagenes = 100
#Inicializacion de las listas de datos
data = []
labels = []
labelsDWT =[]
DATASET_TYPE = 'FMD'
#dataDWT_cA3, dataDWT_cH3, dataDWT_cV3, dataDWT_cD3 = ([] for _ in range(4))
# Lista de categorías
dataset_dir = os.path.join("FMD")
categorias = os.listdir(dataset_dir)
print(f'Categorías: {categorias}')

#wavelets_prueba = ['haar', 'db10', 'sym2', 'coif1', 'bior1.1']
wavelets =  ['haar', 'db10', 'sym2', 'coif1', 'bior1.1']
niveles3 = 3

# Cargar imágenes y asignar etiquetas
start_time = time.time()
for i, categoria in enumerate(categorias):
    categoria_path = os.path.join(dataset_dir, categoria)
    for imagen_file in os.listdir(categoria_path)[:n_imagenes]:
        imagen_path = os.path.join(categoria_path, imagen_file)

        # Verificar si el archivo es una imagen
        if os.path.splitext(imagen_file)[1].lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
            # Cargar y preprocesar la imagen con Keras (a escala de grises)
            img = image.load_img(imagen_path, target_size=image_size, color_mode="grayscale")
            img_array = image.img_to_array(img)
            # Agregar datos y etiquetas
            data.append(img_array)
            labels.append(i)
        else:
            None#print(f"Archivo no válido: {imagen_file}")
print(f'Datos cargados con exito con {len(data)} datos y {len(labels)} etiquetas')
# Convertir a matrices NumPy
data = np.array(data, dtype=np.float64)
labels = np.array(labels)

# Información sobre la forma de los conjuntos de datos
#for i, categoria in enumerate(categorias):
#    print(f'Data {categoria.capitalize()}:', data[labels == i].shape)
for categoria in categorias:
    num_muestras = sum(1 for label in labels if label == categorias.index(categoria))
    print(f'Data {categoria.capitalize()}: ({num_muestras}, {data[labels == categorias.index(categoria)].shape[1:]})')
print(f"Dimensiones de datos: {data.shape}")
print(f"Dimensiones de etiquetas: {labels.shape}")

end_time = time.time()
elapsed_time_up = end_time - start_time
print(f"Tiempo transcurrido: {elapsed_time_up} segundos")

# Codificar las etiquetas a one-hot encoding
labels_or = to_categorical(labels, num_classes=10)
print(labels_or.shape)

# Aumentar etiquetas en 4
labelsDWT = np.repeat(labels, 4)
print(f'Número de nuevas etiquetas: {len(labelsDWT)}, {labelsDWT.shape}')

#---------------Extraccion de caracteristicas Wavelet---------------#
dataDWT_wavelets = {wavelet: [] for wavelet in wavelets}
labelsDWT_wavelets = []
elapsed_times_wavelet = {}


for wavelet in wavelets:
    start_time_wavelet = time.time()
    dataDWT_cA3, dataDWT_cH3, dataDWT_cV3, dataDWT_cD3 = ([] for _ in range(4))

    for imagen_array in data:
        if len(imagen_array.shape) == 3:
            imagen_array = imagen_array[:, :, 0]

        coeffs = pywt.wavedec2(imagen_array, wavelet, level=niveles3)
        (cA, (cH1, cV1, cD1), (cH2, cV2, cD2), (cH3, cV3, cD3)) = coeffs

        dataDWT_cA3.append(cA)
        dataDWT_cH3.append(cH1)
        dataDWT_cV3.append(cV1)
        dataDWT_cD3.append(cD1)

    dataDWT_cA3, dataDWT_cH3, dataDWT_cV3, dataDWT_cD3 = map(np.array, (dataDWT_cA3, dataDWT_cH3, dataDWT_cV3, dataDWT_cD3))
    dataDWT3 = np.concatenate((dataDWT_cA3, dataDWT_cH3, dataDWT_cV3, dataDWT_cD3), axis=0)
    dataDWT_wavelets[wavelet] = dataDWT3

    end_time_wavelet = time.time()
    elapsed_time_wavelet = end_time_wavelet - start_time_wavelet
    elapsed_times_wavelet[wavelet] = elapsed_time_wavelet

    print(f'Descomposición de 3 nivel completada con éxito para {wavelet}...')

    print(f'Tiempo transcurrido: {end_time_wavelet[wavelet]} segundos')

# Imprimir las dimensiones de cada matriz
print("Dimensiones de dataDWT_cA3:", dataDWT3.shape)
#---------------------------------------------------------#

# Imprimir información sobre los datos extraídos para cada wavelet
for wavelet, dataDWT3 in dataDWT_wavelets.items():
    print(f'Wavelet: {wavelet}')
    print(f'Número de elementos: {dataDWT3.shape[0]}')
    print(f'Tamaño de cada elemento: {dataDWT3.shape[1:]}')
    print()


# Codificar las etiquetas a one-hot encoding
labels_DWT_oh = to_categorical(labelsDWT, num_classes=10)
print(labels_DWT_oh.shape)
print(labels_DWT_oh[800])

for wavelet in wavelets:
    data = dataDWT_wavelets[wavelet]
    print(f'Statistics for wavelet: {wavelet}')
    print(f'Mean: {np.mean(data)}, Std: {np.std(data)}, Min: {np.min(data)}, Max: {np.max(data)}')
    print(f'Shape: {data.shape}')
    print()

# Diccionarios para almacenar los datos divididos y normalizados
X_train_wavelets = {}
X_test_wavelets = {}
y_train_wavelets = {}
y_test_wavelets = {}

# Normalización y preparación de datos para entrenamiento y evaluación
for wavelet, dataDWT in dataDWT_wavelets.items():
    print(f'Preparando datos para la wavelet: {wavelet}')

    labels_DWT_oh = to_categorical(labelsDWT, num_classes=10)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train3, X_test3, y_train3, y_test3 = train_test_split(dataDWT, labels_DWT_oh, test_size=0.20, random_state=SEED)

    print(f'Dimensión de X_train: {X_train3.shape}')
    print(f'Dimensión de X_test: {X_test3.shape}')
    print(f'Dimensión de y_train: {y_train3.shape}')
    print(f'Dimensión de y_test: {y_test3.shape}')

    # Normalizar los datos
    X_train_normalized = normalize_data(X_train3)
    X_test_normalized = normalize_data(X_test3)

    # Añadir dimensión del canal
    X_train_normalized = np.expand_dims(X_train_normalized, axis=-1)
    X_test_normalized = np.expand_dims(X_test_normalized, axis=-1)

    print(f'Dimensión de X_train_normalized: {X_train_normalized.shape}')
    print(f'Dimensión de X_test_normalized: {X_test_normalized.shape}')
    print()

    # Almacenar los datos normalizados en los diccionarios
    X_train_wavelets[wavelet] = X_train_normalized
    X_test_wavelets[wavelet] = X_test_normalized
    y_train_wavelets[wavelet] = y_train3
    y_test_wavelets[wavelet] = y_test3


# Verificar el almacenamiento de los datos
for wavelet in wavelets:
    print(f'Datos almacenados para la wavelet: {wavelet}')
    print(f'Dimensión de X_train_wavelets[{wavelet}]: {X_train_wavelets[wavelet].shape}')
    print(f'Dimensión de X_test_wavelets[{wavelet}]: {X_test_wavelets[wavelet].shape}')
    print(f'Dimensión de y_train_wavelets[{wavelet}]: {y_train_wavelets[wavelet].shape}')
    print(f'Dimensión de y_test_wavelets[{wavelet}]: {y_test_wavelets[wavelet].shape}')
    print()

INIT_LR = 1e-4  # Valor inicial de learning rate.
epochs = 200  # Cantidad de iteraciones completas al conjunto de imágenes de entrenamiento
batch_size = 32  # Cantidad de imágenes que se toman a la vez en memoria
# img_width, img_height = 300, 300
# img_width1, img_height1 = 150, 150
# img_width2, img_height2 = 75, 75
# img_width3, img_height3 = 38, 38
img_shape = X_train_normalized[0].shape
n_class=10
experimento = 29
dropout = 0.2
loss_funtion = 'categorical_crossentropy'
metric_acc = 'accuracy'

def cnn_model(input_shape, num_classes):
    #-------------Arquitectura del modelo CNN---------------------#
    model = Sequential()
    model.add(Input(shape=input_shape))
    #model.add(Rescaling(1./255, name='Rescaling'))
    model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', name='Conv2D_1'))
    #model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu', name='Conv2D_2'))
    #model.add(RandomBrightness(factor=0.2, name='Bringhtness'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='MaxPool_1'))
    model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', name='Conv2D_3'))
    #model.add(RandomTranslation(height_factor=(-0.2, 0.3), width_factor=(-0.2, 0.3),fill_mode="reflect", name='RandomTranslation'))
    model.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu', name='Conv2D_4'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='MaxPool_2'))
    model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='Conv2D_5'))
    #model.add(RandomFlip("horizontal_and_vertical", seed=None, name='RandomFlip'))
    model.add(Conv2D(256, (3, 3), strides=1, padding='same', activation='relu', name='Conv2D_6'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='MaxPool_3'))
    #model.add(Conv2D(512, (3,3), strides=1, padding='same', activation='relu', name='Conv2D_7'))
    #model.add(Conv2D(512, (3,3), strides=1, padding='same', activation='relu', name='Conv2D_8'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='MaxPool_3'))
    model.add(GlobalAveragePooling2D(name='GlobalAvgPool'))
    model.add(Flatten(name='Flatten'))
    model.add(Dropout(0.3, name='Dropout'))
    model.add(Dense(num_classes, activation='softmax', name='Output'))

    # Compilación del modelo
    adam_optimizer = Adam(learning_rate=INIT_LR, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam_optimizer, loss=loss_funtion, metrics=[metric_acc])

    return model


# Diccionario para guardar las historias de entrenamiento y tiempos
training_histories = {}
training_times = {}
# Diccionario para almacenar los resultados de evaluación
evaluation_results = {}

for wavelet in wavelets:
    print(f'Entrenando con la wavelet: {wavelet}')

    # Limpiar la sesión previa para liberar memoria
    K.clear_session()

    # Obtener los datos para la wavelet actual
    X_train = X_train_wavelets[wavelet]
    X_test = X_test_wavelets[wavelet]
    y_train = y_train_wavelets[wavelet]
    y_test = y_test_wavelets[wavelet]

    # Crear el modelo
    input_shape = X_train.shape[1:]
    model = cnn_model(input_shape, n_class)

    # Mostrar el resumen del modelo
    model.summary()

    # Definir callbacks
    checkpoints = get_checkpoints(wavelet, experimento, DATASET_TYPE)
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='max', restore_best_weights=True)
    csv_logger = get_csv_logger(wavelet, experimento, DATASET_TYPE)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.001)

    start_time = time.time()
    # Entrenar el modelo
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True, callbacks=[csv_logger, reduce_lr, checkpoints])
    end_time = time.time()
    elapsed_time = end_time - start_time
    training_times[wavelet] = elapsed_time


    print(f'Tiempo de entrenamiento para {wavelet}: {elapsed_time:.2f} segundos')

    # Guardar las métricas de entrenamiento
    training_histories[wavelet] = history.history

        # Guardar el modelo entrenado
    export_model(wavelet, experimento, model)

# Diccionario para almacenar los resultados de evaluación
evaluation_results = {}

for wavelet in wavelets:
    print(f'Evaluando el modelo para la wavelet: {wavelet}')

    # Obtener los datos para la wavelet actual
    X_test = X_test_wavelets[wavelet]
    y_test = y_test_wavelets[wavelet]

    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Precisión del modelo en los datos de prueba para {wavelet}: {accuracy*100:.2f}%')
    print(f'Función de pérdida en los datos de prueba para {wavelet}: {loss:.4f}')

    # Predicciones del conjunto de prueba
    predictions = model.predict(X_test)
    y_pred_classes = np.argmax(predictions, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # Errors are difference between predicted labels and true labels
    errors = (y_pred_classes != y_test_classes)
    Y_pred_classes_errors = y_pred_classes[errors]
    Y_pred_errors = predictions[errors]
    Y_true_errors = y_test_classes[errors]
    X_test_errors = X_test[errors]

    # Guardar resultados de evaluación en el diccionario
    evaluation_results[wavelet] = {
        'loss': loss,
        'accuracy': accuracy,
        'predictions': predictions,
        'y_pred_classes': y_pred_classes,
        'y_test_classes': y_test_classes,
        'errors': errors,
        'Y_pred_classes_errors': Y_pred_classes_errors,
        'Y_pred_errors': Y_pred_errors,
        'Y_true_errors': Y_true_errors,
        'X_test_errors': X_test_errors
    }

# Diccionario para almacenar todos los parámetros
training_info = {}

for wavelet in wavelets:
    training_info[wavelet] = {
        'history': training_histories[wavelet],
        'training_time': training_times[wavelet],
        'img_shape_DWT': img_shape_DWT
    }

# información
#for wavelet, info in training_info.items():
#    print(f'Wavelet: {wavelet}')
#    print(f'Tiempo de entrenamiento: {info["training_time"]:.2f} segundos')
#    print(f'Historia de entrenamiento: {info["history"].keys()}')
#    print()

# Exportar métricas y graficar para cada wavelet
for wavelet, eval_data in evaluation_results.items():

    num_train_samples = X_train.shape[0]
    num_test_samples = X_test.shape[0]
    num_validation_samples = int(0.2 * num_train_samples)

    params_dict = {
        "wavelet": wavelet,
        "image_size": image_size,
        "image_size_DWT": img_shape,
        "n_imagenes": n_imagenes,
        "nivel_DWT": niveles3,
        "INIT_LR": INIT_LR,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_class": n_class,
        "experimento": experimento,
        "Dropout": 0.3,
        "time-up": elapsed_time_up,
        "time-DWT": elapsed_times_wavelet[wavelet],
        "time-train": training_info[wavelet]['training_time'],
        "SEED": SEED
    }

    loss = eval_data['loss']
    accuracy = eval_data['accuracy']
    y_test_classes = eval_data['y_test_classes']
    y_pred_classes = eval_data['y_pred_classes']
    X_test_errors = eval_data['X_test_errors']

    export_metrics.save_parameters(params_dict, wavelet, experimento, DATASET_TYPE)
    export_metrics.save_results(loss, accuracy, y_test_classes, y_pred_classes, wavelet, experimento, DATASET_TYPE)
    export_metrics.plot_performance(training_info[wavelet]['history'], wavelet, experimento, DATASET_TYPE)
    export_metrics.check_prediction(X_test_errors, y_test_classes, y_pred_classes, wavelet, experimento, DATASET_TYPE)
    export_metrics.predic_graph(X_test_errors, y_pred_classes, y_test_classes, wavelet, experimento, DATASET_TYPE)
    export_metrics.clss_report(y_test_classes, y_pred_classes, wavelet, experimento, DATASET_TYPE)

    export_metrics.save_experiment_parameters_to_csv(params_dict, wavelet, experimento, DATASET_TYPE)
    export_metrics.save_experiment_results_to_csv(loss, accuracy, y_test_classes, y_pred_classes, wavelet, experimento, DATASET_TYPE)

    # Calcular métricas de evaluación adicionales
    accuracy = export_metrics.calculate_accuracy(y_test_classes, y_pred_classes)
    precision = export_metrics.calculate_precision(y_test_classes, y_pred_classes)
    conf_matrix = export_metrics.calculate_confusion_matrix(y_test_classes, y_pred_classes, 10)
    print(f'Exactitud: {accuracy * 100:.2f}%')
    print(f'Precisión: {precision:.2f}')
    print('Matriz de Confusión:')
    print(conf_matrix)

export_metrics.plot_performance_for_experiment(training_histories, experimento)

