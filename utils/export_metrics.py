import os, csv 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from keras.models import Sequential
#from keras.layers import Dense, Input
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def create_directories(*dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

def save_results(loss, accuracy, y_true, y_pred, wavelet, experiment, save_metrics=True):
    """
    Guarda los resultados de la evaluación del modelo en un archivo CSV y
    grafica la matriz de confusión.

    Args:
        loss (float): Valor de la pérdida del modelo.
        accuracy (float): Valor de la precisión del modelo.
        y_true (array): Etiquetas verdaderas.
        y_pred (array): Etiquetas predichas.
        wavelet (str): Tipo de wavelet utilizado en el experimento.
        experiment (int): Número del experimento.
        save_metrics (bool, optional): Indica si se deben guardar las métricas en un archivo CSV. Por defecto es True.
    """
    cal_accuracy = calculate_accuracy(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred)
    conf_matrix = calculate_confusion_matrix(y_true, y_pred)
    cal_recall = calculate_recall(y_true, y_pred)
    cal_f1 = calculate_f1_score(y_true, y_pred)

    base_dir = 'saves/metrics'
    metrics_result_dir = os.path.join(base_dir, f'{wavelet}-experiment')
    subdirectories = ['conf_matrix', 'csv_results']
    sub_dirs = [os.path.join(metrics_result_dir, sub) for sub in subdirectories]
    create_directories(metrics_result_dir, *sub_dirs)

    # Guardar resultados en un archivo CSV
    results_dict = {'Loss': loss,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Cal-Accuracy':cal_accuracy, 
                    'Cal_Recall':cal_recall, 
                    'Cal_f1_score':cal_f1, 
                    'Confusion Matrix': conf_matrix}
    csv_result_export = os.path.join(sub_dirs[1], f"model_FDM-{wavelet}-{experiment}-metrics_result.csv")
    results_df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Value'])
    if save_metrics:
        with open(csv_result_export, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Metric", "Value"])
            for key, value in results_dict.items():
                writer.writerow([key, value])
    
        
    
    # Graficar la matriz de confusión
    conf_matrix_export = os.path.join(sub_dirs[0], f"model_FDM-{wavelet}-{experiment}-conf_matrix.png")
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Reds', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    if save_metrics:
        plt.savefig(conf_matrix_export)
    plt.close()

    print("Results saved successfully.")


def plot_performance(history, wavelet, experiment, save_fig=True, fig_title='Model Performance'):
    """
    Grafica la precisión y la pérdida a lo largo del entrenamiento del modelo.

    Args:
        history (History): Historial de entrenamiento del modelo.
        wavelet (str): Tipo de wavelet utilizado en el experimento.
        experiment (int): Número del experimento.
        save_fig (bool, optional): Indica si se debe guardar la figura como un archivo. Por defecto es True.
        fig_title (str, optional): Título de la gráfica. Por defecto es 'Model Performance'.
    """
    base_dir = 'saves/metrics'
    metrics_result_dir = os.path.join(base_dir, f'{wavelet}-experiment')
    subdirectories = ['graph_history']
    sub_dirs = [os.path.join(metrics_result_dir, sub) for sub in subdirectories]
    create_directories(metrics_result_dir, *sub_dirs)

    
    if 'accuracy' in history.history and 'val_accuracy' in history.history and 'loss' in history.history and 'val_loss' in history.history:
        # Graficar la precisión y la pérdida
        graph_metrics_export = os.path.join(sub_dirs[0], f"model_FDM-{wavelet}-{experiment}-graph_val.png")
        plt.plot(history.history['accuracy'], '-c',  label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], '-b', label='Validation Accuracy')
        plt.plot(history.history['loss'], '--y', label='Training Loss')
        plt.plot(history.history['val_loss'], '--g', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training and Validation Metrics')
        plt.legend(loc='best')
        if save_fig:
            plt.savefig(graph_metrics_export)
        plt.close()

        graph_metrics_export_2 = os.path.join(sub_dirs[0], f"model_FDM-{wavelet}-{experiment}-val-acc_loss.png")
        plt.plot(history.history['val_accuracy'], '--y', label='Validation Accuracy')
        plt.plot(history.history['val_loss'], '--g', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Validation Accuracy and Loss')
        plt.legend(loc='best')
        if save_fig:
            plt.savefig(graph_metrics_export_2)
        plt.close()
        
        graph_metrics_export_3 = os.path.join(sub_dirs[0], f"model_FDM-{wavelet}-{experiment}-acc_loss.png")
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['loss'], '--y', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Accuracy and Loss')
        plt.legend(loc='best')
        if save_fig:
            plt.savefig(graph_metrics_export_3)
        plt.close()

        plt.figure(figsize=(10, 8))
        graph_metrics_export_4 = os.path.join(sub_dirs[0], f"model_FDM-{wavelet}-{experiment}-performance.png")
        plt.subplot(2, 1, 1)
        plt.plot(history.history['accuracy'], '-b', label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], '-c', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'], '--y', label='Training Loss')
        plt.plot(history.history['val_loss'], '--g', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.grid(True)
        plt.subplots_adjust(hspace=0.3)
        if save_fig:
            plt
            plt.savefig(graph_metrics_export_4)
        plt.close()
        print("Plots saved successfully.")
    else:
        print("El historial de entrenamiento no contiene la información necesaria para graficar el rendimiento.")


def calculate_accuracy(y_true, y_pred):
    """Calcula la precisión del modelo."""
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred, average='weighted'):
    """Calcula la precisión del modelo."""
    return precision_score(y_true, y_pred, average=average)

def calculate_recall(y_true, y_pred, average='weighted'):
    """Calcula el recall del modelo."""
    return recall_score(y_true, y_pred, average=average)

def calculate_f1_score(y_true, y_pred, average='weighted'):
    """Calcula el F1-score del modelo."""
    return f1_score(y_true, y_pred, average=average)

def calculate_confusion_matrix(y_true, y_pred):
    """Calcula la matriz de confusión del modelo."""
    return confusion_matrix(y_true, y_pred)


def save_parameters(params_dict, wavelet, experiment):
    """
    Guarda los parámetros utilizados en la función en un archivo CSV.

    Args:
        params_dict (dict): Diccionario que contiene los nombres de los parámetros como claves y sus valores.
        wavelet (str): Tipo de wavelet utilizado en el experimento.
        experiment (int): Número del experimento.
    """
    base_dir = 'saves/metrics'
    params_result_dir = os.path.join(base_dir, f'{wavelet}-experiment')
    os.makedirs(params_result_dir, exist_ok=True)

    params_export = os.path.join(params_result_dir, f"model_FDM-{wavelet}-{experiment}-parameters.csv")
    with open(params_export, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Parameter", "Value"])
        for key, value in params_dict.items():
            writer.writerow([key, value])

    print("Parameters saved successfully.")
'''
#Parametros de prueba
wavelet = 'haar'
experimento = 1
image_size=(300, 300)
n_imagenes=100
INIT_LR = 0.022
batch_size = 32
weight_decay = 0.5
experimento = 1
epochs=100

# Genera datos de ejemplo
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=100)
X_test = np.random.rand(20, 10)
y_test = np.random.randint(0, 2, size=20)

# Define y entrena el modelo
model = Sequential()
model.add(Input(shape=(10,)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Historial
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)
predictions = model.predict(X_test)
threshold = 0.5
predicted_classes = (predictions > threshold).astype("int32")
loss, accuracy = model.evaluate(X_test, y_test)

params_dict = {
    "image_size": image_size,
    "n_imagenes": n_imagenes,
    "wavelet": wavelet,
    "INIT_LR": INIT_LR,
    "epochs": epochs,
    "batch_size": batch_size,
    "weight_decay": weight_decay,
    "experimento": experimento,
}

save_parameters(params_dict, wavelet, experimento)
save_results(loss, accuracy, y_test, predicted_classes, wavelet, experimento)
plot_performance(history, wavelet, experimento)
'''