import os 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input
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
    precision = calculate_precision(y_true, y_pred)
    conf_matrix = calculate_confusion_matrix(y_true, y_pred)

    base_dir = 'metrics'
    metrics_result_dir = os.path.join(base_dir, f'{wavelet}-experiment')
    subdirectories = ['conf_matrix', 'csv_results']
    sub_dirs = [os.path.join(metrics_result_dir, sub) for sub in subdirectories]
    create_directories(metrics_result_dir, *sub_dirs)

    # Guardar resultados en un archivo CSV
    results_dict = {'Loss': loss, 'Accuracy': accuracy, 'Precision': precision, 'Confusion Matrix': conf_matrix}
    csv_result_export = os.path.join(sub_dirs[1], f"model_FDM-{wavelet}-{experiment}-metrics_result.csv")
    results_df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['Value'])

    if save_metrics:
        results_df.to_csv(csv_result_export)

    # Graficar la matriz de confusión
    conf_matrix_export = os.path.join(sub_dirs[0], f"model_FDM-{wavelet}-{experiment}-conf_matrix.png")
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    if save_metrics:
        plt.savefig(conf_matrix_export)
    plt.close()


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
    
    base_dir = 'metrics'
    metrics_result_dir = os.path.join(base_dir, f'{wavelet}-experiment')
    subdirectories = ['graph_history']
    sub_dirs = [os.path.join(metrics_result_dir, sub) for sub in subdirectories]
    create_directories(metrics_result_dir, *sub_dirs)

    if 'accuracy' in history.history and 'val_accuracy' in history.history and 'loss' in history.history and 'val_loss' in history.history:
        # Graficar la precisión y la pérdida
        graph_metrics_export = os.path.join(sub_dirs[0], f"model_FDM-{wavelet}-{experiment}-conf_matrix.png")
        plt.plot(history.history['accuracy'], label='Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.plot(history.history['loss'], label='Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title(fig_title)
        plt.legend()

        # Guardar la figura si se especifica
        if save_fig:
            plt.savefig(graph_metrics_export)
        #plt.show()
        plt.close()
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

wavelet = 'haar'
experimento = 1
# Genera datos de ejemplo
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=100)
X_test = np.random.rand(20, 10)
y_test = np.random.randint(0, 2, size=20)

# Define y entrena el modelo
model = Sequential()
model.add(Input(shape=(10,)))  # Capa de entrada con forma (10,)
model.add(Dense(1, activation='sigmoid'))  # Capa oculta
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)
predictions = model.predict(X_test)
# Umbralizar las predicciones
threshold = 0.5
predicted_classes = (predictions > threshold).astype("int32")

# Evaluar el modelo y guardar los resultados
loss, accuracy = model.evaluate(X_test, y_test)

save_results(loss, accuracy, y_test, predicted_classes, wavelet, experimento)
# Opcionalmente, grafica el rendimiento del modelo a lo largo del entrenamiento
plot_performance(history, wavelet, experimento)