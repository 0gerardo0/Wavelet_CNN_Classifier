import os, csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

FMD_LABELS = ['piedra', 'agua', 'metal', 'papel', 'hojas', 'vidrio', 'cuero', 'plástico', 'madera', 'tela']
KHT_LABELS = ['pan_blanco', 'madera', 'hoja_de_lechuga', 'corcho', 'lana', 'papel_de_aluminio', 'galleta', 'pan_integral', 'algodón', 'lino', 'pana']

def get_dataset_info(dataset_type):
    """
    Define y devuelve las etiquetas y el número de clases para un tipo de dataset dado.

    Args:
        dataset_type (str): Tipo de dataset a utilizar ('FMD' o 'KHT').

    Returns:
        tuple: (list de etiquetas, número de clases).
    """
    if dataset_type == 'FMD':
        return FMD_LABELS, len(FMD_LABELS)
    elif dataset_type == 'KHT':
        return KHT_LABELS, len(KHT_LABELS)
    else:
        print(f"Advertencia: Tipo de dataset '{dataset_type}' no reconocido. Usando etiquetas genéricas y 10 clases.")
        return [f"clase_{i}" for i in range(10)], 10

def create_directories(*dirs):
    """Crea directorios si no existen."""
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Crea un mapa de calor para una matriz de confusión.
    """
    if not ax:
        ax = plt.gca()
    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels, rotation=45, ha='left') # Rotación para etiquetas largas
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Etiqueta estimada')
    ax.set_ylabel('Etiqueta real')
    return im, cbar

def annotate_heatmap(im, data=None, fmt="d", threshold=None, **textkw):
    """
    Anota el mapa de calor con los valores de la matriz.
    """
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            current_value = data[i, j]
            kw.update(color="white" if im.norm(current_value) > threshold else "black")
            text = im.axes.text(j, i, format(current_value, fmt), **kw)
            texts.append(text)
    return texts

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred, average='weighted', zero_division=1):
    return precision_score(y_true, y_pred, average=average, zero_division=1)

def calculate_recall(y_true, y_pred, average='weighted'):
    return recall_score(y_true, y_pred, average=average)

def calculate_f1_score(y_true, y_pred, average='weighted'):
    return f1_score(y_true, y_pred, average=average)

def calculate_confusion_matrix(y_true, y_pred, num_classes):
    return confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

def save_results(loss, accuracy, y_true, y_pred, wavelet, experiment, dataset_type, save_metrics=True):
    """
    Calcula y guarda métricas de rendimiento y la matriz de confusión.
    """
    current_labels, current_num_classes = get_dataset_info(dataset_type)

    cal_accuracy = calculate_accuracy(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred)
    conf_matrix = calculate_confusion_matrix(y_true, y_pred, current_num_classes)
    cal_recall = calculate_recall(y_true, y_pred)
    cal_f1 = calculate_f1_score(y_true, y_pred)

    base_dir = 'saves/metrics'
    metrics_result_dir = os.path.join(base_dir, f'{wavelet}-experiment')
    subdirectories = ['conf_matrix', 'csv_results']
    sub_dirs = [os.path.join(metrics_result_dir, sub) for sub in subdirectories]
    create_directories(metrics_result_dir, *sub_dirs)

    results_dict = {'Loss': loss,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Cal-Accuracy': cal_accuracy,
                    'Cal_Recall': cal_recall,
                    'Cal_f1_score': cal_f1}
    csv_result_export = os.path.join(sub_dirs[1], f"model_{dataset_type}-{wavelet}-{experiment}-metrics_result.csv")

    if save_metrics:
        results_df = pd.DataFrame([results_dict])
        results_df.to_csv(csv_result_export, index=False)

    thresh = conf_matrix.max() / 2.
    conf_matrix_export = os.path.join(sub_dirs[0], f"model_{dataset_type}-{wavelet}-{experiment}-conf_matrix.pdf")
    fig, ax = plt.subplots(figsize=(10, 8))
    im, cbar = heatmap(conf_matrix, current_labels, current_labels, ax=ax, cmap=plt.cm.Reds, cbarlabel="Predicciones")
    texts = annotate_heatmap(im, data=conf_matrix, fmt=".0f", threshold=thresh)
    fig.tight_layout()
    if save_metrics:
        plt.savefig(conf_matrix_export, format='pdf')
    plt.close()
    print("Results saved successfully.")

def plot_performance(history, wavelet, experiment, dataset_type , save_fig=True, fig_title='Rendimiento del Modelo'):
    """
    Grafica el rendimiento de entrenamiento y validación de un modelo.
    """
    base_dir = 'saves/metrics'
    metrics_result_dir = os.path.join(base_dir, f'{wavelet}-experiment')
    subdirectories = ['graph_history']
    sub_dirs = [os.path.join(metrics_result_dir, sub) for sub in subdirectories]
    create_directories(metrics_result_dir, *sub_dirs)

    if hasattr(history, 'history'):
        history_data = history.history
    else:
        history_data = history

    if 'accuracy' in history_data and 'val_accuracy' in history_data and \
       'loss' in history_data and 'val_loss' in history_data:

        # Gráfico 1: Todas las métricas juntas
        graph_metrics_export = os.path.join(sub_dirs[0], f"modelo_{dataset_type}-{wavelet}-{experiment}-grafica_val.pdf")
        plt.figure(figsize=(10, 6))
        plt.plot(history_data['accuracy'], '-c', label='Exactitud de entrenamiento')
        plt.plot(history_data['val_accuracy'], '-b', label='Exactitud de validación')
        plt.plot(history_data['loss'], '--y', label='Pérdida de entrenamiento')
        plt.plot(history_data['val_loss'], '--g', label='Pérdida de validación')
        plt.xlabel('Época')
        plt.ylabel('Valor')
        plt.title('Métricas de entrenamiento y validación')
        plt.legend(loc='best')
        if save_fig:
            plt.savefig(graph_metrics_export)
        plt.close()

        # Gráfico 2: Exactitud y Pérdida de Validación
        graph_metrics_export_2 = os.path.join(sub_dirs[0], f"modelo_{dataset_type}-{wavelet}-{experiment}-val-acc_loss.pdf")
        plt.figure(figsize=(10, 6))
        plt.plot(history_data['val_accuracy'], '--y', label='Exactitud de validación')
        plt.plot(history_data['val_loss'], '--g', label='Pérdida de validación')
        plt.xlabel('Época')
        plt.ylabel('Valor')
        plt.title('Exactitud y Pérdida de Validación')
        plt.legend(loc='best')
        if save_fig:
            plt.savefig(graph_metrics_export_2)
        plt.close()

        # Gráfico 3: Exactitud y Pérdida de Entrenamiento
        graph_metrics_export_3 = os.path.join(sub_dirs[0], f"modelo_{dataset_type}-{wavelet}-{experiment}-acc_loss.pdf")
        plt.figure(figsize=(10, 6))
        plt.plot(history_data['accuracy'], label='Exactitud de entrenamiento')
        plt.plot(history_data['loss'], '--y', label='Pérdida de entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Valor')
        plt.title('Exactitud y Pérdida de Entrenamiento')
        plt.legend(loc='best')
        if save_fig:
            plt.savefig(graph_metrics_export_3)
        plt.close()

        # Gráfico 4: Subplots de Exactitud y Pérdida
        plt.figure(figsize=(10, 8))
        graph_metrics_export_4 = os.path.join(sub_dirs[0], f"modelo_{dataset_type}-{wavelet}-{experiment}-rendimiento.pdf")
        plt.subplot(2, 1, 1)
        plt.plot(history_data['accuracy'], '-b', label='Exactitud de entrenamiento')
        plt.plot(history_data['val_accuracy'], '-c', label='Exactitud de validación')
        plt.title('Exactitud de Entrenamiento y Validación')
        plt.xlabel('Épocas')
        plt.ylabel('Exactitud')
        plt.legend(loc='best')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(history_data['loss'], '--y', label='Pérdida de entrenamiento')
        plt.plot(history_data['val_loss'], '--g', label='Pérdida de validación')
        plt.title('Pérdida de Entrenamiento y Validación')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend(loc='best')
        plt.grid(True)
        plt.subplots_adjust(hspace=0.3)
        if save_fig:
            plt.savefig(graph_metrics_export_4)
        plt.close()
        print("Gráficos guardados exitosamente.")
    else:
        print("El historial de entrenamiento no contiene la información necesaria para graficar el rendimiento.")

def save_parameters(params_dict, wavelet, experiment, dataset_type):
    """
    Guarda los parámetros del modelo en un archivo CSV.
    """
    base_dir = 'saves/metrics'
    params_result_dir = os.path.join(base_dir, f'{wavelet}-experiment')
    os.makedirs(params_result_dir, exist_ok=True)
    params_export = os.path.join(params_result_dir, f"model_{dataset_type}-{wavelet}-{experiment}-parameters.csv")
    with open(params_export, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Parameter", "Value"])
        for key, value in params_dict.items():
            writer.writerow([key, value])
    print("Parameters saved successfully.")

def check_prediction(x_test, y_true, y_pred, wavelet, experiment, dataset_type):
    """
    Guarda imágenes de predicciones correctas e incorrectas.
    """
    current_labels, _ = get_dataset_info(dataset_type)

    R = 3
    C = 3
    base_dir = 'saves/metrics'
    predict_save = os.path.join(base_dir, f'{wavelet}-experiment')
    os.makedirs(predict_save, exist_ok=True)
    subdirectories = ['check_pred']
    sub_dirs = [os.path.join(predict_save, sub) for sub in subdirectories]
    create_directories(predict_save, *sub_dirs)

    predic_metrics_export = os.path.join(sub_dirs[0], f"modelo_{dataset_type}-{wavelet}-{experiment}-pred_correct.pdf")
    fig, ax = plt.subplots(R, C, figsize=(12,12))
    axes = ax.ravel()

    correct_indices = np.where(y_pred == y_true)[0]
    num_to_plot = min(R * C, len(correct_indices))

    for i in np.arange(0, num_to_plot):
        axes[i].imshow(x_test[correct_indices[i]], cmap='gray')
        axes[i].set_title("True: %s \nPredict: %s" % (current_labels[y_true[correct_indices[i]]], current_labels[y_pred[correct_indices[i]]]))
        axes[i].axis('off')
    for i in range(num_to_plot, R * C):
        axes[i].axis('off')

    plt.subplots_adjust(wspace=1)
    plt.savefig(predic_metrics_export)
    plt.close()

    predic_metrics_export_1 = os.path.join(sub_dirs[0], f"modelo_{dataset_type}-{wavelet}-{experiment}-pred_incorrect.pdf")
    fig, ax = plt.subplots(R, C, figsize=(12,12))
    axes = ax.ravel()

    misclassified_idx = np.where(y_pred != y_true)[0]
    num_to_plot = min(R * C, len(misclassified_idx))

    for i in np.arange(0, num_to_plot):
        axes[i].imshow(x_test[misclassified_idx[i]], cmap='gray')
        # Usar current_labels para los títulos
        axes[i].set_title("True: %s \nPredicted: %s" % (current_labels[y_true[misclassified_idx[i]]], current_labels[y_pred[misclassified_idx[i]]]))
        axes[i].axis('off')
    for i in range(num_to_plot, R * C): # Oculta los ejes no utilizados
        axes[i].axis('off')
    plt.subplots_adjust(wspace=1)
    plt.savefig(predic_metrics_export_1)
    plt.close()
    print("Chequeo de predicciones completado.")

def predic_graph(x_test, y_pred, y_true, wavelet, experiment, dataset_type):
    """
    Guarda gráficos de 9 predicciones correctas y 9 incorrectas.
    """
    current_labels, _ = get_dataset_info(dataset_type)

    correct = np.where(y_pred==y_true)[0]
    print("Found %d correct labels" % len(correct))
    incorrect = np.where(y_pred!=y_true)[0]
    print("Found %d incorrect labels" % len(incorrect))

    base_dir = 'saves/metrics'
    predict_save = os.path.join(base_dir, f'{wavelet}-experiment')
    os.makedirs(predict_save, exist_ok=True)
    subdirectories = ['check_pred']
    sub_dirs = [os.path.join(predict_save, sub) for sub in subdirectories]
    create_directories(predict_save, *sub_dirs)

    num_correct_to_plot = min(9, len(correct))
    num_incorrect_to_plot = min(9, len(incorrect))

    if num_correct_to_plot > 0:
        tab_predic_export = os.path.join(sub_dirs[0], f"modelo_{dataset_type}-{wavelet}-{experiment}-tab_correct.pdf")
        fig_correct, axes_correct = plt.subplots(3, 3, figsize=(12,12))
        axes_correct = axes_correct.ravel()
        for i in range(num_correct_to_plot):
            idx = correct[i]
            axes_correct[i].imshow(x_test[idx], cmap='gray', interpolation='none')
            axes_correct[i].set_title("{}, {}".format(current_labels[y_pred[idx]], current_labels[y_true[idx]]))
            axes_correct[i].axis('off')
        for i in range(num_correct_to_plot, 9):
            axes_correct[i].axis('off')
        plt.tight_layout()
        plt.savefig(tab_predic_export)
        plt.close(fig_correct)

    if num_incorrect_to_plot > 0:
        tab_predic_export_1 = os.path.join(sub_dirs[0], f"modelo_{dataset_type}-{wavelet}-{experiment}-tab_incorrect.pdf")
        fig_incorrect, axes_incorrect = plt.subplots(3, 3, figsize=(12,12))
        axes_incorrect = axes_incorrect.ravel()
        for i in range(num_incorrect_to_plot):
            idx = incorrect[i]
            axes_incorrect[i].imshow(x_test[idx], cmap='gray', interpolation='none')
            axes_incorrect[i].set_title("{}, {}".format(current_labels[y_pred[idx]], current_labels[y_true[idx]]))
            axes_incorrect[i].axis('off')
        for i in range(num_incorrect_to_plot, 9):
            axes_incorrect[i].axis('off')
        plt.tight_layout()
        plt.savefig(tab_predic_export_1)
        plt.close(fig_incorrect)
    print("Predicciones gráficas guardadas exitosamente.")

def clss_report (y_true, y_pred, wavelet, experiment, dataset_type):
    """
    Genera y guarda un reporte de clasificación detallado.
    """
    current_labels, current_num_classes = get_dataset_info(dataset_type)

    report = classification_report(y_true, y_pred,
                                   labels=np.arange(current_num_classes),
                                   target_names=current_labels,
                                   output_dict=True,
                                   zero_division=0)
    report_df = pd.DataFrame(report).transpose()

    base_dir = 'saves/metrics'
    report_result_dir = os.path.join(base_dir, f'{wavelet}-experiment')
    subdirectories = ['csv_results']
    sub_dirs = [os.path.join(report_result_dir, sub) for sub in subdirectories]
    create_directories(report_result_dir, *sub_dirs)

    report_csv_export = os.path.join(sub_dirs[0], f"model_{dataset_type}-{wavelet}-{experiment}-classification_report.csv")
    report_df.to_csv(report_csv_export)

    print("Classification report saved successfully.")

def plot_performance_for_experiment(histories_dict, experiment, save_fig=True, fig_title='Rendimiento del Modelo'):
    """
    Grafica el rendimiento de varios wavelets en un solo experimento para comparación.
    """
    base_dir = 'saves/metrics'
    metrics_result_dir = os.path.join(base_dir, f'experiment_{experiment}')
    subdirectories = ['graph_history_summary']
    sub_dirs = [os.path.join(metrics_result_dir, sub) for sub in subdirectories]
    create_directories(metrics_result_dir, *sub_dirs)

    wavelet_colors = {
        'haar': 'blue',
        'db10': 'green',
        'sym2': 'red',
        'coif1': 'purple',
        'bior1.1': 'orange',
        'prueba': 'cyan'
    }

    if not histories_dict:
        print("No se proporcionaron historiales para graficar el rendimiento del experimento.")
        return

    has_valid_history = False
    for wavelet, history_data in histories_dict.items():
        if hasattr(history_data, 'history'):
            current_history = history_data.history
        else:
            current_history = history_data

        if 'accuracy' in current_history and 'val_accuracy' in current_history and \
           'loss' in current_history and 'val_loss' in current_history:
            has_valid_history = True
            break

    if not has_valid_history:
        print("Ningún historial de entrenamiento contiene la información necesaria para graficar el rendimiento.")
        return

    fig_acc, ax_acc = plt.subplots(figsize=(12, 8))
    fig_val_acc, ax_val_acc = plt.subplots(figsize=(12, 8))
    fig_loss, ax_loss = plt.subplots(figsize=(12, 8))
    fig_val_loss, ax_val_loss = plt.subplots(figsize=(12, 8))
    fig_combined_acc, ax_combined_acc = plt.subplots(figsize=(12, 8))
    fig_combined_loss, ax_combined_loss = plt.subplots(figsize=(12, 8))

    for wavelet, history_data in histories_dict.items():
        if hasattr(history_data, 'history'):
            current_history = history_data.history
        else:
            current_history = history_data

        if 'accuracy' in current_history and 'val_accuracy' in current_history and \
           'loss' in current_history and 'val_loss' in current_history:

            color = wavelet_colors.get(wavelet, 'black')
            ax_acc.plot(current_history['accuracy'], label=f'{wavelet} - Exactitud de Entrenamiento', color=color)
            ax_val_acc.plot(current_history['val_accuracy'], '--', label=f'{wavelet} - Exactitud de Validación', color=color)
            ax_loss.plot(current_history['loss'], label=f'{wavelet} - Pérdida de Entrenamiento', color=color)
            ax_val_loss.plot(current_history['val_loss'], '--', label=f'{wavelet} - Pérdida de Validación', color=color)

            # Para gráficos combinados
            ax_combined_acc.plot(current_history['accuracy'], label=f'{wavelet} - Exactitud de Entrenamiento', color=color)
            ax_combined_acc.plot(current_history['val_accuracy'], '--', label=f'{wavelet} - Exactitud de Validación', color=color)
            ax_combined_loss.plot(current_history['loss'], label=f'{wavelet} - Pérdida de Entrenamiento', color=color)
            ax_combined_loss.plot(current_history['val_loss'], '--', label=f'{wavelet} - Pérdida de Validación', color=color)
        else:
            print(f"Historial incompleto para el wavelet {wavelet}. No se graficará en las comparativas.")

    ax_acc.set_xlabel('Época')
    ax_acc.set_ylabel('Exactitud')
    ax_acc.set_title('Exactitud de Entrenamiento por Wavelet')
    ax_acc.legend(loc='best')
    if save_fig:
        acc_export = os.path.join(sub_dirs[0], f"experimento-{experiment}-exactitud_entrenamiento_comparativa.pdf")
        fig_acc.savefig(acc_export)
    plt.close(fig_acc)

    ax_val_acc.set_xlabel('Época')
    ax_val_acc.set_ylabel('Exactitud de Validación')
    ax_val_acc.set_title('Exactitud de Validación por Wavelet')
    ax_val_acc.legend(loc='best')
    if save_fig:
        val_acc_export = os.path.join(sub_dirs[0], f"experimento-{experiment}-exactitud_validacion_comparativa.pdf")
        fig_val_acc.savefig(val_acc_export)
    plt.close(fig_val_acc)

    ax_loss.set_xlabel('Época')
    ax_loss.set_ylabel('Pérdida')
    ax_loss.set_title('Pérdida de Entrenamiento por Wavelet')
    ax_loss.legend(loc='best')
    if save_fig:
        loss_export = os.path.join(sub_dirs[0], f"experimento-{experiment}-perdida_entrenamiento_comparativa.pdf")
        fig_loss.savefig(loss_export)
    plt.close(fig_loss)

    ax_val_loss.set_xlabel('Época')
    ax_val_loss.set_ylabel('Pérdida de Validación')
    ax_val_loss.set_title('Pérdida de Validación por Wavelet')
    ax_val_loss.legend(loc='best')
    if save_fig:
        val_loss_export = os.path.join(sub_dirs[0], f"experimento-{experiment}-perdida_validacion_comparativa.pdf")
        fig_val_loss.savefig(val_loss_export)
    plt.close(fig_val_loss)

    ax_combined_acc.set_xlabel('Época')
    ax_combined_acc.set_ylabel('Valor')
    ax_combined_acc.set_title('Exactitud de Entrenamiento y Validación por Wavelet')
    ax_combined_acc.legend(loc='best')
    if save_fig:
        combined_acc_export = os.path.join(sub_dirs[0], f"experimento-{experiment}-exactitud_combinada_comparativa.pdf")
        fig_combined_acc.savefig(combined_acc_export)
    plt.close(fig_combined_acc)

    ax_combined_loss.set_xlabel('Época')
    ax_combined_loss.set_ylabel('Valor')
    ax_combined_loss.set_title('Pérdida de Entrenamiento y Validación por Wavelet')
    ax_combined_loss.legend(loc='best')
    if save_fig:
        combined_loss_export = os.path.join(sub_dirs[0], f"experimento-{experiment}-perdida_combinada_comparativa.pdf")
        fig_combined_loss.savefig(combined_loss_export)
    plt.close(fig_combined_loss)

    print("Gráficos de rendimiento por experimento (comparativa de wavelets) guardados exitosamente.")

def save_experiment_results_to_csv(loss, accuracy, y_true, y_pred, wavelet, experiment, dataset_type):
    """
    Guarda los resultados resumen de un experimento particular en un CSV.
    """
    base_dir = 'saves/metrics'
    metrics_result_dir = os.path.join(base_dir, f'experiment_{experiment}', 'csv_results_summary')
    create_directories(metrics_result_dir)

    cal_accuracy = calculate_accuracy(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred)
    cal_recall = calculate_recall(y_true, y_pred)
    cal_f1 = calculate_f1_score(y_true, y_pred)

    results_dict = {'Wavelet': wavelet,
                    'Experiment': experiment,
                    'Dataset_Type': dataset_type,
                    'Loss': loss,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Cal-Accuracy': cal_accuracy,
                    'Cal_Recall': cal_recall,
                    'Cal_f1_score': cal_f1
                   }

    results_df = pd.DataFrame([results_dict])
    csv_file_path = os.path.join(metrics_result_dir, f"model_{dataset_type}-{wavelet}-{experiment}-summary_metrics_result.csv")
    results_df.to_csv(csv_file_path, index=False)
    print(f"Resultados de resumen para Wavelet '{wavelet}', Experimento '{experiment}', Dataset '{dataset_type}' guardados exitosamente en '{csv_file_path}'.")

def save_experiment_parameters_to_csv(params_dict, wavelet, experiment, dataset_type):
    """
    Guarda los parámetros resumen de un experimento particular en un CSV.
    """
    base_dir = 'saves/metrics'
    params_result_dir = os.path.join(base_dir, f'experiment_{experiment}', 'parameters_summary')
    create_directories(params_result_dir)

    params_dict_extended = params_dict.copy()
    params_dict_extended['Wavelet'] = wavelet
    params_dict_extended['Experiment'] = experiment
    params_dict_extended['Dataset_Type'] = dataset_type 

    params_df = pd.DataFrame([params_dict_extended])
    csv_file_path = os.path.join(params_result_dir, f"model_{dataset_type}-{wavelet}-{experiment}-summary_parameters.csv")
    params_df.to_csv(csv_file_path, index=False)
    print(f"Parámetros de resumen para Wavelet '{wavelet}', Experimento '{experiment}', Dataset '{dataset_type}' guardados exitosamente en '{csv_file_path}'.")