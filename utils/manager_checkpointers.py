from tensorflow.keras.callbacks import ModelCheckpoint
import os

def create_checkpoint(checkpoint_directory, monitor_metric, experiment, dataset_type):
    """
    Crea un objeto ModelCheckpoint para guardar los mejores modelos durante el entrenamiento.

    Args:
        checkpoint_directory (str): Directorio donde se almacenarán los checkpoints.
        monitor_metric (str): Métrica a monitorear para determinar los mejores modelos.
        experiment (int): Número del experimento.

    Returns:
        ModelCheckpoint: Objeto ModelCheckpoint configurado para guardar los mejores modelos.
    """
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)
    checkpoint_path = os.path.join(checkpoint_directory, f"model_{dataset_type}-{monitor_metric}-{experiment}.h5")
    return ModelCheckpoint(checkpoint_path,
                        monitor=monitor_metric,
                        save_best_only=True,
                        mode = 'min' if 'loss' in monitor_metric or 'val_loss' in monitor_metric else 'max',
                        verbose=2)

def get_checkpoints(wavelet, experiment, dataset_type):
    """
    Obtiene una lista de objetos ModelCheckpoint configurados para varios checkpoints.

    Args:
        wavelet (str): Tipo de wavelet utilizado en el experimento.
        experiment (int): Número del experimento.

    Returns:
        list: Lista de objetos ModelCheckpoint configurados para guardar los mejores modelos durante el entrenamiento.
    """
    checkpointers = []
    metrics_to_monitor = ['accuracy', 'loss', 'val_accuracy', 'val_loss']
    for metric in metrics_to_monitor:
        checkpoint_directory = f'saves/checkpointer/{wavelet}-experiment/'
        checkpointers.append(create_checkpoint(checkpoint_directory, metric, experiment, dataset_type))
    print('Checkpointers saved successfully.')
    return checkpointers

'''
# Obtener los checkpoints
wavelet = 'db10'  
experiment = 1 
checkpoints = get_checkpoints(wavelet, experiment)
#Prueba
for checkpoint in checkpoints:
    print(checkpoint.filepath)
'''