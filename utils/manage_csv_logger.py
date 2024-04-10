from tensorflow.keras.callbacks import CSVLogger
import os


def create_csv_logger(csv_directory, wavelet, experiment):
    """
    Crea un objeto CSVLogger para guardar los registros del entrenamiento del modelo en un archivo CSV.

    Args:
        csv_directory (str): Directorio donde se almacenará el archivo CSV.
        wavelet (str): Tipo de wavelet utilizado en el experimento.
        experiment (int): Número del experimento.

    Returns:
        CSVLogger: Objeto CSVLogger configurado para guardar registros en el archivo CSV especificado.
    """
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    csv_logger_path=os.path.join(csv_directory, f"model_FDM-{wavelet}-{experiment}-log.csv")
    return CSVLogger(csv_logger_path, append=True)

def get_csv_logger(wavelet, experiment):
    """
    Obtiene un objeto CSVLogger configurado para un experimento específico.

    Args:
        wavelet (str): Tipo de wavelet utilizado en el experimento.
        experiment (int): Número del experimento.

    Returns:
        CSVLogger: Objeto CSVLogger configurado para guardar registros en el archivo CSV correspondiente al experimento.
    """
    csv_directory=f'saves/csv_logger/{wavelet}-experiment/'
    csv_logger=create_csv_logger(csv_directory, wavelet, experiment)
    print('CSVLogger saved successfully.')
    return csv_logger
'''
# prueba
wavelet = 'haar'
experiment = 1
csv_logger = get_csv_logger(wavelet, experiment)
'''