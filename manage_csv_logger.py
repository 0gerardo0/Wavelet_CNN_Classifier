from tensorflow.keras.callbacks import CSVLogger
import os

def create_csv_logger(csv_directory, wavelet, experiment):
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)
    csv_logger_path=os.path.join(csv_directory, f"model_FDM-{wavelet}-{experiment}-log.csv")
    return CSVLogger(csv_logger_path, append=True)

def get_csv_logger(wavelet, experiment):
    csv_directory=f'csv_logger/{wavelet}-experiment/'
    csv_logger=create_csv_logger(csv_directory, wavelet, experiment)
    return csv_logger

# prueba
wavelet = 'haar'
experiment = 1
csv_logger = get_csv_logger(wavelet, experiment)