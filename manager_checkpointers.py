from tensorflow.keras.callbacks import ModelCheckpoint
import os

def create_checkpoint(checkpoint_directory, monitor_metric, experiment):
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)
    checkpoint_path = os.path.join(checkpoint_directory, f"model_FDM-{monitor_metric}-{experiment}.h5")
    return ModelCheckpoint(checkpoint_path,
                        monitor=monitor_metric,
                        save_best_only=True,
                        mode = 'min' if 'loss' in monitor_metric or 'val_loss' in monitor_metric else 'max',
                        verbose=2)

def get_checkpoints(wavelet, experiment):
    checkpointers = []
    metrics_to_monitor = ['val_loss', 'val_acc', 'loss', 'acc']
    for metric in metrics_to_monitor:
        checkpoint_directory = f'checkpointer/{wavelet}-experiment/'
        checkpointers.append(create_checkpoint(checkpoint_directory, metrics_to_monitor, experiment))
    return checkpointers

