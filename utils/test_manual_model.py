import matplotlib.pyplot as plt
import numpy as np
import os

labels = ['piedra', 'agua', 'metal', 'papel', 'hojas', 'vidrio', 'cuero', 'plástico', 'madera', 'tela']

def show_test(number, x_test, y_true, model, wavelet, experimento, classes=labels):
    fig, ax = plt.subplots(figsize=(5, 5))
    test_image = np.expand_dims(x_test[number], axis=0)
    test_result = model.predict(test_image)
    dict_key = test_result[0]
    
    # Obtener la etiqueta predicha y la etiqueta verdadera
    predicted_class = classes[np.argmax(dict_key)]
    true_class = classes[y_true[number]]
    
    # Mostrar la imagen
    if x_test[number].ndim == 3:
        ax.imshow(x_test[number][:, :, 0], cmap='gray')  # Asumiendo que las imágenes son en escala de grises
    else:
        ax.imshow(x_test[number], cmap='gray')
    
    # Agregar título con las etiquetas predichas y verdaderas
    ax.set_title("Predicción: {} \n Etiqueta Real: {}".format(
        predicted_class, 
        true_class
    ))
    
    ax.axis('off')  # Ocultar ejes para una mejor visualización

    # Agregar cuadro de texto con información adicional
    info_text = "Wavelet: {} \nExperimento: {} \nIndice: {}".format(
        wavelet, 
        experimento, 
        number
    )
    plt.gcf().text(0.5, 0.01, info_text, ha='center', va='bottom', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.5))

    # Crear directorio para guardar las imágenes si no existe
    base_dir = 'saves/metrics'
    predict_save = os.path.join(base_dir, f'{wavelet}-experiment')
    os.makedirs(predict_save, exist_ok=True)
    subdirectories = ['test_images']
    sub_dirs = [os.path.join(predict_save, sub) for sub in subdirectories]
    create_directories(predict_save, *sub_dirs)
    
    # Guardar la imagen en un archivo PDF
    test_image_export = os.path.join(sub_dirs[0], f"test_image_{number}_exp_{experimento}.pdf")
    plt.savefig(test_image_export)
    plt.close()
    print(f"Test image {number} saved successfully.")

def create_directories(*dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)