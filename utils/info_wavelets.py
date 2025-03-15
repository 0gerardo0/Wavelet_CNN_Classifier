import pywt
import csv
import numpy as np
import matplotlib.pyplot as plt

def get_wavelet_info(name):
    """Obtiene la información de una wavelet específica por su nombre."""
    wavelet = pywt.Wavelet(name)
    scaling_filter = wavelet.dec_lo
    wavelet_filter = wavelet.dec_hi
    info = {
        'Wavelet': wavelet.name,
        'Family name': wavelet.family_name,
        'Short name': wavelet.short_family_name,
        'Filters length': len(scaling_filter),
        'Orthogonal': wavelet.orthogonal,
        'Biorthogonal': wavelet.biorthogonal,
        'Symmetry': wavelet.symmetry,
        'DWT': wavelet.orthogonal and wavelet.biorthogonal,
        'CWT': False  # No todos los wavelets soportan Continuous Wavelet Transform (CWT)
    }
    return info, scaling_filter, wavelet_filter

def print_and_save_wavelet_info(name, filename):
    """Imprime y guarda la información de una wavelet en un archivo CSV."""
    info, scaling_filter, wavelet_filter = get_wavelet_info(name)
    print(f"Wavelet {name}")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("Filtro de escalamiento (phi):", scaling_filter)
    print("Filtro de ondaleta (psi):", wavelet_filter)
'''
    # Guardar la información en un archivo CSV
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = list(info.keys())
        fieldnames += ['Scaling filter', 'Wavelet filter']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Escribir encabezados si el archivo está vacío
        if csvfile.tell() == 0:
            writer.writeheader()

        info['Scaling filter'] = scaling_filter
        info['Wavelet filter'] = wavelet_filter
        writer.writerow(info)
'''
'''
def plot_wavelet(wavelet_name, filename, num_points=1000):
    # Definir el rango de valores donde se graficará
    x = np.linspace(0, 1, num_points)

    # Calcular la función de ondaleta
    wavelet_info = pywt.Wavelet(wavelet_name).wavefun()

    # Obtener la función de ondaleta
    wavelet_function = wavelet_info[0]
    wavelet_length = len(wavelet_function)

    # Redimensionar el rango de valores x
    x = np.linspace(0, 1, wavelet_length)

    # Graficar la función de ondaleta
    plt.plot(x, wavelet_function)
    plt.title('Función de ondaleta: {}'.format(wavelet_name))
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.grid(True)

    # Guardar la imagen
    plt.savefig(filename)

    # Mostrar la gráfica
    plt.show()
'''

def plot_wavelet(wavelet_name, filename, num_points=1000):
    # Calcular la función de ondaleta
    wavelet_info = pywt.Wavelet(wavelet_name).wavefun()


    # Obtener la función de ondaleta
    wavelet_function = wavelet_info[0]
    wavelet_length = len(wavelet_function)

    # Definir el rango de valores donde se graficará
    x = np.linspace(0, wavelet_length - 1, wavelet_length)

    # Graficar la función de ondaleta
    plt.plot(x, wavelet_function)
    plt.title('Función de ondaleta: {}'.format(wavelet_name))
    plt.xlabel('Índice')
    plt.ylabel('Amplitud')
    plt.grid(True)

    # Guardar la imagen
    plt.savefig(filename)

    # Mostrar la gráfica
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    wavelet_name = input("dame el nombre de la wavelet:") 
    csv_filename = f'{wavelet_name}_info.png'
    
    print_and_save_wavelet_info(wavelet_name, csv_filename)
    plot_wavelet(wavelet_name, csv_filename)
