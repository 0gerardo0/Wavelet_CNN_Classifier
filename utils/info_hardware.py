import csv
import platform
import psutil
import subprocess 
import io

def get_hardware_info():
    """
    Obtiene información sobre la CPU, RAM y GPU (si es NVIDIA y nvidia-smi está disponible).
    """
    hardware_info = {}

    # --- Información de la GPU (NVIDIA) ---
    try:
        # Ejecutar nvidia-smi y capturar la salida en formato CSV
        # 'text=True' decodifica stdout y stderr como texto (UTF-8 por defecto)
        # 'check=True' lanza una excepción CalledProcessError si el comando falla
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name,driver_version,memory.total", "--format=csv"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_output = process.stdout.strip()

        # Usar el módulo csv para parsear la salida de forma robusta
        # io.StringIO permite que csv.reader trate la cadena de texto como un archivo
        reader = csv.reader(io.StringIO(gpu_output))
        
        header = next(reader, None) # Leer la fila de encabezado (ignorar, pero avanza el lector)
        data_row = next(reader, None) # Leer la primera fila de datos de la GPU

        if data_row:
            # Asignar los valores y limpiar espacios en blanco
            gpu_name = data_row[0].strip()
            driver_version = data_row[1].strip()
            memory_total = data_row[2].strip()

            hardware_info['GPU'] = {
                'Name': gpu_name,
                'Driver Version': driver_version,
                'Memory Total': memory_total
            }
        else:
            print("Advertencia: 'nvidia-smi' se ejecutó, pero no devolvió datos de GPU en formato CSV.")
            hardware_info['GPU'] = {'Status': 'No disponible', 'Razon': 'No se encontraron datos de GPU en la salida de nvidia-smi'}

    except FileNotFoundError:
        print("Advertencia: 'nvidia-smi' no encontrado. La información de la GPU NVIDIA no está disponible.")
        hardware_info['GPU'] = {'Status': 'No disponible', 'Razon': 'nvidia-smi no encontrado en el PATH'}
    except subprocess.CalledProcessError as e:
        print(f"Advertencia: Error al ejecutar 'nvidia-smi' (código {e.returncode}): {e.stderr.strip()}")
        hardware_info['GPU'] = {'Status': 'Error', 'Razon': f'Fallo del comando: {e.stderr.strip()}'}
    except Exception as e:
        print(f"Advertencia: No se pudo obtener o parsear la información de la GPU: {e}")
        hardware_info['GPU'] = {'Status': 'Error', 'Razon': 'No se pudo parsear la salida de nvidia-smi'}


    # --- Información de la CPU ---
    cpu_info = platform.processor()
    hardware_info['CPU'] = {'Processor': cpu_info}

    # --- Información de la RAM ---
    # psutil.virtual_memory().total devuelve la memoria total en bytes
    ram_total_gb = psutil.virtual_memory().total / (1024**3) # Convertir a gigabytes
    hardware_info['RAM'] = {'Total (GB)': f"{ram_total_gb:.2f}"} # Formatear a 2 decimales

    return hardware_info

def export_to_csv(hardware_info, filename='hardware_info.csv'):
    """
    Exporta la información de hardware a un archivo CSV.
    """
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Component', 'Detail', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader() # Escribe la fila de encabezado

        for component, details in hardware_info.items():
            for detail, value in details.items():
                writer.writerow({'Component': component, 'Detail': detail, 'Value': value})

if __name__ == "__main__":
    default_filename = 'hardware_info.csv'
    print("Iniciando la obtención de información de hardware...")
    
    hardware_info = get_hardware_info()
    
    print("\nInformación de hardware obtenida:")
    for component, details in hardware_info.items():
        print(f"  {component}:")
        for detail, value in details.items():
            print(f"    - {detail}: {value}")

    print(f"\nExportando información a '{default_filename}'...")
    export_to_csv(hardware_info, filename=default_filename)
    print(f"Información de hardware exportada exitosamente a '{default_filename}'.")