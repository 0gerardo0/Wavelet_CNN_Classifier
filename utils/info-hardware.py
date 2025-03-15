import csv
import platform
import psutil

def get_hardware_info():
    hardware_info = {}

    # Información de la GPU
    gpu_info = !nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv
    gpu_name = gpu_info[1].split(',')[0]
    driver_version = gpu_info[1].split(',')[1]
    memory_total = gpu_info[1].split(',')[2]
    hardware_info['GPU'] = {'Name': gpu_name, 'Driver Version': driver_version, 'Memory Total': memory_total}

    # Información de la CPU
    cpu_info = platform.processor()
    hardware_info['CPU'] = {'Processor': cpu_info}

    # Información de la RAM
    ram_info = psutil.virtual_memory().total / (1024**3)  # convertir a gigabytes
    hardware_info['RAM'] = {'Total (GB)': ram_info}

    return hardware_info

def export_to_csv(hardware_info, filename='hardware_info.csv'):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Component', 'Detail', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for component, details in hardware_info.items():
            for detail, value in details.items():
                writer.writerow({'Component': component, 'Detail': detail, 'Value': value})

# Obtener información de hardware
hardware_info = get_hardware_info()

# Exportar información a CSV
export_to_csv(hardware_info)                
