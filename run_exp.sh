#!/bin/bash

# Función para revertir los cambios
revertir_cambios() {
    sed -i "s/^wavelet =.*/wavelet = '$palabra_original'/" "$archivo_python"
    echo "Cambios restaurados en $archivo_python"
}

# Función para revertir los cambios en caso de error
excepcion_errores() {
    if [[ -f "$archivo_python.bak" ]]; then
        cp "$archivo_python.bak" "$archivo_python"
    fi
}

# Manejo de excepciones
trap 'excepcion_errores; exit 1' ERR

# Verificar si está activo el entorno virtual
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "Activando entorno virtual..."
    source .venv/bin/activate
fi

# Ruta del archivo
archivo_python='FDM-WCNN_arc1b.py'

# Verificar si el archivo de Python existe
if [ ! -f "$archivo_python" ]; then
    echo "Error: El archivo de Python no existe: $archivo_python"
    exit 1
fi

# Definir la lista de wavelets y el número de experimentos
wavelets=('haar' 'db10' 'sym2' 'coif1' 'bior1.1')

# Solicitar al usuario que ingrese el número del experimento
read -p "Ingrese el número del experimento: " num_experimento

# Almacenar la palabra original
palabra_original=$(sed -n 's/^wavelet = \(\w*\).*/\1/p' "$archivo_python")

# Copiar el archivo original como respaldo
cp "$archivo_python" "$archivo_python.bak"

# Iterar sobre las wavelets
for wavelet in "${wavelets[@]}"; do
    # Modificar la línea de wavelet en el archivo Python
    sed -i "s/^wavelet =.*/wavelet = '$wavelet'/" "$archivo_python"

    # Modificar la línea de experimento en el archivo Python
    sed -i "s/^experimento =.*/experimento = $num_experimento/" "$archivo_python"

    # Ejecutar el programa en Python
    echo "Ejecutando $archivo_python con wavelet $wavelet"
    if python3 "$archivo_python"; then
        echo "Finalizado la ejecución de $archivo_python con wavelet $wavelet"
    else
        echo "Error de ejecución: $archivo_python con wavelet $wavelet"
    fi
done

# Restaurar la palabra original al final
revertir_cambios
