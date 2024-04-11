#!/bin/bash

if [[ -z "${VIRTUAL_ENV}" ]]; then
    source .venv/bin/activate
fi

python_files=("FDM-WCNN_arc1a-haar.py" "FDM-WCNN_arc1b-db10.py" "FDM-WCNN_arc1b-sym2.py" "FDM-WCNN_arc1b-coif1.py" "FDM-WCNN_arc1b-bior1.1.py")

for file in "${python_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: El archivo de Python no existe: $file"
        continue
    fi
    echo "Ejecutando $file"
    python3 "$file"
    if [ $? -eq 0 ]; then
        echo "Finalizado $file"
    else
        echo "Error de ejecucion: $file"
    fi
done
echo "Todos los scripts ejecutados exitosamente"

