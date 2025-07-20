import argparse
import os
import sys

# Añade el directorio actual al path de Python para que pueda encontrar la carpeta 'utils'.
# Esto hace que el script sea más robusto en el entorno de Colab.
sys.path.append(os.getcwd())

from utils.trainer import start_training

def main():
    """
    Punto de entrada principal para lanzar el entrenamiento desde Colab.
    Este script actúa como un intermediario, pasando los argumentos
    a la función de entrenamiento centralizada en utils/trainer.py.
    """
    parser = argparse.ArgumentParser(description="Lanzador de Entrenamiento de YOLOv8 para Colab")
    
    # Argumentos que necesita utils.trainer.start_training
    parser.add_argument('--data_yaml', type=str, required=True, help='Ruta al archivo data.yaml del dataset.')
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo .pt para iniciar (ej. yolov8s.pt o best.pt).')
    parser.add_argument('--save_dir', type=str, default='runs/train', help='Directorio base para guardar los resultados (project_name).')
    parser.add_argument('--run_name', type=str, required=True, help='Nombre específico para la carpeta de esta ejecución.')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas para entrenar.')

    args = parser.parse_args()

    print("--- Iniciando el lanzador de entrenamiento de Colab ---")
    print(f"Llamando a la función de entrenamiento con los siguientes parámetros:")
    print(f"  - Archivo de datos: {args.data_yaml}")
    print(f"  - Modelo inicial: {args.model}")
    print(f"  - Directorio del proyecto: {args.save_dir}")
    print(f"  - Nombre de la ejecución: {args.run_name}")
    print(f"  - Épocas: {args.epochs}")
    print("----------------------------------------------------")

    # Llamar a la función de entrenamiento centralizada
    start_training(
        data_config_path=args.data_yaml,
        model_path=args.model,
        project_name=args.save_dir,
        run_name=args.run_name,
        epochs=args.epochs
    )

if __name__ == '__main__':
    main()




