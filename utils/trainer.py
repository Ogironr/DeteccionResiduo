import torch
from ultralytics import YOLO

def start_training(data_config_path: str, model_path: str, project_name: str, run_name: str, epochs: int):
    """
    Función para entrenar un modelo YOLOv8 con hiperparámetros avanzados
    y data augmentation específica para la detección de residuos.

    Args:
        data_config_path (str): Ruta al archivo data.yaml del dataset.
        model_path (str): Ruta al modelo .pt para iniciar el entrenamiento (ej. 'yolov8s.pt' o 'best.pt').
        project_name (str): Nombre del proyecto para guardar los resultados.
        run_name (str): Nombre específico para esta ejecución del entrenamiento.
        epochs (int): Número de épocas para el entrenamiento.
    """
    # Selecciona el dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Usando dispositivo: {device}')

    # Cargar el modelo especificado (para entrenamiento desde cero o fine-tuning)
    print(f"Cargando modelo desde: {model_path}")
    model = YOLO(model_path)

    # Entrenamiento del modelo con hiperparámetros y aumentos de datos
    print("Iniciando el entrenamiento avanzado de YOLOv8...")
    results = model.train(

        data=data_config_path,    # Ruta al archivo data.yaml (dónde están las imágenes y clases)
        epochs=epochs,            # Número de vueltas completas al dataset
        imgsz=640,               # Tamaño de imagen para entrenamiento (640x640 píxeles)
        batch=64,                # Número de imágenes procesadas simultáneamente
        workers=16,              # Número de procesos paralelos para cargar datos
        device=0 if device == 'cuda' else 'cpu', # GPU a usar (0 = primera GPU)
        project=project_name,    # Carpeta donde guardar resultados
        name=run_name,          # Nombre específico de esta ejecución
        
        # --- Hiperparámetros Clave ---
        lr0=0.001,              # Tasa de aprendizaje inicial (qué tan rápido aprende)
        optimizer='SGD',        # Algoritmo de optimización (SGD es robusto y estable)
        patience=50,            # Épocas a esperar sin mejora antes de parar automáticamente
        
        # --- Aumento de Datos (Data Augmentation) ---
        augment=True,           # Habilita todas las técnicas de aumento de datos
        cache=True,             # Guarda imágenes en RAM para acceso más rápido
        
        # --- Transformaciones de Color ---
        hsv_h=0.015,           # Variación de matiz (color): ±1.5%
        hsv_s=0.7,             # Variación de saturación: ±70%
        hsv_v=0.4,             # Variación de brillo: ±40%
        
        # --- Transformaciones Geométricas ---
        degrees=5.0,           # Rotación aleatoria: ±5 grados
        translate=0.1,         # Desplazamiento: ±10% del tamaño de imagen
        scale=0.5,             # Escalado: ±50% del tamaño original
        flipud=0.0,            # Volteo vertical: 0% (deshabilitado)
        fliplr=0.5,            # Volteo horizontal: 50% de probabilidad
        
        # --- Parámetros de Guardado y Visualización ---
        plots=True,  # Asegura que se generen gráficos
        save=True,   # Asegura que se guarden los logs
    )

    print("Entrenamiento finalizado.")
    print(f"Resultados y modelo final guardados en: {model.trainer.save_dir}")
    return model.trainer.save_dir
