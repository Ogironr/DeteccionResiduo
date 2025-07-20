import argparse
import os
from ultralytics import YOLO

class ModelValidator:
    """
    Una clase para encapsular la lógica de validación de un modelo YOLOv8.

    Esta clase carga un modelo y un dataset, y proporciona métodos para ejecutar
    la validación en diferentes conjuntos de datos (train, val, test).
    """
    def __init__(self, model_path: str, data_yaml_path: str, project_dir: str):
        """
        Inicializa el validador.

        Args:
            model_path (str): Ruta al archivo de pesos del modelo (.pt).
            data_yaml_path (str): Ruta al archivo de configuración del dataset (.yaml).
            project_dir (str): Directorio donde se guardarán los resultados de la validación.
        """
        self.model_path = model_path
        self.data_yaml_path = data_yaml_path
        self.project_dir = project_dir

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Archivo del modelo no encontrado en: {self.model_path}")
        if not os.path.exists(self.data_yaml_path):
            raise FileNotFoundError(f"Archivo data.yaml no encontrado en: {self.data_yaml_path}")

        print("Cargando modelo desde la clase ModelValidator...")
        self.model = YOLO(self.model_path)
        print("Modelo cargado exitosamente.")

    def validate_split(self, split_name: str):
        """Ejecuta la validación en un conjunto de datos específico."""
        print(f"\n--- INICIANDO VALIDACIÓN EN EL CONJUNTO '{split_name.upper()}' ---")
        
        metrics = self.model.val(
            data=self.data_yaml_path,
            split=split_name,
            project=self.project_dir,
            name=f'{split_name}_set_results',
            imgsz=640,
            batch=16,
            plots=True
        )
        
        print(f"\n--- Resultados del conjunto '{split_name.upper()}' ---")
        print(f"  - mAP50-95 ({split_name}): {metrics.box.map:.4f}")
        print(f"  - mAP50 ({split_name}):    {metrics.box.map50:.4f}")
        print(f"Resultados guardados en: {os.path.join(self.project_dir, f'{split_name}_set_results')}")
        return metrics

    def run_all_validations(self):
        """Ejecuta la validación en los conjuntos 'val' y 'train'."""
        self.validate_split('val')
        self.validate_split('train')
        print("\n\n--- PROCESO DE VALIDACIÓN COMPLETO ---")

def main():
    """
    Punto de entrada para ejecutar el script de validación desde la línea de comandos.
    """
    parser = argparse.ArgumentParser(description="Lanzador de Validación de YOLOv8 para Colab")
    
    parser.add_argument('--model_path', type=str, required=True, help='Ruta al archivo .pt del modelo a validar.')
    parser.add_argument('--data_yaml', type=str, required=True, help='Ruta al archivo data.yaml del dataset.')
    parser.add_argument('--save_dir', type=str, default='runs/validation', help='Directorio base para guardar los resultados.')

    args = parser.parse_args()

    print("--- Iniciando el lanzador de validación de Colab ---")
    try:
        validator = ModelValidator(
            model_path=args.model_path,
            data_yaml_path=args.data_yaml,
            project_dir=args.save_dir
        )
        validator.run_all_validations()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}")

if __name__ == '__main__':
    main()
