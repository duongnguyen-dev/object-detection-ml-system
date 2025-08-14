import os
import argparse
from loguru import logger
from ultralytics import YOLO

def export(
        model_path: str,
        format: str,
        imgsz: int,
        half: bool,
        device
    ):
    if not os.path.exists(model_path):  
        raise FileNotFoundError(f"{model_path} does not exist!")
    else:
        model = YOLO(model_path)
        logger.info(f"Start converting model to {format }format.")
        model.export(
            format=format,
            imgsz=imgsz,
            half=half,
            nms=False,
            batch=1,
            device=device
        )
        logger.info("Converted successfully.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, help="Directory to model checkpoint")
    
    parser.add_argument("--format", type=str, help="Desired model format")

    parser.add_argument("--imgsz", type=int, help="Input image size")

    parser.add_argument("--half", type=bool, help="Allow quantization")

    parser.add_argument("--device", help="Inference device")

    args = parser.parse_args()

    export(
        model_path=args.model_path,
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        device=args.device
    )