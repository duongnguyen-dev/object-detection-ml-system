from ultralytics import YOLO, settings

settings.update({"mlflow": True})
settings.reset()