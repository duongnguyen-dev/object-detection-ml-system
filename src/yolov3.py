import os
import argparse
import mlflow
from loguru import logger
from dotenv import load_dotenv
from ultralytics import YOLO, settings

load_dotenv()
settings.update(
	{
		"mlflow": True,
  	}
)

def train_yolov3(
	model_name: str,
	dataset_path: str, 
	epochs: int,
	imgsz: int,
	lr0: float,
	lrf: float,
	batch_size: int,
	optimizer: str,
	mosaic: float,
	fliplr: float,
	scale: float,
	hsv_h: float,
	hsv_s: float,
	hsv_v: float,
	resume: bool,
	last_checkpoint: str,
	exp_id: str
):
	mlflow_uri = os.getenv("MLFLOW_URI")
	if mlflow_uri is not None:
		mlflow.set_tracking_uri(mlflow_uri)
		mlflow.set_experiment(experiment_id=exp_id if exp_id != None else "YOLOv3 Experiment")
	else:
		raise ValueError("MLFLOW_URI environment variable is not set.")

	with mlflow.start_run(run_name="YOLOv3 Experiment"):
		if resume == True:
			if last_checkpoint != '':
				model = YOLO(last_checkpoint)
				model.train(resume=resume)
			else:
				logger.error("Please provide directory to last checkpoint!")
		else:
			model = YOLO(model_name)
			model.train(
				data=dataset_path, 
				epochs=epochs, 
				imgsz=imgsz, 
				lr0=lr0, 
				lrf=lrf,
				batch=batch_size,
				optimizer=optimizer,
				mosaic=mosaic,
				fliplr=fliplr,
				scale=scale,
				hsv_h=hsv_h,
				hsv_s=hsv_s,
				hsv_v=hsv_v,
				device='mps',
				save=True
			)

if __name__=="__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--resume", type=bool)

	parser.add_argument("--last_checkpoint", type=str)

	parser.add_argument("--exp_id", type=str)

	args = parser.parse_args()

	train_yolov3(
		model_name="yolov3u.pt",
		dataset_path='../data/pickleball/data.yaml',
		epochs=50,
		imgsz=640,
		batch_size=4,
		optimizer='AdamW',
		lr0=1e-4,
		lrf=0.001,
		mosaic=1.0,
		fliplr=0.5,
		scale=0.5,
		hsv_h=0.015,
		hsv_s=0.7,
		hsv_v=0.4,
		resume=args.resume,
		last_checkpoint=args.last_checkpoint,
		exp_id="878700292683961880"
	)