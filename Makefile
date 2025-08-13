validate_mps:
	python src/validate_mps.py

convert_coco_to_yolo:
	python src/convert_coco_to_yolo.py --labels_dir data/train_v2 --save_dir data/train_v2/labels
	python src/convert_coco_to_yolo.py --labels_dir data/val_v2 --save_dir data/val_v2/labels 