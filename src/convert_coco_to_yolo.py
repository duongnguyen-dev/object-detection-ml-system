import os
import shutil
import argparse
from pathlib import Path
from loguru import logger
from ultralytics.data.converter import convert_coco

def convert_coco_to_yolo(labels_dir: str, save_dir: str):
    if not os.path.exists(labels_dir):
        raise FileNotFoundError(f"{labels_dir} doesn't exists!")

    try:
        convert_coco(
            labels_dir=labels_dir,
            save_dir=save_dir,
            use_segments=False,
            use_keypoints=False,
            cls91to80=False,
            lvis=False
        )
        logger.info("Successfully convert COCO to YOLO format.")
    except:
        logger.error("Failed to convert COCO to YOLO format.")
    
    save_dir_parent = Path(save_dir).parent

    try:
        shutil.move(f"{save_dir}/labels/annotations", save_dir_parent)
        logger.info(f"Moved directory successfully")
    except FileNotFoundError:
        logger.error(f"Error: The source directory does not exist.")

    try:
        shutil.rmtree(save_dir)
        logger.info(f"Successfully removed empty directory: {save_dir}")
    except OSError as e:
        logger.error(f"Error: {e}") # e.g., "Directory not empty"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--labels_dir", type=str, required=True, help="The directory to the COCO annotation file."
    )

    parser.add_argument(
        "--save_dir", type=str, required=True, help="The target directory to save annotation files"
    )

    args = parser.parse_args()

    convert_coco_to_yolo(args.labels_dir, args.save_dir)