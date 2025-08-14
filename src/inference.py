import coremltools as ct
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# 1. Load model
model = YOLO('../runs/detect/train13/weights/best.mlpackage')

cap = cv2.VideoCapture('../data/2023 PPA Atlanta Open - Johns_Waters vs. Newman_Todd - Match Highlights - Mixed Doubles Championship.mp4')  # hoặc 0 cho webcam

while True:
    ret, frame = cap.read()
    # if not ret:
    #     break

    # # 4. Chạy inference, tên input phụ thuộc vào model
    # pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((640, 640))
    # result = model.predict({'image': pil_img})

    # boxes = result["coordinates"]
    # scores = result["confidence"]
  
    # coords = result.get('coordinates', np.array([]))
    # confidences = result.get('confidence', np.array([]))

    # if coords.size and confidences.size:
    #     c = coords[0]
    #     conf = float(confidences[0].max())

    #     if conf > 0.3:
    #         norm_x, norm_y, norm_w, norm_h = c

    #         frame_h, frame_w = frame.shape[:2]

    #         x_center = norm_x * frame_w
    #         y_center = norm_y * frame_h
    #         box_w = norm_w * frame_w
    #         box_h = norm_h * frame_h

    #         xmin = int(x_center - box_w / 2)
    #         ymin = int(y_center - box_h / 2)
    #         xmax = int(x_center + box_w / 2)
    #         ymax = int(y_center + box_h / 2)

    #         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    #         cv2.putText(frame, f'{conf:.2f}', (xmin, ymin - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if ret:
        results = model.track(frame, tracker="botsort.yaml")
        annotated_frame = results[0].plot(font_size=1, line_width=1)
        cv2.imshow("YOLOv3 CoreML Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else: 
        break

cap.release()
cv2.destroyAllWindows()