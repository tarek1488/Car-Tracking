from collections import defaultdict
import os
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Create output directory if it doesn't exist
output_path = "output"
os.makedirs(output_path, exist_ok=True)

# Initialize the model
#model = YOLO(r"best (1).pt")
model = YOLO('yolov8s-seg.pt')

# Capture video
cap = cv2.VideoCapture("input_videos/in4.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize video writer
out = cv2.VideoWriter(os.path.join(output_path, "out4-before.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

# Initialize track history (although not used in this script, can be useful for debugging or future needs)
track_history = defaultdict(lambda: [])

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)
    results = model.track(im0, persist=True)

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for mask, track_id in zip(masks, track_ids):
            color = colors(int(track_id), True)
            txt_color = annotator.get_txt_color(color)
            annotator.seg_bbox(mask=mask, mask_color=color, label=str(track_id), txt_color=txt_color)

    out.write(im0)
    cv2.imshow("output-before-fine-tuning", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Video processing interrupted by user.")
        break

# Release resources
out.release()
cap.release()
cv2.destroyAllWindows()
