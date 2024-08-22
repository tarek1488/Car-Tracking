import ast
import cv2
import numpy as np
import pandas as pd

def draw_border(img, top_left, bottom_right, plate_number, color=(0, 255, 0), thickness=5, line_length=50):
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Top side
    cv2.line(img, (x1, y1), (x1 + line_length, y1), color, thickness)  # Top-left horizontal
    cv2.line(img, (x2, y1), (x2 - line_length, y1), color, thickness)  # Top-right horizontal

    # Bottom side
    cv2.line(img, (x1, y2), (x1 + line_length, y2), color, thickness)  # Bottom-left horizontal
    cv2.line(img, (x2, y2), (x2 - line_length, y2), color, thickness)  # Bottom-right horizontal

    # Left side
    cv2.line(img, (x1, y1), (x1, y1 + line_length), color, thickness)  # Top-left vertical
    cv2.line(img, (x1, y2), (x1, y2 - line_length), color, thickness)  # Bottom-left vertical

    # Right side
    cv2.line(img, (x2, y1), (x2, y1 + line_length), color, thickness)  # Top-right vertical
    cv2.line(img, (x2, y2), (x2, y2 - line_length), color, thickness)  # Bottom-right vertical

    # Display the plate number centered between the top corners of the car border
    text_x = (x1 + x2) // 2
    text_y = y1 - 10  # Slightly above the top border
    cv2.putText(img, plate_number, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), thickness)

    return img


results = pd.read_csv('out_full.txt', sep="," ,  encoding='ISO-8859-1')

# Load video
video_path = r'input_videos\in6.mp4'
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    license_plate[car_id] = {'license_crop': None,
                             'license_plate_number': results[(results['car_id'] == car_id) &
                                                             (results['license_number_score'] == max_)]['license_number'].iloc[0]}
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                             (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
    ret, frame = cap.read()

    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                              (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[car_id]['license_crop'] = license_crop


frame_nmr = -1

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Read frames
ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        for row_indx in range(len(df_)):
            # Draw car with license plate number
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            plate_number = license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number']
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), plate_number, (0, 255, 0), 3)

            # Draw license plate
            x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

            # Crop license plate
            license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

            H, W, _ = license_crop.shape
        
        out.write(frame)
        frame = cv2.resize(frame, (1280, 720))
        # if frame_nmr == 510:
        #     break
        # cv2.imshow('frame', frame)
        # cv2.waitKey(0)

out.release()
cap.release()
