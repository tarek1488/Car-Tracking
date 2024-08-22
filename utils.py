import cv2
from easyocr import Reader

reader = Reader(['en'])
def process_plate(plate_crop):
    """
    process plate and read text from it

    Args:
        plate_crop (image): plate number crop

    Returns:
        text in the plate, and detection score
    """
    gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    read =  reader.readtext(gray_plate)
    if len(read) != 0:
        text_bbox, text, score = read[0]
        text = text.upper()
        return text, score
    return -1,-1


def assign_plate_to_car(plate_coordinate, car_tracks):
    """
    Assign plate to cars.
    
    Args:
        plate_coordinate (list): Bounding box of the plate [x1, y1, x2, y2].
        car_tracks (list): List of car tracking data.
    
    Returns:
        dict or None: Car data (bounding box and ID) if a match is found, else None.
    """
    for track in car_tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = plate_coordinate
        x1_car, y1_car, x2_car, y2_car = bbox

        # Debugging output to check the coordinates
        #print(f"Plate: {plate_coordinate}, Car: {bbox}")

        # Check if the plate is within the car's bounding box
        if x1 >= x1_car and y1 >= y1_car and x2 <= x2_car and y2 <= y2_car:
            #print("Match found!")
            return track  # Return the track if a match is found

    #print("No match found")
    return -1  # Return None if no match is found after the loop

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                            car_id,
                            '[{} {} {} {}]'.format(
                                results[frame_nmr][car_id]['car']['bbox'][0],
                                results[frame_nmr][car_id]['car']['bbox'][1],
                                results[frame_nmr][car_id]['car']['bbox'][2],
                                results[frame_nmr][car_id]['car']['bbox'][3]),
                            '[{} {} {} {}]'.format(
                                results[frame_nmr][car_id]['license plate']['bbox'][0],
                                results[frame_nmr][car_id]['license plate']['bbox'][1],
                                results[frame_nmr][car_id]['license plate']['bbox'][2],
                                results[frame_nmr][car_id]['license plate']['bbox'][3]),
                            results[frame_nmr][car_id]['license plate']['plate_bbox_score'],
                            results[frame_nmr][car_id]['license plate']['text'],
                            results[frame_nmr][car_id]['license plate']['text_score']))
        f.close()
        
