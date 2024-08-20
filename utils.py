import cv2

def plate_preprocessing(image):
    pass


def assign_plate_to_car(plate_coordinate, car_tracks):
    """
    assign plate to cars
    Args:
        plate_coordinate (list): list of the bounding box of the plate
        car_tracks (lsit): list of car tracking data
    Returns:
        list: car data for the given plate the bounding box of the car and it's id 
    """
    for track in car_tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = plate_coordinate
        x1_car, y1_car, x2_car, y2_car = bbox
        
        if x1 >= x1_car and y1 >= y1_car and x2 <= x2_car and y2 <= y2_car:
            return track
        
        return -1
        
        
    