import cv2
import numpy as np
import os
from collections import deque, namedtuple
import re
import copy
# Define a namedtuple to hold object properties
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="./model_3_6pm.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class TrackedObject:
    def __init__(self, object_id, bbox, center, shape):
        self.id = object_id
        self.bbox = bbox # x, y, w, h
        self.center = center# center of object
        self.shape = shape #area of bounding box
        self.missed_frames = 0 # Not exist in how many frame
        self.appear = 0 #time of appearance
        self.cls = None # classification label
        
# Function to compute the shape similarity (area ratio)
def shape_similarity(area1, area2):
    return abs(area1 - area2) / float(max(area1, area2))

# Function to compute the Euclidean distance between two points
def euclidean_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# Function to detect and track objects in a series of images
def detect_and_track_objects(image_dir, output_dir, detect_mode):
    # Initialize the background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(detectShadows= False)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all images in the directory
    image_files = os.listdir(image_dir)

    image_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    tracked_objects = []  # List to store tracked objects
    next_object_id = 0  # ID counter for new objects
    box_index = 0

    for idx, image_file in enumerate(image_files):
        # Read the image
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        high_res = copy.deepcopy(frame)
        frame = cv2.resize(frame, (960, 540))

        # Apply background subtractor to get the foreground mask
        fg_mask = back_sub.apply(frame)

        # Threshold the mask to get binary image
        _, fg_mask = cv2.threshold(fg_mask, 100, 255, cv2.THRESH_BINARY)

        # Find contours in the mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_objects = []  # List to store detected objects in the current frame

        for contour in contours:
            # Filter out small contours to remove noise
            if cv2.contourArea(contour) > 25:
                # Get the bounding box coordinates
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                shape = w * h  # Using area as a shape descriptor
                current_objects.append((x, y, w, h, center, shape))

        # Mark all tracked objects as missed in the current frame
        for obj in tracked_objects:
            obj.missed_frames += 1

        for (x, y, w, h, center, shape) in current_objects:
            matched = False
            for obj in tracked_objects:
                if shape_similarity(obj.shape, shape) < 0.5 and euclidean_distance(obj.center, center) < 20*obj.missed_frames:
                    # Update the tracked object
                    obj.bbox = (x, y, w, h)
                    obj.center = center
                    obj.shape = shape
                    obj.missed_frames = 0
                    matched = True
                    obj.appear+=1
                    break

            if not matched:
                # Create a new tracked object
                new_object = TrackedObject(next_object_id, (x, y, w, h), center, shape)
                tracked_objects.append(new_object)
                next_object_id += 1

        # Remove objects that haven't been seen for three consecutive frames
        tracked_objects = [obj for obj in tracked_objects if obj.missed_frames <= 10]

        # Draw bounding boxes for objects that have appeared at least twice
        for obj in tracked_objects:
            if obj.missed_frames == 0 and obj.id >= 0 and obj.appear>5:

                x, y, w, h = obj.bbox
                box = high_res[max(y*4-20,0):min((y+h)*4+20, 2160), max(x*4-20,0):min((x+w)*4+20,3840)]
                box = cv2.resize(box, (128,128))
                box = np.asarray(box, dtype=np.float32).reshape(1, 128, 128, 3)
                interpreter.set_tensor(input_details[0]['index'], box)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]

                
                obj.cls= np.argmax(output_data)
                print(obj.cls)

                if obj.cls ==2:
                    box = high_res[y*4:(y+h)*4, x*4:(x+w)*4]
                    box_index+=1
                    cv2.rectangle(high_res, (x*4, y*4), ((x + w)*4, (y + h)*4), (0, 255, 0), 2)
                    cv2.putText(high_res, f'ID: {"drone"}', (x*4, y*4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if obj.cls ==1:
                    box = high_res[y*4:(y+h)*4, x*4:(x+w)*4]
                    box_index+=1
                    cv2.rectangle(high_res, (x*4, y*4), ((x + w)*4, (y + h)*4), (0, 0, 255), 2)
                    cv2.putText(high_res, f'ID: {"bird"}', (x*4, y*4 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if detect_mode:
            # Save the processed frame
            output_path = os.path.join(output_dir, f'output_{idx:04}.png')
            cv2.imwrite(output_path, high_res)

            # Optionally display the frame
            cv2.imshow('Frame', high_res)
            if cv2.waitKey(30) & 0xFF == 27:
                break

    cv2.destroyAllWindows()

image_directory = './6_1'
detect_mode = True
if detect_mode:
    output_directory = './detect_bird'
    # Detect and track objects in the images
    detect_and_track_objects(image_directory, output_directory, detect_mode)
else:
    output_directory = './box_bird'
    detect_and_track_objects(image_directory, output_directory, detect_mode)

