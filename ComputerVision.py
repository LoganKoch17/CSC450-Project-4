"""
ComputerVision.py is the Computer Vision component referenced in section 3.2 of the SDD, A03_SDD_Team4.docx. 




Copyright (c) 2020 Fall Detection System, All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1.	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2.	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
    in the documentation and/or other materials provided with the distribution.

3.	Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
    from this software without specific prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

"""

import numpy as np
from cv2 import cv2
from queue import Queue 
from timeit import default_timer as timer

from imutils.object_detection import non_max_suppression
from imutils import paths
import imutils

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

"""
BoundingBox referenced and defined in section 2.0 of the SDD, A03_SDD_Team4.docx. 

"""

class BoundingBox:
    
    def __init__(self, x1, x2, y1, y2, width, height):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.width = width
        self.height = height
    
    def get_x_coordinates(self):
        return self.x1, self.x2
    
    def get_y_coordinates(self):
        return self.y1, self.y2
    
    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def get_area(self):
        return self.height * self.width
    
    def get_center(self):
        
        x_center = self.x1 + (self.x2 - self.x1)//2
        y_center = self.y1 + (self.y2 - self.y1)//2

        return x_center, y_center
    
    # Adjusts bounding box size to maintain average.
    def change_dimensions(self, width, height):
        x_center, y_center = self.get_center()

        self.width = width

        self.x1 = x_center - (width//2)
        if self.x1 < 0 or self.x1 > SCREEN_WIDTH: self.x1 = 0

        self.x2 = x_center + (width//2)
        if self.x2 < 0 or self.x2 > SCREEN_WIDTH: self.x2 = SCREEN_WIDTH
        
        self.height = height

        self.y1 = y_center - (height//2)
        if self.y1 < 0 or self.y1 > SCREEN_HEIGHT: self.y1 = 0

        self.y2 = y_center + (height//2)
        if self.y2 < 0 or self.y2 > SCREEN_HEIGHT: self.y2 = SCREEN_HEIGHT
"""
FrameInfo referenced and defined in section 2.0 of the SDD, A03_SDD_Team4.docx. 

"""
class FrameInfo:

    def __init__(self, edgeClassification, foregroundClassification):
        self.edge_classification = edgeClassification
        self.foreground_classification = foregroundClassification

"""
FrameHistory referenced and defined in section 2.0 of the SDD, A03_SDD_Team4.docx. 

    The Functional Requirement:

    FR.3 The system must detect when an object classified as “human” falls when an object is obstructing them.

    is addressed in this class and utilized in the display function.
"""

class FrameHistory:

    def __init__(self, boundingBoxSaveCount = 10, frameInfoSaveCount = 10):

        self.bounding_box_save_count = boundingBoxSaveCount
        self.bounding_boxes = np.array([])

        self.frame_info_save_count = frameInfoSaveCount
        self.frame_classifications = np.array([])

    # Averages the size of the bounding box array.    
    def average_dimensions(self):
        width = 0
        height = 0

        for bounding_box in self.bounding_boxes:
            width += bounding_box.get_width()
            height += bounding_box.get_height()
        
        width = width//len(self.bounding_boxes)
        height = height//len(self.bounding_boxes)

        return width, height
    
    # Checks to see if the bounding box has increased recently.
    def check_continuous_decrease(self):

        previous_area = 640 * 480
        continuous_decrease = True
        
        for bounding_box in self.bounding_boxes:
            area = bounding_box.get_area()

            if area > previous_area:
                continuous_decrease = False

            previous_area = area

        return continuous_decrease

    def average_area(self):
        width, height = self.average_dimensions()
        return width * height
    
    def add_bounding_box(self, boundingBox):
        self.bounding_boxes = np.append(self.bounding_boxes, boundingBox)

    def forget_bounding_box(self, frameCount):
        self.bounding_boxes = np.delete(self.bounding_boxes, np.arange(frameCount))

    def bounding_box_full(self):
        if len(self.bounding_boxes) == self.bounding_box_save_count:
            return True
        else:
            return False
    
    def add_frame_info(self, frameInfo):
        self.frame_classifications = np.append(self.frame_classifications, frameInfo)

    def forget_frame_info(self, frameCount):
        self.frame_classifications = np.delete(self.frame_classifications, np.arange(frameCount))
    
    def frame_info_full(self):
        if len(self.frame_classifications) == self.frame_info_save_count:
            return True
        else:
            return False
    
    def majority_falling(self):

        fall_classifications = 0
        total_classifications = 0

        for frame in self.frame_classifications:
            if frame.edge_classification == 'falling':
                fall_classifications += 1
            if frame.foreground_classification == 'falling':
                fall_classifications +=1
        
        if fall_classifications > (total_classifications/2):
            return True
        else:
            return False
    
"""
ImageManipulator referenced and defined in section 2.0 of the SDD, A03_SDD_Team4.docx. 

The Functional Requirement:

FR.1 The system must detect and classify a human as a "human" object.

is addressed in this class in the focus_movement() method.


The Fucntional Requirement:

FR.7 The system must eliminate background noise of inanimate objects and non-human animated objects.

is addressed in extract_foreground() and extract_edges() of the ImageManipulator class.
"""
class ImageManipulator:

    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=False)
    bounding_box_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))

    # Initialize default values, makes copy of current frame.
    def __init__(self, frame):
        self.source = frame
        self.detection_frame = self.source.copy()
        self.gray = self.convert_gray_filtered(self.source)
        self.foreground = self.fgbg.apply(self.gray, learningRate = 0.02)
        
        self.bounding_box = self.focus_movement(self.source)
        if self.bounding_box is not None:
            self.movement_detected = True
        else:
            self.movement_detected = False
        
        self.close_kernel = self.get_dynamic_kernel_size()
    
    def check_movement_detected(self):
        return self.movement_detected
    
    def get_bounding_box(self):
        return self.bounding_box
    
    def set_bounding_box(self, boundingBox):
        self.bounding_box = boundingBox

    # Adjusts the kernel size dynamically based on a distance calculation.
    def get_dynamic_kernel_size(self):
        kernel_size = 30

        if (self.movement_detected):

            width = self.bounding_box.get_width()

            # Find Distance by Subject's Width Relative to Camera.
            if(width >= 250):
                kernel_size = 30

            if(width < 250 and width >= 120):
                kernel_size = 25

            if(width < 120 and width >= 100):
                kernel_size = 20

            if(width < 100 and width >= 60):
                kernel_size = 15

            if(width < 60 and width >= 40):
                kernel_size = 10

            if(width < 40 and width >= 20):
                kernel_size = 5

            if(width < 20):
                kernel_size = 10

        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    def convert_gray_filtered(self, source):
        # Converting the image to grayscale.
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        # Smoothing without removing edges.
        gray_filtered = cv2.bilateralFilter(gray, 7, 75, 75)
        return gray_filtered
        
    # Splices the foreground image and returns it.
    def extract_foreground(self):
        if self.bounding_box is not None and self.bounding_box.get_area() > 1000:
            y1, y2 = self.bounding_box.get_y_coordinates()
            x1, x2 = self.bounding_box.get_x_coordinates()
            foreground = cv2.morphologyEx(self.foreground, cv2.MORPH_CLOSE, self.close_kernel)
            extracted_foreground = np.copy(foreground[y1:y2, x1:x2])
            extracted_foreground = cv2.resize(extracted_foreground, dsize = (50,75), interpolation=cv2.INTER_CUBIC)
            return extracted_foreground
        else:
            return None

    # Splices the edge image and returns it.
    def extract_edges(self):
        if self.bounding_box is not None and self.bounding_box.get_area() > 1000:
            y1, y2 = self.bounding_box.get_y_coordinates()
            x1, x2 = self.bounding_box.get_x_coordinates()

            # Performs Canny edge detection on filtered frame.
            edges_filtered = cv2.Canny(self.gray, 60, 120)

            # Crop off the edges out of the moving area
            cropped_edges = (self.foreground // 255) * edges_filtered

            extracted_edges = np.copy(cropped_edges[y1:y2, x1:x2])
            extracted_edges = cv2.resize(extracted_edges, dsize = (50,75), interpolation=cv2.INTER_CUBIC)
            return extracted_edges
        else:
            return None

    # Creates the bounding boxes and finds the best one.
    def focus_movement(self, source):

        bounding_box = None

        foreground = cv2.morphologyEx(self.foreground, cv2.MORPH_CLOSE, self.bounding_box_kernel)
        bounding_ret, bounding_thresh = cv2.threshold(foreground, 91, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(bounding_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(contours) != 0:
            contour = max(contours, key = cv2.contourArea)
            x_pos, y_pos, width, height = cv2.boundingRect(contour)
            bounding_rect = np.array([[x_pos, y_pos, x_pos + width, y_pos + height]])
            optimal_pick = non_max_suppression(bounding_rect, probs=None, overlapThresh=0.65)
            for (x1, y1, x2, y2) in optimal_pick:
                bounding_box = BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2, width= x2 - x1, height= y2 - y1)

        return bounding_box

    # Draws the best bounding box to the video feed.
    def draw_bounding_box(self, source, minArea=500, bufferSpace=40):

        width = self.bounding_box.get_width()
        height = self.bounding_box.get_height()
        x_coordinates = self.bounding_box.get_x_coordinates()
        y_coordinates = self.bounding_box.get_y_coordinates()

        box_area = width * height

        if (abs(box_area) > minArea):

            if(width + bufferSpace > height): 
                
                cv2.rectangle(source, (x_coordinates[0], y_coordinates[0]), (x_coordinates[1], y_coordinates[1]), (0, 0, 255), 2)

            else:
                
                cv2.rectangle(source, (x_coordinates[0], y_coordinates[0]), (x_coordinates[1], y_coordinates[1]), (0, 255, 0), 2)
    
    # Display the live video feed and/or foreground frame.
    def display_cv(self, showBox = True):
        if self.bounding_box is not None and showBox:
            self.draw_bounding_box(self.detection_frame)

        cv2.imshow('Detection Frame', self.detection_frame)
        cv2.imshow('Foreground Frame', self.foreground)
        
def showImage(source):

    cv2.imshow('Image', source)

    while True:

        if cv2.waitKey(25) & 0xFF == ord('q'):

            break

    return cv2.destroyAllWindows()

"""
The Functional Requirement:

FR.2 The system must alert the user when an object classified as a “human” falls in an indoor environment.

is addressed under the checkTemplates if branch in the display function.
The code in this branch also represents the Alert component referenced in section 3.2 of the SDD, A03_SDD_Team4.docx. 


The Functional Requirement:

FR.6 The system must access the live video feed.

is addressed in the display function()
"""
def display(foregroundClassifier, edgeClassifier, videoPath = None, saveTemplate = False, checkTemplate = False, sessionName = None):

    BOUNDING_BOX_COUNT = 5
    FRAME_INFO_COUNT = 5
    frame_history = FrameHistory(boundingBoxSaveCount=BOUNDING_BOX_COUNT, frameInfoSaveCount=FRAME_INFO_COUNT)
    fall_detected = False
    fall_counter = 0
    frame_count = 0

    if saveTemplate:
        print("Saving frames as templates")

    if videoPath is not None:
        cap = cv2.VideoCapture(videoPath)

    else:
        cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")   

    print("""
    Button      Command
    1           Toggles saving 
    2           Toggles classification 
    """)

    # Read the video
    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:

            current_frame = ImageManipulator(frame)

            if current_frame.check_movement_detected():

                # Current frame bounding box gets added to frame history
                if frame_history.bounding_box_full():
                    frame_history.forget_bounding_box(1)
                frame_history.add_bounding_box(current_frame.get_bounding_box())

                if not frame_history.check_continuous_decrease():
                    # Check for potential obstruction
                    average_area = frame_history.average_area()
                    current_bounding_box = current_frame.get_bounding_box()

                    if current_bounding_box.get_area() < average_area:
                        
                        width, height = frame_history.average_dimensions()
                        current_bounding_box.change_dimensions(width, height)
                        current_frame.set_bounding_box(current_bounding_box)
                    
                    extracted_edges = current_frame.extract_edges()
                    extracted_foreground = current_frame.extract_foreground()

                    # Displays to the console which classification, if any, is detected each frame.
                    if checkTemplate:

                        edge_classification = edgeClassifier.classify(extracted_edges)
                        foreground_classification = foregroundClassifier.classify(extracted_foreground)
                        
                        if (edge_classification == 'falling') or (foreground_classification == 'falling'):
                            print("fall")
                            fall_counter = fall_counter + 1
                        elif (edge_classification == 'upright') or (foreground_classification == 'upright'):
                            print("upright")
                        elif (edge_classification == 'sitting') or (foreground_classification == 'sitting'):
                            print("sitting")
                        elif (edge_classification == 'lying') or (foreground_classification == 'lying'):
                            print("lying")
                        elif (edge_classification == 'unrecognized') or (foreground_classification == 'unrecognized'):
                            print("unrecognized object")
                        
                    if saveTemplate:

                        save_path_foreground_template = f"./templates/cropped_templates/foreground/{sessionName if videoPath is None else videoPath[15:-4]}_{str(frame_count)}.png"
                        save_path_edge_template = f"./templates/cropped_templates/edge/{sessionName if videoPath is None else videoPath[15:-4]}_{str(frame_count)}.png"
                        
                        cv2.imwrite(save_path_foreground_template, extracted_foreground)
                        cv2.imwrite(save_path_edge_template, extracted_edges)
                    
                    # Display the resulting frame
                    current_frame.display_cv()

                else:
                    current_frame.display_cv(showBox=False)

            # controls
            key = cv2.waitKey(25) & 0xFF

            # Press Q on keyboard to exit
            if key == ord('q'):
                break

            # save frame as template
            elif key == ord('1'):
                saveTemplate = not saveTemplate
                print(f'saveTemplate: {saveTemplate}')

            # clasify frame with templates
            elif key == ord('2'):
                checkTemplate = not checkTemplate
                print(f'checkTemplate: {checkTemplate}')

            frame_count += 1

        # Break the loop
        else: 
            break

    # Release the video capture object
    cap.release()

    cv2.destroyAllWindows()
    if (fall_counter > 3):
        fall_detected = True
    return fall_detected

def imagePathToByteString(path):
    with open(path, 'rb') as f:
        return f.read()

def byteStringToImage(byteString):
    decoded = cv2.imdecode(np.frombuffer(byteString, np.uint8), -1)
    return decoded

def imagePathToByteArray(path):
    with open(path, 'rb') as f:
        byte_array = bytearray(f.read())
        return byte_array
