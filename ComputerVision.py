import numpy as np
from cv2 import cv2
from timeit import default_timer as timer

# from imutils.object_detection import non_max_suppression
# from imutils import paths
# import imutils

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

class ImageManipulator:

    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=False)
    
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
    open_kernel = np.ones((10,10),np.uint8)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))

    bounding_box_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))

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
    
    def check_movement_detected(self):
        return self.movement_detected

    def convert_gray_filtered(self, source):
        # Converting the image to grayscale.
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        # Smoothing without removing edges.
        gray_filtered = cv2.bilateralFilter(gray, 7, 75, 75)
        return gray_filtered
        
    def extract_foreground(self):
        if self.bounding_box is not None:
            y_coordinates = self.bounding_box.get_y_coordinates()
            x_coordinates = self.bounding_box.get_x_coordinates()
            foreground = cv2.morphologyEx(self.foreground, cv2.MORPH_CLOSE, self.close_kernel)
            # foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel_close)            
            # foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel_open)   
            # foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel_close) 
            # foreground = cv2.dilate(foreground,kernel_dilate,iterations = 1)
            extracted_foreground = np.copy(foreground[y_coordinates[0]:y_coordinates[1], x_coordinates[0]:x_coordinates[1]])
            extracted_foreground = cv2.resize(extracted_foreground, dsize = (50,75), interpolation=cv2.INTER_CUBIC)
            return extracted_foreground
        else:
            return None

    def extract_edges(self):
        if self.bounding_box is not None:
            y_coordinates = self.bounding_box.get_y_coordinates()
            x_coordinates = self.bounding_box.get_x_coordinates()
            # Performs Canny edge detection on filtered frame.
            edges_filtered = cv2.Canny(self.gray, 60, 120)

            # Crop off the edges out of the moving area
            cropped_edges = (self.foreground // 255) * edges_filtered

            extracted_edges = np.copy(cropped_edges[y_coordinates[0]:y_coordinates[1], x_coordinates[0]:x_coordinates[1]])
            extracted_edges = cv2.resize(extracted_edges, dsize = (50,75), interpolation=cv2.INTER_CUBIC)
            return extracted_edges
        else:
            return None

    # image splice by contour detection for foreground
    def focus_movement(self, source):

        bounding_box = None

        foreground = cv2.morphologyEx(self.foreground, cv2.MORPH_CLOSE, self.bounding_box_kernel)
        bounding_ret, bounding_thresh = cv2.threshold(foreground, 91, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(bounding_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(contours) != 0:
            contour = max(contours, key = cv2.contourArea)
            x_pos, y_pos, width, height = cv2.boundingRect(contour)
            # bounding_rect = np.array([[x_pos, y_pos, x_pos + width, y_pos + height]])
            # optimal_pick = non_max_suppression(bounding_rect, probs=None, overlapThresh=0.65)
            bounding_box = BoundingBox(x1=x_pos, x2=x_pos + width, y1=y_pos, y2=y_pos + height, width=width, height=height)
            # bounding_box = {'x': (x_pos, x_pos + width), 'y': (y_pos, y_pos + height), "width": width, "height": height}

        return bounding_box

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
    
    def display_cv(self):
        if self.bounding_box is not None:
            self.draw_bounding_box(self.detection_frame)
        cv2.imshow("Detection Frame", self.detection_frame)
        
def showImage(source):

    cv2.imshow('Image', source)

    while True:

        if cv2.waitKey(25) & 0xFF == ord('q'):

            break

    return cv2.destroyAllWindows()

def display(foregroundClassifier, edgeClassifier, videoPath = None, saveTemplate = False, checkTemplate = False, sessionName = None):

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
            extracted_edges = current_frame.extract_edges()
            extracted_foreground = current_frame.extract_foreground()

            if current_frame.check_movement_detected():

                if checkTemplate:
                    
                    total_comparison_time_start = timer()

                    edge_classification = edgeClassifier.classify(extracted_edges)
                    foreground_classification = foregroundClassifier.classify(extracted_foreground)
                    
                    total_comparison_time_end = timer()
                    # print(f"Total comparison time {total_comparison_time_end - total_comparison_time_start} seconds.")
                    
                    if (edge_classification == 'falling') & (foreground_classification == 'falling'):
                        print("fall")
                    elif (edge_classification == 'upright') & (foreground_classification == 'upright'):
                        print("upright")

                if saveTemplate:

                    save_path_foreground_template = f"./templates/cropped_templates/foreground/{sessionName if videoPath is None else videoPath[15:-4]}_{str(frame_count)}.png"
                    save_path_edge_template = f"./templates/cropped_templates/edge/{sessionName if videoPath is None else videoPath[15:-4]}_{str(frame_count)}.png"
                    
                    cv2.imwrite(save_path_foreground_template, extracted_foreground)
                    cv2.imwrite(save_path_edge_template, extracted_edges)
                    
                    frame_count += 1
            
            # Display the resulting frame
            current_frame.display_cv()

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

        # Break the loop
        else: 
            break

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

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