import cv2 
import random
from typing import Sequence
import numpy as np

PLATE_RATIO = 120 / 520

def find_countours(image: cv2.typing.MatLike) -> Sequence[cv2.typing.MatLike]:
    structure = cv2.getStructuringElement( 
                              shape = cv2.MORPH_RECT, ksize =(20, 3)) 

    grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(grayed, (5, 5), 0)

    sobelx = cv2.Sobel(blured, cv2.CV_8U, 1, 0, ksize = 3)  
        
    _, threshold_img = cv2.threshold(sobelx, 0, 255, 
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

    element = structure
    threshhold_copy = threshold_img.copy() 
    cv2.morphologyEx(src = threshold_img, 
                        op = cv2.MORPH_CLOSE, 
                        kernel = element, 
                        dst = threshhold_copy) 


    contours, _ = cv2.findContours(threshhold_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    return contours


def validate_plate(contour: cv2.typing.MatLike) -> bool:
    (_, _), (width, height), angle = cv2.minAreaRect(contour)

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    if not(-10 <= angle <= 10 or 80 <= angle <= 100):
        return False

    if height < 5 or width < 40:
        return False
        
    if width > 1000 or height > 1000:
        return False
    

    # TODO: Check if the plate is rotated and check for vertical plates
    e = 1.0
    if abs(float(height) / float(width) - PLATE_RATIO) >= e:
        return False

    return True


def clean_plate(plate: cv2.typing.MatLike):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) 
    thresh = cv2.adaptiveThreshold(gray, 
                                    255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 
                                    11, 2) 
        
    contours, _ = cv2.findContours(thresh.copy(), 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_NONE) 

    if contours: 
        areas = [cv2.contourArea(c) for c in contours] 
            
        max_index = np.argmax(areas)  

        max_cnt = contours[max_index] 

        x, y, w, h = cv2.boundingRect(max_cnt) 
        (_, _), (width, height), _ = cv2.minAreaRect(max_cnt) 

        print(PLATE_RATIO)
        print(width/height)

        e = 1.0
        if abs(float(width) / float(height) - PLATE_RATIO) >= e:
            return None, None
            
        return plate, [x, y, w, h] 
        
    else: 
        return None, None


def extract_candidates(image: cv2.typing.MatLike) -> Sequence[cv2.typing.MatLike]:
    candidates = []
    contours = find_countours(image)
    for contour in contours:
        if validate_plate(contour):
            x, y, w, h = cv2.boundingRect(contour)
            plate_area_img = image[y:y+h, x:x+w]
            cleaned, coords = clean_plate(plate_area_img)
            if cleaned is not None:
                print("Plate found")
                x, y, w, h = coords
                candidates.append(cleaned)
                cv2.imshow("Plate" + str(random.random()), cleaned)
    
    cv2.waitKey(0)