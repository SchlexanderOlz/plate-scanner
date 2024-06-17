import cv2
from skimage import measure
import imutils
import numpy as np

def parse_plate(plate: cv2.typing.MatLike) -> str:
    return "ABC1234"


def extract_characters(plate: cv2.typing.MatLike) -> list[str]:
    value_channel = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
    plate = imutils.resize(plate, width=400)


    threshhold = cv2.adaptiveThreshold(value_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    threshhold = cv2.bitwise_not(threshhold)
    threshhold = imutils.resize(threshhold, width=400)


    lables = measure.label(threshhold, connectivity=2, background=0)

    candidates = np.zeros(threshhold.shape, dtype="uint8")

    for label in np.unique(lables):
        if label == 0:
            continue

        labelMask = np.zeros(threshhold.shape, dtype="uint8")
        labelMask[lables == label] = 255

        contoures = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contoures = contoures[1] if imutils.is_cv3() else contoures[0]

        if len(contoures) == 0: continue

        max_contour = max(contoures, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(max_contour)



        aspectRatio = w / h
        solidity = cv2.contourArea(max_contour) / float(w * h)
        heightRatio = h / plate.shape[0]

        keep_aspect_ratio = aspectRatio < 1.0
        keep_solidity = solidity > 0.15
        keep_height = heightRatio > 0.4 and heightRatio < 0.95


        if keep_aspect_ratio and keep_solidity and keep_height:
            hull = cv2.convexHull(max_contour)
            cv2.drawContours(candidates, [hull], -1, 255, -1)

    contoures, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    chars = []
    bg_thresh = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    if contoures:
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in contoures] 
            
        (contoures, boundingBoxes) = zip(*sorted(zip(contoures, 
                                                        boundingBoxes), 
                                                      key = lambda b: b[1][i], 
                                                      reverse = False)) 

        inc = 4
        for contour in contoures:
            x, y, w, h = cv2.boundingRect(contour)

            if y > inc:
                y = y - inc
            else:
                y = 0
            if x > inc:
                x = x - inc
            else:
                x = 0
            
            _, converted = cv2.threshold(bg_thresh[y:y+h+(inc * 2), x:x+w+(inc * 2)], 150, 255, cv2.THRESH_BINARY)
            chars.append(cv2.cvtColor(converted, cv2.COLOR_GRAY2RGB))
        return chars 
    return []