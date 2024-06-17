import cv2
import random
from typing import Sequence
import numpy as np

PLATE_RATIO = 520 / 120


def find_contours(image: cv2.typing.MatLike) -> Sequence[cv2.typing.MatLike]:
    structure = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(20, 3))

    grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(grayed, (5, 5), 0)

    sobelx = cv2.Sobel(blured, cv2.CV_8U, 1, 0, ksize=3)

    _, threshold_img = cv2.threshold(
        sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    element = structure
    threshhold_copy = threshold_img.copy()
    cv2.morphologyEx(
        src=threshold_img, op=cv2.MORPH_CLOSE, kernel=element, dst=threshhold_copy
    )

    contours, _ = cv2.findContours(
        threshhold_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def ratioCheck(area, width, height):

    min = 4100
    max = 15000

    ratioMin = 3
    ratioMax = 6

    ratio = float(width) / float(height)

    if ratio < 1:
        ratio = 1 / ratio

    if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
        return False

    return True


def validate_plate(contour: cv2.typing.MatLike) -> bool:
    (_, _), (width, height), angle = cv2.minAreaRect(contour)
    # TODO: Check if the plate is rotated and check for vertical plates

    if width > height:
        angle = -angle
    else:
        angle = 90 + angle

    if angle > 30:
        return False

    if height == 0 or width == 0:
        return False

    area = width * height

    min = 4100
    max = 15000

    ratioMin = 2.5
    ratioMax = 7

    ratio = float(width) / float(height)

    if ratio < 1:
        ratio = 1 / ratio

    if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
        return False

    return True


def clean_plate(plate: cv2.typing.MatLike):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if contours:
        areas = [cv2.contourArea(c) for c in contours]

        max_index = np.argmax(areas)

        max_cnt = contours[max_index]
        max_cntArea = areas[max_index]
        x, y, w, h = cv2.boundingRect(max_cnt)
        rect = cv2.minAreaRect(max_cnt)
        if not ratioCheck(max_cntArea, plate.shape[1], plate.shape[0]):
            return None, None

        return plate, [x, y, w, h]
    else:
        return None, None


def extract_candidates(image: cv2.typing.MatLike) -> Sequence[cv2.typing.MatLike]:
    candidates = []
    contours = find_contours(image)
    for contour in contours:
        if validate_plate(contour):
            x, y, w, h = cv2.boundingRect(contour)
            plate_area_img = image[y : y + h, x : x + w]

            cleaned, coords = clean_plate(plate_area_img)
            if cleaned is not None:
                print("Plate found")
                x, y, w, h = coords
                candidates.append(cleaned)

    cv2.waitKey(0)
    return candidates
