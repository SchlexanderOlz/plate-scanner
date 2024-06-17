import cv2
from crop import extract_candidates
from parse import parse_plate, extract_characters
import imutils
import random
from ocr import OCR

if __name__ == "__main__":
    image: cv2.typing.MatLike = cv2.imread("data/license_plates/3375_1389548347.jpg", cv2.IMREAD_COLOR)

    model = OCR(model_path="model/binary_128_0.50_ver3.pb", 
                            lable_path="model/binary_128_0.50_labels_ver2.txt") 

    image = imutils.resize(image, width=500)

    candidates = extract_candidates(image)
    for candidate in candidates:
        cv2.imshow("Plate" + str(random.random()), candidate)
        res = extract_characters(candidate)

        chars = extract_characters(candidate)
        for char in chars:
            cv2.imshow("char" + str(random.random()), char)
        prediction = model.lable_image_list(chars, 128)
        print(prediction)



    while cv2.waitKey(0) != 27:
        pass
    cv2.destroyAllWindows()