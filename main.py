import cv2
from crop import crop
import imutils

if __name__ == "__main__":
    image: cv2.typing.MatLike = cv2.imread("data/license_plates/3375_1389548347.jpg", cv2.IMREAD_COLOR)

    image = imutils.resize(image, width=500)
    croped = crop(image)
    cv2.imshow("croped", croped)


    while cv2.waitKey(0) != 27:
        pass
    cv2.destroyAllWindows()