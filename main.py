import cv2
from crop import crop

if __name__ == "__main__":
    image: cv2.typing.MatLike = cv2.imread("data/license_plates/3375_1389548347.jpg", cv2.IMREAD_GRAYSCALE)

    cv2.imshow("default", image)
    croped = crop(image)
    cv2.imshow("croped", croped)

