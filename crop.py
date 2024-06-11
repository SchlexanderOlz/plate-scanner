import cv2 

def crop(image: cv2.typing.MatLike):
    blured: cv2.typing.MatLike = cv2.GaussianBlur(image, (5, 5), 0)
    return image
