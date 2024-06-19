import sklearn.svm
import cv2
import numpy as np
import sklearn
import random
import joblib

class OCR:

    def __init__(self, model_path: str) -> None:
        self.model: sklearn.svm.SVC = self.load_graph(model_path)
    
    def load_graph(self, model_path: str) -> sklearn.svm.SVC:
        model = None
        with open(model_path, 'rb') as file:
            model = joblib.load(file)
        return model 


    def array_from_image(self, image: cv2.typing.MatLike):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(gray_image, (28, 28))
        cv2.imshow("adasd" + str(random.random()), image)
        return image.reshape(1, -1)


    def translate_lable(self, lable: int):
        if lable < 10:
            return str(lable)
        else:
            section = int((lable - 10) / 26)

            if section == 0:
                return chr(65 + (lable - 10))
            else:
                return chr(97 + (lable - 10))

    def lable_image(self, image: np.array) -> str:
        prediction = self.model.predict(image)
        return self.translate_lable(prediction[0])
    
    def lable_image_list(self, images: list[cv2.typing.MatLike]) -> str:
        return "".join([self.lable_image(self.array_from_image(image)) for image in images])
