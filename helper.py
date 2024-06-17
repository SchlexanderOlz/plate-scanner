from tensorflow.keras.datasets import mnist
import numpy as np

AZ_DEFAULT_PATH = "data/az_dataset.npz"

def load_az_dataset(path: str = AZ_DEFAULT_PATH) -> tuple[np.ndarray, np.ndarray]:
    data = []
    lables = []

    with open(path) as f:
        for line in f:
            row = line.strip().split(",")
            lable = int(row[0])
            image = np.array([int(x) for x in row[1:]], dtype="uint8") # Grayscale image values

            # Model requires shape of 28x28
            image = image.reshape((28, 28))

            data.append(image)
            lables.appen(lable)

    data = np.array(data, dtype="float32")
    lables = np.array(lables, dtype="int")

    return (data, lables)

def load_minst_dataset() -> tuple[np.ndarray, np.ndarray]:
    (train_data, train_lables), (test_data, test_lables) = mnist.load_data()
    data = np.vstack([train_data, test_data])
    lables = np.hstack([train_lables, test_lables])

    return (data, lables)
