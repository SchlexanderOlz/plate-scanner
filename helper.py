import numpy as np
import keras
import pandas as pd


AZ_DEFAULT_PATH: str = "data/chars/A_Z_Handwritten_Data.csv"
NUMBER_OFFSET: int = 10

def load_az_dataset(path: str = AZ_DEFAULT_PATH) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path, header=None) 
    lables = data.iloc[:, 0] + 10
    return (data.iloc[:, 1:], lables)



def load_minst_dataset() -> tuple[np.ndarray, np.ndarray]:
    (train_data, train_lables), (test_data, test_lables) = keras.datasets.mnist.load_data()
    data = np.vstack([train_data, test_data])
    data = data.reshape(-1, 28*28)
    lables = np.hstack([train_lables, test_lables])

    return (data, lables)
