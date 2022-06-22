from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from src.feature_engineering import data_preprocessing

extracted_dataframe = data_preprocessing.feature_extraction()


def label_encoding(y):
    encoder = LabelEncoder()
    y = to_categorical(encoder.fit_transform(y=y))
    encoder_dict = {
        'y': y,
        'encoder_object': encoder
    }
    return encoder_dict


def split_data():
    X = np.array(
        extracted_dataframe['feature'].tolist()
    )

    y = np.array(
        extracted_dataframe['class'].tolist()
    )

    encoder_dict = label_encoding(y=y)
    y = encoder_dict['y']

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.1,
                                                        random_state=0
                                                        )
    encoder = encoder_dict['encoder_object']
    return X_train, X_test, y_train, y_test, encoder
