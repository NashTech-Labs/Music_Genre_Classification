import numpy as np

from src.feature_engineering import data_preprocessing
from src.model_training import train

TEST_AUDIO_PATH = '/home/knoldus/Downloads/Data/genres_original/blues/blues.00000.wav'
trained_model_object = train.train_model()
model = trained_model_object['model']
history = trained_model_object['history_model']
X_train = trained_model_object['X_train']
X_test = trained_model_object['X_test']
y_train = trained_model_object['y_train']
y_test = trained_model_object['y_train']
encoder = trained_model_object['encoder']


def predicting_model():
    predicting_X_test = np.argmax(model.predict(X_test),
                                  axis=1
                                  )
    print(f'Prediction of X_test: {predicting_X_test}')

    # predicting genre of audio
    mfccs_scaled_features = data_preprocessing.feature_extractor(TEST_AUDIO_PATH)

    print(f'mfccs features before reshaping: {mfccs_scaled_features}')

    # Reshaping features
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    print(f'mfccs features after reshaping: {mfccs_scaled_features}')

    print(f'Shape of scaled features: {mfccs_scaled_features.shape}')

    # predicting the label for test audio

    label_predicted = np.argmax(model.predict(mfccs_scaled_features), axis=1)

    print(f'The predicted label: {label_predicted}')

    # predicting class of the test audio
    class_predicted = encoder.inverse_transform(label_predicted)

    print(f'The predicted genre for the input audio is {class_predicted[0]}')


# Driver Code
# To train the model and make prediction
if __name__ == '__main__':
    predicting_model()
