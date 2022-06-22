from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import time
from datetime import datetime
from src.feature_engineering import split_dataset

EPOCHS = 250
BATCH_SIZE = 32


def define_model_layers(num_labels):
    model = Sequential()
    model.add(Dense(1024, input_shape=(40,),
                    activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(512, input_shape=(40,),
                    activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(256, input_shape=(40,),
                    activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, input_shape=(40,),
                    activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, input_shape=(40,),
                    activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(32, input_shape=(40,),
                    activation="relu"))
    model.add(Dropout(0.3))

    # Final Layer
    model.add(Dense(num_labels, activation="softmax"))

    print(model.summary())
    return model


def train_model():
    X_train, X_test, y_train, y_test, encoder = split_dataset.split_data()
    print('------------Shape----------')
    print(f'Shape of X_train: {X_train.shape}')
    print(f'Shape of Y_train: {y_train.shape}')
    print(f'Shape of X_test: {X_test.shape}')
    print(f'Shape of Y_test: {y_test.shape}')
    NUM_LABELS = y_train.shape[1]
    model = define_model_layers(NUM_LABELS)

    # Compling the model.
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer='adam'
                  )

    current_time = time.strftime("%H:%M:%S",
                                 time.localtime()
                                 )
    # Defining Checkpointer for callbacks
    check_pointer = ModelCheckpoint(filepath=f'saved_models/genre_classification_{current_time}.hdf5',
                                    verbose=1,
                                    save_best_only=True
                                    )
    # Storing Start time
    start_time = datetime.now()

    # Training model

    history_model = model.fit(X_train,
                              y_train,
                              batch_size=BATCH_SIZE,
                              epochs=EPOCHS,
                              validation_data=(X_test, y_test),
                              callbacks=[check_pointer],
                              verbose=1
                              )

    print(f'Total Time taken in training is {datetime.now() - start_time}')
    print("-----Model Evaluation----")
    print(model.evaluate(X_test, y_test, verbose=0))
    return_dict = {
        'model': model,
        'history_model': history_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'encoder': encoder
    }
    return return_dict


# Driver Code
# To train the model
if __name__ == '__main__':
    train_model()
