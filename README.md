# Music_Genre_Classification

This repository contains the code to build and train a classification model that will predict the genre of the music/audio.

### To install the requirements for the project

`pip install -r requirements.txt`

#### To train the model and make the predictions:
1. First download the dataset from [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download)
2. Then navigate to `src/feature_engineering/data_preprocessing.py`.
3. Update the path variables with:
   1. `DATASET_PATH`
   2. `METADATA_PATH`
4. Now navigate to `src/model_training/model_evaluation_prediction.py`
5. Update the `TEST_AUDIO_PATH`.
6. Go to main function and run it.
7. Your model will be trained and ready prediction for the test audio will be printed.

### Thank You!!

