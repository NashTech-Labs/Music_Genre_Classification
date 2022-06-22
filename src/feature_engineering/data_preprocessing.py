import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
from tqdm import tqdm

DATASET_PATH = '/home/knoldus/Downloads/Data/genres_original'
METADATA_PATH = '/home/knoldus/Downloads/Data/features_30_sec.csv'


def feature_extractor(input_file):
    audio, sample_rate = librosa.load(input_file,
                                      res_type='kaiser_fast'
                                      )
    mfccs_features = librosa.feature.mfcc(y=audio,
                                          sr=sample_rate,
                                          n_mfcc=40
                                          )
    mfccs_features_scaled = np.mean(mfccs_features.T,
                                    axis=0
                                    )
    return mfccs_features_scaled


def feature_extraction():
    metadata = pd.read_csv(METADATA_PATH)
    extracted_features = []

    for index, row in tqdm(metadata.iterrows(), desc='Extracting'):
        try:
            class_labels = row['label']
            input_file_name = os.path.join(os.path.abspath(DATASET_PATH),
                                           class_labels + '/',
                                           str(row['filename'])
                                           )
            extracted_data = feature_extractor(input_file_name)
            extracted_features.append([extracted_data, class_labels])
        except Exception as e:
            print(f'Error Occurred: {e}')
            continue

    extracted_features_df = pd.DataFrame(extracted_features,
                                         columns=['feature', 'class']
                                         )
    return extracted_features_df
