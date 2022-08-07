import random
import multiprocessing as mp

import librosa
from librosa import feature
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

fn_list_i = [
    feature.chroma_stft,
    feature.spectral_centroid,
    feature.spectral_bandwidth,
    feature.spectral_rolloff,
    feature.mfcc,
    feature.chroma_cqt,
    feature.chroma_cens,
    feature.melspectrogram,
    feature.spectral_contrast,
    feature.poly_features,
    feature.tonnetz,
    feature.tempogram,
    feature.fourier_tempogram,
]

fn_list_ii = [
    feature.rms,
    feature.zero_crossing_rate,
    feature.spectral_flatness,
]


def get_feature_vector(y, sr):
    feat_vect_i = [np.mean(funct(y, sr)) for funct in fn_list_i]
    feat_vect_ii = [np.mean(funct(y)) for funct in fn_list_ii]
    feature_vector = feat_vect_i + feat_vect_ii
    return feature_vector


def generate_features(row):
    audio = row['audio']

    features = dict()
    mfcc = librosa.feature.mfcc(y=audio['array'], sr=audio['sampling_rate'], n_mfcc=32)
    features.update({f'mfcc_{i}': np.mean(v) for i, v in enumerate(mfcc)})

    features['zero_crossing'] = np.mean(librosa.feature.zero_crossing_rate(y=audio['array']))
    features['spec_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio['array'], sr=audio['sampling_rate']))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio['array'], sr=audio['sampling_rate']))

    # vec = get_feature_vector(y=audio['array'], sr=audio['sampling_rate'])
    # features = {f'f_{i}': v for i, v in enumerate(vec)}

    return features


def generate_dataset(source_data):

    dataset = list()
    for i, row in tqdm(enumerate(source_data)):

        # if i > 10000:
        #     break
        #
        # j = random.randint(0, len(source_data) - 1)
        # row = source_data[j]
        features = generate_features(row)
        features['target'] = row['label']
        dataset.append(features)

    dataset = pd.DataFrame(dataset)
    features_columns = [col for col in dataset.columns if col != 'target']

    X, y = dataset[features_columns], dataset['target']
    return X, y
    # return dataset


def generate_dataset_parallel(source_data):


    num_workers = mp.cpu_count()
    # num_workers = max(num_workers - 2, 1)
    num_workers = 10
    with mp.Pool(processes=num_workers) as pool:
        # load_df_partial = partial(calculate_stats, texts=texts)
        result = pool.map(generate_dataset, np.array_split(source_data, num_workers))

    dataset = pd.concat(result)
    # return  result

    features_columns = [col for col in dataset.columns if col != 'target']

    X, y = dataset[features_columns], dataset['target']
    return X, y
