import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

class OurSMOTE():
    def __init__(self) -> None:
        pass

    def fit_resample(self, X, y):
        pos_label, neg_label = (0, 1) if 0 in y else ('positive', 'negative')

        X_resampled = []
        y_resampled = []
        pos_samples = [p[0] for p in zip(X, y) if p[1] == pos_label]
        neg_samples = [p[0] for p in zip(X, y) if p[1] == neg_label]
        neigh = NearestNeighbors(n_neighbors=5)
        neigh.fit(X)

        for i in range(len(neg_samples)-len(pos_samples)):
            sample_1 = random.choice(pos_samples)
            _, indices = neigh.kneighbors([sample_1], 5)
            index = random.choice(indices[0])
            sample_2 = X[index]
            new_sample = (sample_1 + sample_2) / 2

            X_resampled.append(new_sample)
            y_resampled.append(pos_label)

        X_resampled = np.concatenate((X, X_resampled))
        y_resampled = np.concatenate((y, y_resampled))

        return X_resampled, y_resampled
