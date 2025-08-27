import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


def mutual_information(file_path):
    file = pd.read_csv(file_path, low_memory=False)

    time = 'time'

    nodes = file.columns[file.columns != time].tolist()

    file_cleaned = file.dropna(subset=nodes)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(file_cleaned[nodes])

    mi_matrix = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                mi_matrix[i, j] = mutual_info_regression(data_scaled[:, i].reshape(-1, 1),
                                                         data_scaled[:, j], random_state=42)[0]
                mi_matrix[i, j] = round(mi_matrix[i, j], 2)
            else:
                mi_matrix[i, j] = 0.0

    return mi_matrix


