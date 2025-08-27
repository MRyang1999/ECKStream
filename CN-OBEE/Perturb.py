import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def direct_perturb(sampled_data, epsilon, mean_value):  # 直接扰动
    for col in sampled_data.columns[1:]:
        values = sampled_data[col].values.tolist()
        noise_data = [0] * len(values)
        max_value = max(map(float, values))
        min_value = min(map(float, values))
        a = max_value - min_value
        max_dif = np.max(np.diff(values))
        min_dif = np.min(np.diff(values))
        b = max_dif - min_dif
        if a <= 2 * b:
            scale = a / epsilon
            epsilon_1 = epsilon
            epsilon_2 = 0
        else:
            scale = a / (epsilon / 2)
            epsilon_1 = 0
            epsilon_2 = epsilon / 2
        noise_j = np.random.laplace(0, scale)
        noise_data[0] = values[0] + noise_j
        for j in range(1, len(values)):
            if a <= 2 * b:
                scale = a / epsilon_1
                noise_data[j] = values[j] + np.random.laplace(0, scale)
            else:
                scale = b / epsilon_2
                noise_data[j] = (values[j] - values[j - 1]) + np.random.laplace(0, scale) + noise_data[j - 1]

        sampled_data.loc[sampled_data.index, col] = noise_data
    return sampled_data


def MSE(df1, df2):
    real_values = list(df1.values())
    noise_values = [df2.get(key) for key in df1.keys()]

    mse = mean_squared_error(real_values, noise_values)

    return mse


def MAE(df1, df2):
    real_value = list(df1.values())
    noise_value = [df2.get(key) for key in df1.keys()]

    mae = mean_absolute_error(real_value, noise_value)

    return mae


def RMSE(M):
    return np.sqrt(M)


def NMAE(df1, A):
    real_value = list(df1.values())
    dif_value = np.max(real_value) - np.min(real_value)

    nmae = A / dif_value

    return nmae


def main():
    min_epsilon, max_epsilon, step_epsilon = 0.5, 5.5, 0.5
    file = pd.read_csv('Power.csv')

    mean_value = {}
    for node in file.columns[1:]:
        mean_value[node] = file[node].mean()

    with open("results_1.txt", "a") as f:
        f.write("Epsilon\tMAE\tRMSE\tNMAE\n")
        f.flush()

        for ep in np.arange(min_epsilon, max_epsilon, step_epsilon):
            num_runs = 20
            all_mean_distances = []
            all_A_3 = []
            all_A_4 = []
            all_RM_3 = []
            all_RM_4 = []
            all_NA_3 = []
            all_NA_4 = []
            all_mae_error = []

            for run in range(num_runs):
                std_value = {}
                noise_mean_value = {}
                noise_std_value = {}
                euclidean_distance = {}
                mae_error = {}
                df = pd.read_csv('Power.csv', low_memory=False)
                for node in df.columns[1:]:
                    std_value[node] = df[node].std()
                nodes = df.columns[1:]

                direct_data = direct_perturb(df.copy(), ep, mean_value)
                for node in nodes:
                    orig = df[node].values
                    perturbed = direct_data[node].values
                    diff = orig - perturbed
                    mask = ~np.isnan(diff)
                    vaild_count = np.sum(mask)
                    abs_error = np.abs(orig - perturbed)
                    mae_error[node] = np.sum(abs_error[mask]) / vaild_count
                    euclidean_distance[node] = np.sqrt(np.sum((diff[mask]) ** 2))

                mean_mae = sum(mae_error.values()) / len(nodes)
                all_mae_error.append(mean_mae)
                mean_distance = sum(euclidean_distance.values()) / len(euclidean_distance)
                all_mean_distances.append(mean_distance)

                for node in df.columns[1:]:
                    noise_mean_value[node] = direct_data[node].mean()
                    noise_std_value[node] = direct_data[node].std()

                M_3 = MSE(mean_value, noise_mean_value)
                M_4 = MSE(std_value, noise_std_value)

                A_3 = MAE(mean_value, noise_mean_value)
                A_4 = MAE(std_value, noise_std_value)

                RM_3 = RMSE(M_3)
                RM_4 = RMSE(M_4)

                NA_3 = NMAE(mean_value, A_3)
                NA_4 = NMAE(std_value, A_4)

                all_A_3.append(A_3)
                all_A_4.append(A_4)
                all_RM_3.append(RM_3)
                all_RM_4.append(RM_4)
                all_NA_3.append(NA_3)
                all_NA_4.append(NA_4)

            avg_mean_distance = np.mean(all_mean_distances)
            avg_mae_error = np.mean(all_mae_error)
            print(avg_mean_distance)
            print(avg_mae_error)

            avg_A_3 = np.mean(all_A_3)
            avg_A_4 = np.mean(all_A_4)

            avg_RM_3 = np.mean(all_RM_3)
            avg_RM_4 = np.mean(all_RM_4)

            avg_NA_3 = np.mean(all_NA_3)
            avg_NA_4 = np.mean(all_NA_4)

            print("MAE", avg_A_3, ep)
            print("MAE", avg_A_4, ep)

            print("RMSE", avg_RM_3, ep)
            print("RMSE", avg_RM_4, ep)

            print("NMAE", avg_NA_3, ep)
            print("NMAE", avg_NA_4, ep)

            f.write("mean_distance"f"{ep}\t{avg_mean_distance}\n")
            f.write("mean_mae"f"{ep}\t{avg_mae_error}\n")
            f.write("mean_value"f"{ep}\t{avg_A_3}\t{avg_RM_3}\t{avg_NA_3}\n")
            f.write("std_value"f"{ep}\t{avg_A_4}\t{avg_RM_4}\t{avg_NA_4}\n")

            f.flush()


main()
