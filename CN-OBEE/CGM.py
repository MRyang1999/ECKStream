import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import agm


def correlated_data(sampled_data, mean_value):
    for col in sampled_data.columns[1:]:
        C = mean_value[col]
        values = sampled_data[col].values.tolist()
        for i in range(1, len(values)):
            if values[i] - values[i - 1] > C:
                values[i] = values[i - 1] + C
            elif values[i] - values[i - 1] < -C:
                values[i] = values[i - 1] - C
            else:
                values[i] = values[i]

        sampled_data.loc[sampled_data.index, col] = values

    return sampled_data


def cgm(sampled_data, epsilon, mean_value):
    sampled_data = correlated_data(sampled_data, mean_value)
    for col in sampled_data.columns[1:]:
        values = sampled_data[col].values.tolist()
        max_value = max(map(float, values))
        min_value = min(map(float, values))
        a = max_value - min_value
        max_dif = np.max(np.abs(np.diff(values)))
        min_dif = np.min(np.diff(values))
        b = max_dif - min_dif
        sigma = agm.AnalyticGaussian(epsilon, 1.e-5, a)
        if sigma <= 0:
            sigma = 0.1
        noise_values = []
        v_i = 1
        noise_i = np.random.normal(loc=0, scale=sigma)
        per_value = values[0] + noise_i
        noise_values.append(per_value)
        for i in range(1, len(values)):
            r_i = (1 - b) / ((1 - b) ** 2 + v_i)
            sigma_i = float(((1 - r_i) * a + r_i * b) * sigma)
            noise_i = np.random.normal(loc=0, scale=sigma_i) + r_i * noise_i
            per_value = values[i] + noise_i
            v_i = v_i / ((1 - b) ** 2 + v_i)
            noise_values.append(per_value)

        sampled_data.loc[sampled_data.index, col] = noise_values
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
    min_epsilon, max_epsilon, step_epsilon = 0.4, 5.0, 0.5
    file = pd.read_csv('Power.csv', low_memory=False)
    mean_value = {}

    for col in file.columns[1:]:
        mean_value[col] = file[col].mean()

    with open("results_3.txt", "a") as f:
        f.write("Epsilon\tMAE\tRMSE\tNMAE\n")
        f.flush()

        for ep in np.arange(max_epsilon, min_epsilon, -step_epsilon):
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

                cgm_data = cgm(df.copy(), ep, mean_value)
                for node in nodes:
                    orig = df[node].values
                    perturbed = cgm_data[node].values
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
                    noise_mean_value[node] = cgm_data[node].mean()
                    noise_std_value[node] = cgm_data[node].std()

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
