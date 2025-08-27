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


def roll_window_perturb(sampled_data, epsilon, w, mean_value):  # 利用固定的滑动窗口，来对数据进行加噪，其中利用Wiener 滤波器
    sampled_data = correlated_data(sampled_data, mean_value)
    df_sample = sampled_data
    for col in df_sample.columns[1:]:
        window_size = w
        values = df_sample[col].values.tolist()

        num_windows = len(values) // window_size
        for i in range(num_windows):
            clip = []
            noise_data = []
            window_data = values[i * window_size:(i + 1) * window_size]
            max_value = max(window_data)
            min_value = min(window_data)
            for k in range(1, window_size):
                dif = abs(window_data[k] - window_data[k - 1])
                clip.append(dif)
            dif_max = max(clip)
            dif_min = min(clip)
            a = max_value - min_value
            b = dif_max - dif_min
            sigma = agm.AnalyticGaussian(epsilon, 1.e-5, a)
            v_j = 1
            noise_j = np.random.normal(loc=0, scale=sigma)
            noise_value = window_data[0] + noise_j
            noise_data.append(noise_value)

            for j in range(1, window_size):
                r_j = (1 - b) / ((1 - b) ** 2 + v_j)
                sigma_j = float(((1 - r_j) * a + r_j * b) * sigma)
                noise_j = np.random.normal(loc=0, scale=sigma_j) + r_j * noise_j
                noise_value = window_data[j] + noise_j
                noise_data.append(noise_value)
                v_j = v_j / ((1 - b) ** 2 + v_j)

            values[i * window_size:(i + 1) * window_size] = noise_data

        if len(values) % window_size != 0:
            clip = []
            noise_data = []
            window_data = values[num_windows * window_size:]
            max_value = max(window_data)
            min_value = min(window_data)
            if len(window_data) == 1:
                diff = window_data[0]
                clip.append(diff)
            else:
                for k in range(1, len(window_data)):
                    dif = abs(window_data[k] - window_data[k - 1])
                    clip.append(dif)
            dif_max = max(clip)
            dif_min = min(clip)
            a = max_value - min_value
            b = dif_max - dif_min
            sigma = agm.AnalyticGaussian(epsilon, 1.e-5, a)
            v_j = 1
            noise_j = np.random.normal(loc=0, scale=sigma)
            noise_value = window_data[0] + noise_j
            noise_data.append(noise_value)

            for j in range(1, len(window_data)):
                r_j = (1 - b) / ((1 - b) ** 2 + v_j)
                sigma_j = float(((1 - r_j) * a + r_j * b) * sigma)
                noise_j = np.random.normal(loc=0, scale=sigma_j) + r_j * noise_j
                noise_value = window_data[j] + noise_j
                noise_data.append(noise_value)
                v_j = v_j / ((1 - b) ** 2 + v_j)

            values[num_windows * window_size:] = noise_data

        df_sample.loc[df_sample.index, col] = values
    return df_sample


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


def num(df, mean_value):
    window = 0
    window_num = []
    all_data = list(df)
    data_size = len(all_data)

    for k in range(1, data_size):
        value = all_data[k]
        pre_value = all_data[k - 1]

        diff = abs(value - pre_value)

        if diff < mean_value:
            window += 1
        else:
            window_num.append(window)
            window = 0
    return window_num


def main():
    min_epsilon, max_epsilon, step_epsilon = 0.5, 5.5, 0.5
    file = pd.read_csv('Power.csv', low_memory=False)

    mean_value = {}
    for col in file.columns[1:]:
        mean_value[col] = file[col].mean()

    with open("results_4.txt", "a") as f:
        f.write("Epsilon\tMAE\tRMSE\tNMAE\n")
        f.flush()

        for ep in np.arange(min_epsilon, max_epsilon, step_epsilon):
            num_runs = 10
            all_mean_distances = []
            all_A_3 = []
            all_A_4 = []
            all_RM_3 = []
            all_RM_4 = []
            all_NA_3 = []
            all_NA_4 = []
            all_mae_error = []
            w = 10

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

                roll_data = roll_window_perturb(df.copy(), ep, w, mean_value)
                for node in nodes:
                    orig = df[node].values
                    perturbed = roll_data[node].values
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
                    noise_mean_value[node] = roll_data[node].mean()
                    noise_std_value[node] = roll_data[node].std()

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
