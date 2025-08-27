import numpy as np
import pandas as pd
import common_tools
import sensitivity_calc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def comporder(ex, domain_low, domain_high, eps, delay_time, flag=0, interval_=10, num_=100):
    total_time = len(ex)
    dim = len(ex[0])
    eps_post = eps / 4
    eps_pub = eps - eps_post
    eps_1 = eps_post / 2
    eps_2 = eps_post / 2

    published_result = []
    flag_ = []

    whether_update = 0
    sensitivity_ = domain_high

    rho_ = common_tools.add_noise(sensitivity_, eps_1 / 2, dim)

    for i in range(total_time):
        if whether_update == 0:
            eps_pub = eps - eps_post

        if i % num_ == 0:
            if (i / num_) % interval_ == 0 and flag == 1 and (i + 1) * num_ <= total_time:
                eps_s = eps_pub / 2
                eps_pub = eps_pub - eps_s

                data_sens = np.zeros(num_, dtype=int)
                cc = 0
                for qq in range(i * num_, (i + 1) * num_):
                    data_sens[cc] = ex[qq][0]
                    cc += 1

                sensitivity_ = sensitivity_calc.quality_func(data_sens, domain_low, domain_high,
                                                             interval_,
                                                             eps_s)
                whether_update = 1
            else:
                whether_update = 0

        if ex[i][0] > sensitivity_:
            noise_result = sensitivity_ + common_tools.add_noise(sensitivity_, eps_pub, dim)
        else:
            noise_result = ex[i][0] + common_tools.add_noise(sensitivity_, eps_pub, dim)

        temp = []
        if i + delay_time < total_time:
            for j in range(i + 1, i + delay_time):
                if ex[i][0] - ex[j][0] + common_tools.add_noise(sensitivity_,
                                                                eps_2 / (2 * (2 * delay_time - 1)),
                                                                dim) > rho_:
                    # if ex[i][0] - ex[j][0] > 0:
                    temp.append(0)
                else:
                    temp.append(1)
        elif i + 1 < total_time:
            for j in range(i + 1, total_time):
                if ex[i][0] - ex[j][0] + common_tools.add_noise(sensitivity_,
                                                                eps_2 / (2 * (2 * delay_time - 1)),
                                                                dim) > rho_:
                    # if ex[i][0] - ex[j][0] > 0:
                    temp.append(0)
                else:
                    temp.append(1)
        else:
            temp = []

        flag_.append(temp)

        low_ = 0
        high_ = 1000000000
        if i > delay_time - 1:
            for j in range(i - delay_time + 1, i - 1):
                if flag_[j][i - j - 1] == 0:
                    if published_result[j] > low_:
                        low_ = published_result[j]
                else:
                    if published_result[j] < high_:
                        high_ = published_result[j]
        else:
            for j in range(0, i - 1):
                if flag_[j][i - j - 1] == 0:
                    if published_result[j] > low_:
                        low_ = published_result[j]
                else:
                    if published_result[j] < high_:
                        high_ = published_result[j]

        if noise_result > low_ or noise_result < high_:
            if low_ > high_:
                noise_result = (low_ + high_) / 2

        published_result.append(noise_result)

    return published_result


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


if __name__ == "__main__":
    min_epsilon, max_epsilon, step_epsilon = 0.4, 5.0, 0.5

    with open("results_6.txt", "a") as f:
        f.write("Epsilon\tMAE\tRMSE\tNMAE\n")
        f.flush()

        for ep in np.arange(max_epsilon, min_epsilon, -step_epsilon):
            A_1 = 0
            A_2 = 0
            A_3 = 0
            A_4 = 0
            RM_1 = 0
            RM_2 = 0
            RM_3 = 0
            RM_4 = 0
            NA_1 = 0
            NA_2 = 0
            NA_3 = 0
            NA_4 = 0
            round_ = 20
            all_mean_distances = []
            all_mae_error = []
            for i in range(round_):

                filename = "Power.csv"
                df = pd.read_csv(filename, low_memory=False)

                original_data = {}
                noise_data = {}
                mean_value = {}
                std_value = {}
                noise_mean_value = {}
                noise_std_value = {}
                euclidean_distance = {}
                mae_error = {}
                nodes = df.columns[1:]

                for col in df.columns[1:]:
                    ex = df[[col]].astype(float).values.tolist()

                    data = np.zeros(len(ex), dtype=float)

                    for j in range(len(ex)):
                        data[j] = ex[j][0]

                    delay_time = 10
                    sensitivity_ = max(data)

                    domain_low = min(data)
                    domain_high = max(data)

                    original_data[col] = df[col].astype(float).values.tolist()
                    noise_data[col] = comporder(ex, domain_low, domain_high, ep, delay_time, flag=0, interval_=5,
                                                num_=100)

                    orig = np.array(original_data[col])
                    perturbed = np.array(noise_data[col])
                    diff = orig - perturbed
                    mask = ~np.isnan(diff)
                    vaild_count = np.sum(mask)
                    abs_error = np.abs(orig - perturbed)
                    mae_error[col] = np.sum(abs_error[mask]) / vaild_count
                    euclidean_distance[col] = np.sqrt(np.sum((diff[mask]) ** 2))

                    mean_value[col] = np.nanmean(original_data[col])
                    std_value[col] = np.nanstd(original_data[col])
                    noise_mean_value[col] = np.nanmean(noise_data[col])
                    noise_std_value[col] = np.nanstd(noise_data[col])

                mean_mae = sum(mae_error.values()) / len(nodes)
                all_mae_error.append(mean_mae)
                mean_distance = sum(euclidean_distance.values()) / len(euclidean_distance)
                all_mean_distances.append(mean_distance)

                M_3 = MSE(mean_value, noise_mean_value)
                M_4 = MSE(std_value, noise_std_value)

                a_3 = MAE(mean_value, noise_mean_value)
                a_4 = MAE(std_value, noise_std_value)

                A_3 += a_3
                A_4 += a_4

                RM_3 += RMSE(M_3)
                RM_4 += RMSE(M_4)

                NA_3 += NMAE(mean_value, a_3)
                NA_4 += NMAE(std_value, a_4)

            avg_mean_distance = np.mean(all_mean_distances)
            avg_mae_error = np.mean(all_mae_error)
            print(avg_mean_distance)
            print(avg_mae_error)

            MAE_3 = A_3 / round_
            MAE_4 = A_4 / round_

            RMSE_3 = RM_3 / round_
            RMSE_4 = RM_4 / round_

            NMAE_3 = NA_3 / round_
            NMAE_4 = NA_4 / round_

            print("MAE", MAE_3, ep)
            print("MAE", MAE_4, ep)

            print("RMSE", RMSE_3, ep)
            print("RMSE", RMSE_4, ep)

            print("NMAE", NMAE_3, ep)
            print("NMAE", NMAE_4, ep)

            f.write("mean_distance"f"{ep}\t{avg_mean_distance}\n")
            f.write("mean_mae"f"{ep}\t{avg_mae_error}\n")
            f.write("mean_value"f"{ep}\t{MAE_3}\t{RMSE_3}\t{NMAE_3}\n")
            f.write("std_value"f"{ep}\t{MAE_4}\t{RMSE_4}\t{NMAE_4}\n")

            f.flush()

