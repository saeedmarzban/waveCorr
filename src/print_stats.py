import boto3
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
plt.rcParams.update({'font.size': 15})
import numpy as np
import datetime as dt
import shutil

path = 'C:\SAEED\HEC\Thesis\waveCorr\waveCorr\src/results/all_results/'

# Copy all files in one folder
################################
# lst_sub = [x[0] for x in os.walk(path)]
# lst_sub = lst_sub[1:]
#
# for folder_name in lst_sub:
#     if folder_name.split('/')[-1] == 'all_results':
#         continue
#     else:
#         for file_name in os.listdir(folder_name):
#             if str(file_name).endswith('.csv'):
#                 copy_from = folder_name + '/' + file_name
#                 copy_to = path + 'all_results/' + folder_name.split('/')[-1] + '_' + file_name
#                 shutil.copy(copy_from, copy_to)



# cats_labels = ['Cost-sensitive']
cats_labels = ['WaveCorr','CS-LSTM-CNN','CS-CNN','EIIE']
data_set = 'covid'
# cats_labels = ['eiie']
# cats_labels = ['30 stocks','40 stocks','50 stocks']
# cats = [1111]
# cats = [5003]
cats = [data_set + '_waveCorr',data_set + '_cs_LSTM_CNN',data_set + '_cs_CNN',data_set + '_eiie']
# cats = [5003,5004,5005]
# cats = [5003,5004,5008]
# cats = [2001,2002,2000]

# no short
# cats_labels = ['WaveCorr-direct','WaveCorr-PVM','Cost-sensitive-direct','Cost-sensitive-PVM']
# cats = [5007,5003,5008,5004]

# with short
# cats_labels = ['WaveCorr-direct','WaveCorr-PVM','Cost-sensitive-direct','Cost-sensitive-PVM']
# cats = [5005,5000,5006,5001]

# cats_labels = ['Off-policy','On-policy']
# cats = [4999,5003]

f1 = pd.DataFrame(columns=['name'])
w1 = pd.DataFrame(columns=['name'])
curve1 = pd.DataFrame(columns=['name'])


mean_std_graph = 1

if mean_std_graph == 1:

    # Load the file names
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and 'APV_' in i and str(i).endswith('.csv'):
            f1 = f1.append({'name': i},ignore_index=True)

    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and 'weightsMatrix' in i and str(i).endswith('.csv'):
            w1 = w1.append({'name': i},ignore_index=True)

    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and 'track_test_train' in i and str(i).endswith('.csv'):
            curve1 = curve1.append({'name': i},ignore_index=True)


    figsize_x = 10
    figsize_y = 5
    plt.figure(figsize=(figsize_x, figsize_y), dpi=100)


    #Plot equally weighted
    f1_cat = np.array(f1)[0,0]
    apv_df = pd.read_csv(path + f1_cat)
    apv_ar = np.array(apv_df.iloc[:, 1:])
    dates_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in apv_df.iloc[:, 0]]
    dates = matplotlib.dates.date2num(dates_list)

    counter = 0
    for cat_value in cats:
        f1_cat = np.array(f1[f1['name'].str.contains(str(cat_value))])[:,0]

        apv_all = []
        for iii in range(len(f1_cat)):
            #Load apv data
            apv_df = pd.read_csv(path + f1_cat[iii])
            apv_ar = np.expand_dims(np.array(apv_df.iloc[:, 1:]),0)

            if apv_all == []:
                apv_all = apv_ar
            else:
                apv_all = np.append(apv_all, apv_ar, 0)

        apv_mean = np.mean(apv_all,0)
        apv_min = np.min(apv_all, 0)
        apv_max = np.max(apv_all, 0)

        # Plot the APV
        dates_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in apv_df.iloc[:, 0]]
        dates = matplotlib.dates.date2num(dates_list)
        # plt.plot_date(dates, data_np1[:, 0], label='Experiment ' + str(i), linestyle='solid', marker='None')
        plt.plot_date(dates, apv_mean[:, 0], label=str(cats_labels[counter]) , linestyle='solid', marker='None')
        plt.fill_between(dates, apv_min[:, 0], apv_max[:, 0], alpha=.1)
        counter = counter + 1

    plt.plot_date(dates, apv_ar[0, :, 1], label='EW', linestyle='solid', marker='None')

    date_form = DateFormatter("%y-%m")
    axes = plt.gca()
    # axes.xaxis.set_major_formatter(date_form)
    # axes.set_xlim([xmin, xmax])
    # axes.set_ylim([0, 4])
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Portflio value')
    plt.savefig(path + data_set + 'ReturnsComparison.png')
    plt.close()



    figsize_x = 10
    figsize_y = 5
    plt.figure(figsize=(figsize_x, figsize_y), dpi=100)
    counter = 0
    for cat_value in cats:
        curve1_cat = np.array(curve1[curve1['name'].str.contains(str(cat_value))])[:,0]
        curve_all = []
        for iii in range(len(curve1_cat)):
            # Load test data
            curve_df = pd.read_csv(path + curve1_cat[iii])
            curve_ar = np.expand_dims(np.array(curve_df.iloc[:, 1:]), 0)

            if curve_all == []:
                curve_all = curve_ar
            else:
                curve_all = np.append(curve_all, curve_ar[:,0:curve_all.shape[1],:], 0)

        curve_mean = np.mean(curve_all,0)
        curve_min = np.min(curve_all, 0)
        curve_max = np.max(curve_all, 0)

        # Plot the training Curve
        plt.plot(np.arange(len(curve_mean[:, 0])), curve_mean[:, 0], label=str(cats_labels[counter]) , linestyle='solid', marker='None')
        plt.fill_between(np.arange(len(curve_mean[:, 0])), curve_min[:, 0], curve_max[:, 0], alpha=.1)
        counter = counter + 1

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Out-of-sample learning curves')
    plt.savefig(path + data_set + 'Out_of_sample.png')
    plt.close()




    for cat_value in cats:
        f1_cat = np.array(f1[f1['name'].str.contains(str(cat_value))])[:, 0]
        w1_cat = np.array(w1[w1['name'].str.contains(str(cat_value))])[:, 0]
        statistics_lst = np.zeros([len(f1_cat),13])
        curve1_cat = np.array(curve1[curve1['name'].str.contains(str(cat_value))])[:, 0]
        curve_all = []
        for iii in range(len(curve1_cat)):
            # Load test data
            curve_df = pd.read_csv(path + curve1_cat[iii])
            curve_ar = np.expand_dims(np.array(curve_df.iloc[:, 0:]), 0)

            if curve_all == []:
                curve_all = curve_ar
            else:
                curve_all = np.append(curve_all, curve_ar[:, 0:curve_all.shape[1], :], 0)


        for iii in range(len(f1_cat)):
            apv = pd.read_csv(path + f1_cat[iii])
            data_apv = np.array(apv.iloc[:, 1:])

            weights = pd.read_csv(path + w1_cat[iii])
            data_weights = np.array(weights.iloc[:, 1:])

            annual_return = data_apv[-1, 0] ** (252 / len(data_apv)) - 1
            annual_return_EW = data_apv[-1, 1] ** (252 / len(data_apv)) - 1
            annual_vol = np.std(np.diff(np.log(data_apv[:, 0].tolist()), n=1, axis=0)) * np.sqrt(252)
            annual_vol_EW = np.std(np.diff(np.log(data_apv[:, 1].tolist()), n=1, axis=0)) * np.sqrt(252)
            reward_risk = annual_return / annual_vol
            reward_risk_EW = annual_return_EW / annual_vol_EW
            dd = [data_apv[i, 0] / max(data_apv[0:i, 0]) - 1 for i in range(1, len(data_apv))]
            max_drawdown = min(dd)
            dd_EW = [data_apv[i, 1] / max(data_apv[0:i, 1]) - 1 for i in range(1, len(data_apv))]
            max_drawdown_EW = min(dd_EW)
            lst_ret = [data_apv[i, 0] / data_apv[i - 1, 0] - 1 for i in range(1, len(data_apv))]
            lst_ret_EW = [data_apv[i, 1] / data_apv[i - 1, 1] - 1 for i in range(1, len(data_apv))]
            beta_vs_benchmark = np.cov(lst_ret, lst_ret_EW)[0, 1] / np.var(lst_ret_EW)
            daily_hit_rate = sum([lst_ret[i] > lst_ret_EW[i] for i in range(len(lst_ret))]) / len(lst_ret)
            trackingError = np.std(np.diff(np.log(data_apv[:, 0].tolist()), n=1, axis=0)
                                        - np.diff(np.log(data_apv[:, 1].tolist()), n=1, axis=0)) * np.sqrt(252)
            portfolio_turnover = np.sum(np.linalg.norm(np.subtract(data_weights[1:, 1:], data_weights[0:-1, 1:]), ord=1, axis=1)) / (
                                                  2 * (len(data_weights) - 1))
            generalization_error = 1 - curve_all[iii,-1,1]/curve_all[iii,-1,0]

            statistics_lst[iii,:] = [annual_return,annual_return_EW,annual_vol,annual_vol_EW,reward_risk,reward_risk_EW,max_drawdown,max_drawdown_EW,beta_vs_benchmark,daily_hit_rate,trackingError,portfolio_turnover,generalization_error]

        statistics_mean = np.round(np.mean(statistics_lst,0),2)
        statistics_std = np.round(np.std(statistics_lst, 0),2)
        print('Results for category: '+str(cat_value))
        print('--------------------------------------------------')
        res_list = ['Annual return   :' + str(np.round(statistics_mean[0]*100))+'\%' + ' (' + str(np.round(statistics_std[0]*100))+'\%' + ')']
        res_list.append('Annual vol      :' + str(np.round(statistics_mean[2]*100))+'\%' + ' (' + str(np.round(statistics_std[2]*100))+'\%' + ')')
        res_list.append('SR     :' + str(statistics_mean[4]) + ' (' + str(statistics_std[4]) + ')')
        res_list.append('MDD    :' + str(-np.round(statistics_mean[6]*100))+'\%' + ' (' + str(np.round(statistics_std[6]*100))+'\%'+ ')')
        # res_list.append('Beta vs. Bench  :' + str(statistics_mean[8]) + ' : ' + str(statistics_std[8]))
        res_list.append('Daily hit rate  :' + str(np.round(statistics_mean[9]*100))+'\%' + ' (' + str(np.round(statistics_std[9]*100))+'\%' + ')')
        # res_list.append('Tracking error  :' + str(statistics_mean[10]) + ' : ' + str(statistics_std[10]))
        res_list.append('Turnover        :' + str(statistics_mean[11]) + ' (' + str(statistics_std[11]) + ')')
        res_list.append('GE        :' + str(statistics_mean[12]) + ' (' + str(statistics_std[12]) + ')')

        res_list.append('Annual return EW:' + str(np.round(statistics_mean[1]*100))+'\%' + ' (0\%)')
        res_list.append('Annual vol EW   :' + str(np.round(statistics_mean[3]*100))+'\%' + ' (0\%)')
        res_list.append('SR EW  :' + str(statistics_mean[5]) + ' (0)')
        res_list.append('MDD EW :' + str(-np.round(statistics_mean[7]*100))+'\%' + ' (0\%)')

        print(*res_list, sep="\n")
        print('--------------------------------------------------')

if mean_std_graph == 0:

    # Load the file names
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'APV_' in i and str(i).endswith('.csv'):
            f1 = f1.append({'name': i}, ignore_index=True)

    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'weightsMatrix' in i and str(i).endswith('.csv'):
            w1 = w1.append({'name': i}, ignore_index=True)

    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'track_test_train' in i and str(i).endswith('.csv'):
            curve1 = curve1.append({'name': i}, ignore_index=True)

    figsize_x = 10
    figsize_y = 5
    plt.figure(figsize=(figsize_x, figsize_y), dpi=100)

    counter = 1
    for cat_value in cats:
        f1_cat = np.array(f1[f1['name'].str.contains(str(cat_value))])[:, 0]

        apv_all = []
        for iii in range(len(f1_cat)):
            # Load apv data
            apv_df = pd.read_csv(path + f1_cat[iii])
            apv_ar = np.array(apv_df.iloc[:, 2:])
            # Plot the APV
            dates_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in apv_df.iloc[:, 1]]
            dates = matplotlib.dates.date2num(dates_list)
            plt.plot_date(dates, apv_ar[:, 0], label='Experiment ' + str(counter), linestyle='solid', marker='None')
            # plt.plot_date(dates, apv_mean[:, 0], label=str(cats_labels[counter]), linestyle='solid', marker='None')
            # plt.fill_between(dates, apv_min[:, 0], apv_max[:, 0], alpha=.1)
            counter = counter + 1

    axes = plt.gca()
    # axes.set_xlim([xmin, xmax])
    axes.set_ylim([0, 4])
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Portflio value')
    plt.savefig(path + data_set + 'ReturnsComparison.png')
    plt.close()




