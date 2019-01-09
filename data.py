#!usr/bin/python3.6
""" data import and manipulation """
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator
from datetime import timedelta
from datetime import datetime
import multiprocessing
import sys


def import_raw_data(file_name):
    # File path
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    abs_file_path = os.path.join(script_dir + "/data/" + file_name)

    # Select fields
    fields = ['date', 'time', 'close', 'volume']

    # Read file
    print("Opening file and merging date and time column \n \n")
    try:
        df = pd.read_csv(abs_file_path, parse_dates=[['date', 'time']], header=0, usecols=fields)
    except Exception as ex:
        print("Something went wrong when reading df from file, error code: " + str(ex))
        return

    print("Finished opening file \ndata has dimensions: " + str(df.shape) + "\n\n")

    return df, True


def import_processed_data(file_name, size, interval):
    # File path
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    abs_file_path = os.path.join(script_dir + "/NN-data/" + file_name)

    # Select fields
    fields = ['date_time', 'close', 'volume', 'y1', 'y2']

    # Read file
    print("Opening file... \n")
    try:
        df = pd.read_csv(abs_file_path, header=0, usecols=fields)
        df['date_time'] = df['date_time'].astype('datetime64[ns]')  # correct date_time type definition
    except Exception as ex:
        print("Something went wrong when reading df from file, error code: " + str(ex))
        return

    # Select range of dataset
    if size:
        df = df.iloc[(len(df.index)-(size*interval)):(len(df.index)):interval]

    # copy date_time and Y data to targets
    targets = df.iloc[:, [0, 3, 4]].values  # Buy signal target

    # - df normalization
    print("Normalizing X \n")
    df['date_time'] = pd.to_timedelta(df['date_time']).dt.total_seconds().astype(int)  # convert timestamp to float
    df = normalize_data(df)

    df.drop('y1', axis=1, inplace=True)
    df.drop('y2', axis=1, inplace=True)

    print("Finished opening file \nX has dimensions: " + str(df.shape) + ", Y has dimensions: " + str(targets.shape) + "\n")

    return df, targets, True


def calc_y(df):
    # - Y calculation
    print("\n creating Y \n")
    # Multiprocessing:
    num_cores = multiprocessing.cpu_count() - 1  # leave one free to not freeze machine
    df_split = np.array_split(df, num_cores)
    pool = multiprocessing.Pool(num_cores)
    result = pool.map(traverse, df_split)
    y = np.concatenate(result, axis=0)
    pool.close()
    pool.join()

    # save data to csv
    df_save = df
    df_save['y1'] = y[:, 1]
    df_save['y2'] = y[:, 2]
    filename = input("Specify filename for output CSV: \n")
    if filename:
        path = "/home/stian/git/stian9k/NN-data/"
        df.to_csv(path + filename)

    df['date_time'] = pd.to_timedelta(df['date_time']).dt.total_seconds().astype(int)  # convert timestamp to float
    df = normalize_data(df)

    print("Finished formatting data \nX has dimensions: " + str(df.shape) + ", Y has dimensions: " + str(
        y.shape) + "\n")

    return df, y, True


def traverse(df, stop_loss=0.993, goal=1.008):
    y = df.values
    y = y[:, 0]
    y = np.c_[y, np.zeros(y.shape[0], dtype=int)]
    y = np.c_[y, np.zeros(y.shape[0], dtype=int)]
    n = len(df.index)

    for i in range(0, len(df.index)-1, 1):
        # progressbar
        j = (i + 1) / n
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
        sys.stdout.write('\r')
        sys.stdout.flush()

        # initialize variables
        date_time = df.iloc[i, 0]
        buy = df.iloc[i, 1]
        date_search = df.iloc[i + 1, 0]
        buy_result_found = 0
        short_result_found = 0

        for k in range(i, len(df.index)-1, 1):
            # Check if one day as gone since buy/short
            under_one_day = datetime.strptime(str(date_search), "%Y-%m-%d %H:%M:%S") <= datetime.strptime(
                str(date_time + timedelta(hours=24)), "%Y-%m-%d %H:%M:%S")
            date_search = df.iloc[k+1, 0]
            # Check buy trade
            if under_one_day:
                close = df.iloc[k, 1]
                if close / buy >= goal:
                    y[i, 1] = 1
                    buy_result_found = 1
                elif close / buy <= stop_loss:
                    y[i, 1] = 0
                    buy_result_found = 1
            else:
                y[i, 1] = 0
                buy_result_found = 1

            # Check short trade
            if under_one_day:
                close = df.iloc[k, 1]
                if close / buy <= 1 / goal:
                    y[i, 2] = 1
                    short_result_found = 1
                elif close / buy >= stop_loss:
                    y[i, 2] = 0
                    short_result_found = 1
            else:
                y[i, 2] = 0
                short_result_found = 1

            if buy_result_found and short_result_found:
                break
    print("done \n")
    return y


def normalize_data(df):
    # columns to normalize
    cols_to_norm = ['date_time', 'close', 'volume']
    df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df


def plot_result(df, targets, span=1000, start=0):
    # Plotting:
    fig, ax = plt.subplots()
    x = targets[start:(start + span), 0]
    y1 = df.iloc[start:(start + span), 1]
    ax.plot(x, y1)
    plt.xticks(rotation=80)  # rotate x ticks from horizontal
    plt.tight_layout()  # fit everything into window
    plt.grid(b=True, which='major', color='k', linestyle='--')  # Set grid

    ax.fmt_xdata = DateFormatter('%Y-%m-%d %H:%M:%S')   # date_time format
    ax.xaxis.set_major_locator(MaxNLocator(20))     # number of x-axis ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S"))    # x-axis ticks visual format

    for r in targets[start:(start+span)]:
        if r[1]:
            plt.axvline(x=r[0], color='g', alpha=0.2)
        if r[2]:
            plt.axvline(x=r[0], color='r', alpha=0.2)
    plt.draw()
    return plt







