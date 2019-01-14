#!usr/bin/python3.6
""" data import and manipulation """
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator
from datetime import timedelta
from datetime import datetime
import multiprocessing
import sys

""" variables """
data_load_done = False
format_data_done = False
raw_data = []
pre_processed_data = []
nn_data_ready = False


def import_raw_data(file_name):
    # File path
    script_dir = os.path.dirname(os.path.abspath(__file__))  # <-- absolute dir the script is in
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
    print("\n creating Y \n")
    # Multiprocessing:
    num_cores = multiprocessing.cpu_count() - 1  # leave one free to not freeze machine
    df_split = np.array_split(df, num_cores)
    pool = multiprocessing.Pool(num_cores)
    # - inputs
    stop_loss, goal, hold, horizon = 0.993, 1.004, 0.002, 24
    params = [df_split, stop_loss, goal, hold, horizon]
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


def traverse(df, stop_loss=0.993, goal=1.004, hold=0.002, horizon=24):
    y = df.values
    y = y[:, 0]
    y = np.c_[y, np.zeros(y.shape[0], dtype=int)]   # create column for long y bool
    y = np.c_[y, np.zeros(y.shape[0], dtype=int)]   # create column for short y bool
    y = np.c_[y, np.zeros(y.shape[0], dtype=int)]   # create column for hold y bool
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
        long_result_found = 0
        short_result_found = 0
        hold_search = 1

        for k in range(i, len(df.index)-1, 1):
            # Check if one day as gone since buy/short
            within_horizon = datetime.strptime(str(date_search), "%Y-%m-%d %H:%M:%S") <= datetime.strptime(
                str(date_time + timedelta(hours=horizon)), "%Y-%m-%d %H:%M:%S")
            date_search = df.iloc[k+1, 0]
            # Check if price reaches target (buy)
            if within_horizon:
                close = df.iloc[k, 1]
                if close / buy >= goal:
                    y[i, 1] = 1
                    long_result_found = 1
                elif close / buy <= stop_loss:
                    y[i, 1] = 0
                    long_result_found = 1
                    
            # Check if price is neutral (hold)
            if within_horizon and hold_search:
                close = df.iloc[k, 1]
                if close / buy >= 1 + hold or close / buy <= 1 - hold:
                    y[i, 3] = 1
                    hold_search = 0
            elif not within_horizon and hold_search:
                y[i, 3] = 1

            # Check if price reaches target (short)
            if within_horizon:
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

            if long_result_found and short_result_found:
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
        if r[3]:
            plt.axvline(x=r[0], color='b', alpha=0.2)
    plt.draw()
    return plt

def data_menu():
    global data_load_done
    global format_data_done
    global raw_data
    global pre_processed_data
    global target

    print("\n\nPrepare data:")
    print("1: import raw data, imported state: " + str(data_load_done))
    print("2: import formatted data, imported state: " + str(format_data_done))
    if data_load_done:
        print("3: Format raw data, format state: " + str(format_data_done))
    if format_data_done:    
        print("4: Plot calculated y")
    print("exit: exit program \n")
    choice = input("select action: ")
    while choice != 'exit':
        if choice == '1':
            filename = input("specify file name: \n")
            raw_data, data_load_done = import_raw_data(filename)
            print("raw data imported \n")
            data_menu()
        elif choice == '2':
            filename = input("specify file name: \n")
            filename = 'full.csv' if filename == '' else filename
            size = input("Specify size of dataset (x100000): \n")
            size = 100000 if size == '' else int(size)*100000
            pre_processed_data, target, format_data_done = import_processed_data(filename, size, INTERVAL)
            data_menu()
        elif choice == '3' and data_load_done:
            pre_processed_data, target, format_data_done = calc_y(raw_data)
            print("Format data completed \n")
            data_menu()
        elif choice == '4' and format_data_done:
            span = input("specify span: \n")
            span = 500 if span == '' else int(span)
            start = input("specify start: \n")
            start = 0 if start == '' else int(start)
            plt = plot_result(pre_processed_data, target, span, start)
            plt.show()
            data_menu()
        else:
            print("invalid input \n")
            data_menu()

    print("Exiting \n")
    exit()


""" Start script """
data_menu()







