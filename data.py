#!/usr/bin/python3.6
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
from functools import partial
import sys

""" variables """
data_load_done = False
format_data_done = False
raw_data = []
pre_processed_data = []
nn_data_ready = False
INTERVAL = 1


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
    print("raw data imported")
    print("-----------------------------------------------------------\n")

    return df, True


def import_processed_data(filename, size, interval, file=None):
    if file is None:
        # File path
        script_dir = os.path.dirname(os.path.abspath(__file__))  # <-- absolute dir the script is in
        file = os.path.join(script_dir + "/NN-data/" + filename)

    # Read file
    print("Opening file... \n")
    try:
        df = pd.read_csv(file, header=0, index_col=False)
        df['date_time'] = df['date_time'].astype('datetime64[ns]')  # correct date_time type definition
        df.rename(columns={'Unnamed: 0':'ix'}, inplace=True)
    except Exception as ex:
        print("Something went wrong when reading df from file, error code: " + str(ex))
        return

    # Select range of dataset
    if size:
        df = df.iloc[(len(df.index)-(size*interval)):(len(df.index)):interval]

    # copy Y data to targets
    print("Finished opening file, data has dimensions: " + str(df.shape) + "\n" + str(df.keys()) + "\n")
    for key in df.keys():
        if not key == 'date_time':
            df[key] = pd.to_numeric(df[key], downcast='float')
    targets = df.iloc[:, [0, 2, 4, 5, 6]].values  # index, close, buy, short, hold

    # - df normalization
    print("Normalizing X \n")
    df = normalize_data(df)

    df.drop('y1', axis=1, inplace=True)
    df.drop('y2', axis=1, inplace=True)
    df.drop('y3', axis=1, inplace=True)

    print("Finished opening file \nX has dimensions: " + str(df.shape) + ", Y has dimensions: " + str(targets.shape))
    print("-----------------------------------------------------------\n")

    return df, targets, True


def calc_y(df):
    print("\n creating Y \n")
    # - inputs
    stop_loss = input("enter stop loss (default is 0.993): ")
    stop_loss = 0.993 if stop_loss == '' else float(stop_loss)
    goal = input("enter target goal (default is 1.003): ")
    goal = 1.003 if goal == '' else float(goal)
    hold = input("enter neutral variance (default is 0.002): ")
    hold = 0.002 if hold == '' else float(hold)
    horizon = input("enter prediction horizon (default is one 1 hour): ")
    horizon = 1 if horizon == '' else int(horizon)
    params = [stop_loss, goal, hold, horizon]
    print("\n")

    # Multiprocessing:
    num_cores = multiprocessing.cpu_count() - 1  # leave one free to not freeze machine
    df_split = np.array_split(df, num_cores)
    pool = multiprocessing.Pool(num_cores)
    func = partial(traverse, params)
    result = pool.map(func, df_split)
    y = np.concatenate(result, axis=0)
    pool.close()
    pool.join()

    # ad ix column to y
    ix = np.arange(0, len(y), 1)
    ix.reshape(len(ix), 1)
    y = np.c_[ix, y]

    # remove date_time from array
    y = y[:,[0, 2, 3, 4, 5]]

    # save data to csv
    df_save = df
    # df_save['close'] = y[:, 2]
    df_save['y1'] = y[:, 2]
    df_save['y2'] = y[:, 3]
    df_save['y3'] = y[:, 4]
    filename = input("Specify filename for output CSV: \n")
    if filename:
        # File path
        script_dir = os.path.dirname(os.path.abspath(__file__))  # <-- absolute dir the script is in
        abs_file_path = os.path.join(script_dir + "/NN-data/")
        df.to_csv(abs_file_path + filename)

    # normalize data
    # df['date_time'] = pd.to_timedelta(df['date_time']).dt.total_seconds().astype(int)  # convert timestamp to float
    df = normalize_data(df)

    print("Finished formatting data \nX has dimensions: " + str(df.shape) + ", Y has dimensions: " + str(
        y.shape) + "\n")

    print("Format data completed")
    print("-----------------------------------------------------------\n")

    return df, y, True


def traverse(params, df):
    stop_loss, goal, hold, horizon = params[0], params[1], params[2], params[3]
    y = df.iloc[:, [0, 1]].values
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
        open = df.iloc[i, 1]
        date_search = df.iloc[i + 1, 0]
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
                if close / open >= goal:
                    y[i, 2] = 1
                elif close / open <= stop_loss:
                    y[i, 2] = 0
                    
            # Check if price is neutral (hold)
            if within_horizon and hold_search:
                if close / open >= 1 + hold or close / open <= 1 - hold:
                    y[i, 4] = 0
                    hold_search = 0
            elif not within_horizon and hold_search:
                y[i, 4] = 1

            # Check if price reaches target (short)
            if within_horizon:
                if close / open <= 1 / goal:
                    y[i, 3] = 1
                    short_result_found = 1
                elif close / open >= stop_loss:
                    y[i, 3] = 0
                    short_result_found = 1
            elif not within_horizon and short_result_found:
                break
            elif not within_horizon and not short_result_found:
                y[i, 3] = 0
                break

    print("done \n")
    return y


def normalize_data(df):
    # columns to normalize
    cols_to_norm = ['close']
    df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df


def plot_result(targets, span=1000, start=0):
    # Plotting:
    fig, ax = plt.subplots()
    x = targets[start:(start + span), 0]
    y1 = targets[start:(start + span), 1]
    ax.plot(x, y1)
    plt.xticks(rotation=80)  # rotate x ticks from horizontal
    plt.tight_layout()  # fit everything into window
    plt.grid(b=True, which='major', color='k', linestyle='--')  # Set grid

    # ax.fmt_xdata = DateFormatter('%Y-%m-%d %H:%M:%S')   # date_time format
    ax.xaxis.set_major_locator(MaxNLocator(20))     # number of x-axis ticks
    # x.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S"))    # x-axis ticks visual format

    for r in targets[start:(start+span)]:
        if r[2]:
            plt.axvline(x=r[0], color='g', alpha=0.2)
        if r[3]:
            plt.axvline(x=r[0], color='r', alpha=0.2)
        if r[4]:
            plt.axvline(x=r[0], color='lightyellow', alpha=0.4)
    plt.draw()
    return plt


def data_menu():
    global data_load_done
    global format_data_done
    global raw_data
    global pre_processed_data
    global target

    print("\nPrepare data:")
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
            data_menu()
        elif choice == '2':
            filename = input("specify file name: \n")
            filename = 'full.csv' if filename == '' else filename
            size = input("Specify size of dataset (x100000): \n")
            size = 5000 if size == '' else int(float(size)*100000)
            pre_processed_data, target, format_data_done = import_processed_data(filename, size, INTERVAL)
            data_menu()
        elif choice == '3' and data_load_done:
            pre_processed_data, target, format_data_done = calc_y(raw_data)
            data_menu()
        elif choice == '4' and format_data_done:
            span = input("specify span (default is 500): \n")
            span = 500 if span == '' else int(span)
            start = input("specify start (default is index 0): \n")
            start = 0 if start == '' else int(start)
            plt = plot_result(target, span, start)
            plt.show()
            data_menu()
        else:
            print("invalid input \n")
            data_menu()

    print("Exiting \n")
    exit()

if __name__ == '__main__':
    """ Start script """
    data_menu()
    exit()






