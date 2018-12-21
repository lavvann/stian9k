#!usr/bin/python3.6
""" data import and manipulation """
import os
# import numpy as np
import pandas as pd


def import_data(file_name):
    # File path
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    abs_file_path = os.path.join(script_dir + "/data/" + file_name)

    # Select fields
    fields = ['date', 'time', 'close', 'volume']

    # Read file
    print("Opening file and merging date and time column \n \n")
    try:
        data = pd.read_csv(abs_file_path, parse_dates=[['date', 'time']], header=0, usecols=fields)
    except Exception as ex:
        print("Something went wrong when reading data from file: " + ex)
        return

    data_load_done = True
    print("Finished opening file \nData has dimensions: " + str(data.shape) + "\n\n")

    return data, data_load_done


def format_data_nn(data, num_batch, time_steps, features):
    # shape data to 3-dimensional X
    x = data.reshape(num_batch, time_steps, features)
    return x, True


