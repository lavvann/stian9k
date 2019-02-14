#!/usr/bin/python3.6
""" data import and manipulation """
import os
import sys
import numpy as np
import pandas as pd

""" THIS FILE CONTAINS DIFFERENT DATA EVALUATON STRATEGIES 
- Binary traverse: Check if value has changed above threshold within horizon. y1,y2,y3: binary buy, short, hold
- Gradient traverse: max/min value within horizon. y1 float value 
"""


def binary_traverse(params, df):
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
                str(date_time + timedelta(minutes=horizon)), "%Y-%m-%d %H:%M:%S")
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


def gradient_traverse(params, df):
    horizon = params[0]
    y = df.iloc[:, [0, 1]].values
    y = np.c_[y, np.zeros(y.shape[0], dtype=float)]   # create column for gradient y
    y = np.c_[y, np.zeros(y.shape[0], dtype=int)]   # create column for spare y bool
    y = np.c_[y, np.zeros(y.shape[0], dtype=int)]   # create column for spare y bool
    n = len(df.index)
    for i in range(0, len(df.index)-1, 1):
        # progressbar
        j = (i + 1) / n
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
        sys.stdout.write('\r')
        sys.stdout.flush()
        spot = df.iloc[i, 1]
        # Y1: find min/max within horizon and calculate gradient "a" in y=ax 
        top = df.iloc[i:i+(horizon-1), 1].max()
        index_top = df.iloc[i:i+(horizon-1), 1].idxmax()
        bot = df.iloc[i:i+(horizon-1), 1].min()
        index_bot = df.iloc[i:i+(horizon-1), 1].idxmin()
        # find and calc gradient
        if (top-spot) > (spot-bot) and not index_top == i:
            y[i, 2] = (top-spot)
        elif (spot-bot) > (top-spot) and not index_bot  == i:
            y[i, 2] = (bot-spot)
        # Y2: Calucalte mean value within horizon
        y[i,3] = df.iloc[i:i+(horizon-1), 1].mean()

    print("done \n")
    return y