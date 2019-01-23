#!/usr/bin/python3.6
""" IG gold trading simulation """
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# parameters
bank = 10000.0          # Start amount EUR
sg = 0.0                # spot gold price EUR
m = 0.0                 # margin (deposit for each trade)
sim_finished = False    # BOOL true when test set is completed
goal = 1.003            # default training gain factor 
spread = 0.3            # default IG spread $
std_pos = 100           # default trade amount


def plot_result(y, span=1000, start=0):
    # Plotting:
    fig, ax = plt.subplots()
    x = y[start:(start + span), 0]
    y1 = y[start:(start + span), 1]
    y2 = y[start:(start + span), 5]
    ax.plot(x, y1, y2)
    plt.xticks(rotation=80)  # rotate x ticks from horizontal
    plt.tight_layout()  # fit everything into window
    plt.grid(b=True, which='major', color='k', linestyle='--')  # Set grid

    # ax.fmt_xdata = DateFormatter('%Y-%m-%d %H:%M:%S')   # date_time format
    ax.xaxis.set_major_locator(MaxNLocator(20))     # number of x-axis ticks
    # x.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M:%S"))    # x-axis ticks visual format

    for r in y[start:(start+span)]:
        if r[2]:
            plt.axvline(x=r[0], color='g', alpha=0.2)
        if r[3]:
            plt.axvline(x=r[0], color='r', alpha=0.2)
        if r[4]:
            plt.axvline(x=r[0], color='lightyellow', alpha=0.4)
    plt.draw()
    return plt


""" ------- MAIN --------- """
print("\n ------- IG simulation -------  \n")
# check for input csv and read
if not len(sys.argv) > 1:
    exit()
try:
    fields = ['date_time', 'close', 'volume', 'y1', 'y2', 'y3']
    df = pd.read_csv(sys.argv[1], header=0, index_col=False)
    df['date_time'] = df['date_time'].astype('datetime64[ns]')  # correct date_time type definition
    df.rename(columns={'Unnamed: 0':'ix'}, inplace=True)
    y = df.iloc[:, [0, 2, 4, 5, 6]].values
    print("Finished opening file, Y has dimensions: " + str(df.shape) + "\n" + str(df.keys())+ "\n")
except Exception as ex:
    print("Something went wrong when reading df from file, error code: " + str(ex))
    sim_finished = True
    exit()

y = np.c_[y, np.zeros(y.shape[0], dtype=float)]   # create column for bank

# Start trading
while not sim_finished:
    trading_l = False
    trading_s = False
    for i in range(0, len(y)-1, 1):
        sg = y[i, 1] # current spot gold
        # long
        if y[i, 3]:
            if not trading_l:
                # 1 Contract cost
                m1 = (sg+spread)*0.057
                if bank/m1 > std_pos:   # if enough money to open standard position
                    position = std_pos  # else as many as possible, break if 0
                else:
                    position = int(bank/m1)
                    if position == 0: break
                l_start = sg
                trading_l = True
            target = y[i, 1] * goal   # Update target

        # short
        if y[i, 4]:
            if not trading_s:
                # 1 Contract cost
                m1 = (sg-spread)*0.057
                if bank/m1 > std_pos:   # if enough money to open standard position
                    position = std_pos  # else as many as possible, break if 0
                else:
                    position = int(bank/m1)
                    if position == 0: break
                s_start = sg
                trading_s = True
            target = y[i, 1] * (1/goal)  # Update target

        # hold or end position
        if trading_l:   # end long
            if y[i, 1] >= target:
                position = 0
                bank += std_pos*(l_start-sg) - (std_pos*spread)  # Trade gain EUR
                trading_l = False
                target = 0
        if trading_s:  # end short
            if y[i, 1] <= target:
                position = 0
                bank += std_pos*(sg-s_start) - (std_pos*spread)  # Trade gain EUR
                trading_s = False
                target = 0

        # logging
        y[i, 5] = bank

    sim_finished = True
print("Simulation completed \n")
plot = plot_result(y)
plot.show()
exit()

