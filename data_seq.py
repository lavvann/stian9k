#!/usr/bin/python3.6
import sys
import data
import numpy as np
import pandas as pd
# dataset builder
# input: raw data .csv
# output: X0, X1, X2 csv files with price history within timeframes and Steps number of points

# PARAMETERS
timeframe_1 = 14        # [days]
timeframe_2 = 45        # [days] 
STEPS       = 200       # number of timesteps within each timeframe
INTERVAL    = 5         # base interval [min]

""" ------- MAIN --------- """
# check for input csv
if not len(sys.argv) > 1:
    exit()
df, dn, success = data.import_processed_data("", 0, INTERVAL, sys.argv[1])
if not success:
    print("Could not import file, exiting.\n")
    exit()
print(str(dn[0]) + " \n")

# row range
timeframe_1_range = int(timeframe_1*24*60/INTERVAL)
timeframe_2_range = int(timeframe_2*24*60/INTERVAL)
# row interval
interval_1 = round(timeframe_1_range/STEPS)
interval_2 = round(timeframe_2_range/STEPS)
# make default array
X0 = np.zeros((len(dn),STEPS))
X1 = np.zeros((len(dn),STEPS))
X2 = np.zeros((len(dn),STEPS))
n = len(df.index)
for i in range(0, len(df.index)-1, 1):
    #progressbar
    j = (i + 1) / n
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
    sys.stdout.write('\r')
    sys.stdout.flush()
    if i < timeframe_2_range:
        continue
    X0[i,:] = df.iloc[i-STEPS:i:1,0].values.transpose()
    X1[i,:] = df.iloc[i-timeframe_1_range:i:interval_1, 0].values.transpose()
    X2[i,:] = df.iloc[i-timeframe_2_range:i:interval_2, 0].values.transpose()

# save data to csv
df = pd.DataFrame(X0)
name = 'X' + str(INTERVAL) + 'Min.csv'
df.to_csv(name,index=False)
df = pd.DataFrame(X1)
name = 'X' + str(timeframe_1) + '.csv'
df.to_csv(name,index=False)
df = pd.DataFrame(X2)
name = 'X' + str(timeframe_2) + '.csv'
df.to_csv(name,index=False)

print(X0.shape)
print(X1.shape)
print(str(X2.shape) + "\n")

exit()