#!/usr/bin/python3.6
""" IG gold trading simulation """
import sys
import numpy as np
import pandas as pd


# parameters
bank = 10000.0          # Start amount EUR
sg = 0.0                # spot gold price EUR
m = 0.0                 # margin (deposit for each trade)
sim_finished = False    # BOOL true when test set is completed



""" ------- MAIN --------- """
print("\n ------- IG simulation -------  \n")
while bank > m and not sim_finished:
    # check for input csv
    if not len(sys.argv) > 1:
       exit()
    try:
        fields = ['date_time', 'close', 'volume', 'y1', 'y2', 'y3']
        df = pd.read_csv(sys.argv[1], header=0, index_col=False)
        df['date_time'] = df['date_time'].astype('datetime64[ns]')  # correct date_time type definition
        df.rename(columns={'Unnamed: 0':'ix'}, inplace=True)
        print("Finished opening file, Y has dimensions: " + str(df.shape) + "\n" + str(df.keys())+ "\n")
    except Exception as ex:
        print("Something went wrong when reading df from file, error code: " + str(ex))
        sim_finished = True
        exit()
        
    exit()