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
goal = 1.003            # default training gain factor 
spread = 0.3            # default IG spread $


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
    print("Finished opening file, Y has dimensions: " + str(df.shape) + "\n" + str(df.keys())+ "\n")
except Exception as ex:
    print("Something went wrong when reading df from file, error code: " + str(ex))
    sim_finished = True
    exit()
    
# Start trading
while not sim_finished:
    trading_l = False
    trading_s = False
    for i in range(0, len(df.index)-1, 1):
        sg = df.iloc[i, 1] # current spot gold
        # long
        If df.iloc[i, 3]
            if not trading_l:
                # 1 Contract cost
                m1 = (sg+spread)*0,057
                if bank/m1 > std_pos:   # if enough money to open standard position
                    position = std_pos  # else as many as possible, break if 0
                else:
                    position = int(bank/m1)
                    if position == 0: break
                l_start = sg
                trading_l = True
            target = df.iloc[i, 1] * goal   # Update target
                  
        # short
        If df.iloc[i, 4]
            if not trading_s:
                # 1 Contract cost
                m1 = (sg-spread)*0,057
                if bank/m1 > std_pos:   # if enough money to open standard position
                    position = std_pos  # else as many as possible, break if 0
                else:
                    position = int(bank/m1)
                    if position == 0: break
                s_start = sg
                trading_s = True
            target = df.iloc[i, 1] * (1/goal)  # Update target
            
        # hold or end position
        if trading_l:   # end long
            if df.iloc[i, 1] >= target:
                position = 0
                bank += std_pos*(l_start-sg) - (std_pos*spread)  # Trade gain EUR
                trading_l = False
                target = 0
         if trading_s:  # end short
            if df.iloc[i, 1] <= target:
                position = 0
                bank += std_pos*(sg-s_start) - (std_pos*spread)  # Trade gain EUR
                trading_s = False
                target = 0
        
        # logging
        pass
    sim_finished = True
print("Simulation completed \n")
exit()