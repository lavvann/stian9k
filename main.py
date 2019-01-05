#!/usr/bin/python3.6
import sys  
import data
import nn


""" variables """
data_load_done = False
format_data_done = False
raw_data = []
pre_processed_data = []
nn_data_ready = False

print(sys.version)


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
        print("5: Start NN")
    print("exit: exit program \n")
    choice = input("select action: ")
    while choice != 'exit':
        if choice == '1':
            filename = input("specify file name: \n")
            raw_data, data_load_done = data.import_raw_data(filename)
            print("raw data imported \n")
            data_menu()
        elif choice == '2':
            filename = input("specify file name: \n")
            filename = 'full.csv' if filename == '' else filename
            size = input("Specify size of dataset (x100000): \n")
            size = 100000 if size == '' else int(size)*100000
            long = input("specify long 1/0: \n")
            long = 1 if long == '' else 0
            pre_processed_data, target, format_data_done = data.import_processed_data(filename, size, long)
            data_menu()
        elif choice == '3':
            long = input("specify long 1/0: \n")
            long = 1 if long == '' else 0
            pre_processed_data, target, format_data_done = data.calc_y(raw_data, long)
            print("Format data completed \n")
            data_menu()
        elif choice == '4':
            span = input("specify span: \n")
            span = 500 if span == '' else int(span)
            start = input("specify start: \n")
            start = 0 if start == '' else int(start)
            plt = data.plot_result(pre_processed_data, span, start)
            plt.show()
            data_menu()
        elif choice == '5':
            nn.nn_gen(pre_processed_data, target)
            print("nn generation and training completed\n")
            data_menu()
        else:
            print("invalid input \n")
            data_menu()

    print("Exiting \n")
    exit()


""" Start script """
data_menu()

exit()

