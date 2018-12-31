#!/usr/bin/python3.6
import sys  
import data


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

    print("\n\nPrepare data:")
    print("1: import raw data, imported state: " + str(data_load_done))
    print("2: import formatted data, imported state: " + str(format_data_done))
    if data_load_done:
        print("3: Format raw data, format state: " + str(format_data_done))
    if format_data_done:    
        print("4: Plot calculated y")
        print("5: Continue to Machine learning")
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
            pre_processed_data, format_data_done = data.import_processed_data(filename)
            data_menu()
        elif choice == '3':
            pre_processed_data, format_data_done = data.calc_y(raw_data)
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
            return
        else:
            print("invalid input \n")
            data_menu()

    print("Exiting \n")
    exit()

def nn_meny():
    global pre_processed_data
    global nn_data_ready
    
    print("\n\nMachine learning:")
    print("1: create test and train sequence, NN data ready: " + str(nn_data_ready))
    print("2: Plot calculated y")
    print("3: start NN fitting")
    print("exit: exit program \n")

    choice = input("select action: ")
    while choice != 'exit':
        if choice == '1':
            x_train, x_test, nn_data_ready = data.create_nn_seq(pre_processed_data, 400)
            print("test and train data sequence created\n")
            nn_meny()
        elif choice == '2':
            span = input("specify span: \n")
            span = 500 if span == '' else int(span)
            start = input("specify start: \n")
            start = 0 if start == '' else int(start)
            plt = data.plot_result(pre_processed_data, span, start)
            plt.show()
            nn_menu()
        elif choice == '3':
            nn_meny()
        else:
            print("invalid input\n")
            nn_meny()
    print("Exiting \n")
    exit()

""" Start script """
data_menu()
nn_meny()

exit()

