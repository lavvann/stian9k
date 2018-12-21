#!/usr/bin/python3.6
import sys  
import data

""" variables """
data_load_done = False
format_data_done = False
gold_data = []

print(sys.version)


def main_menu():
    global data_load_done
    global format_data_done
    global gold_data

    print("\n\nMain:")
    print("1: import gold data, imported state: " + str(data_load_done))
    print("2: import oil data, imported state: " + str(data_load_done))
    if data_load_done:
        print("3: Format data, format state: " + str(format_data_done))
        print("4: Start training") 
    print("exit: exit program")
    choice = input("select action: ")
    while choice != 'exit':
        if choice == '1':
            gold_data, data_load_done = data.import_data("GCtest.csv")
            print("gold data imported \n")
            main_menu()
        elif choice == '2':
            oil_data, data_load_done = data.import_data("CL.csv")
            main_menu()
        elif choice == '3':
            x, format_data_done = data.format_data_nn(gold_data, 2, 400, 2)
            print("Format data completed \n")
            main_menu()
        else:
            print("invalid input \n")
            main_menu()

    print("Exiting")
    exit()


""" Start script """
main_menu()

exit()

