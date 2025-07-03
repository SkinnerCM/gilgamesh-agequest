import csv
from typing import List

def list_writer(ur_list, file_name):

    with open(file_name, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(ur_list)
        
def list_reader(file,  encoding="utf8"):

    with open(file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    data = data[0]

    return data