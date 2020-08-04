import csv

def save_to_csv(file, fields):
    with open(file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
