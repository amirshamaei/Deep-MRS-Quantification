import csv
import json

from matplotlib import pyplot as plt


import engine as eng
def main():
    """
    It opens a csv file, then opens a json file, then for each run in the json file, it creates an engine object, trains it,
    and tests it
    """
    file = open('training_report.csv', 'w+')
    writer = csv.writer(file)
    json_file_path = 'runs/exp4.json'
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
    for run in contents['runs']:
        if run["version"] in ["5/"]:
            engine = eng.Engine(run)
            # engine.tuner()
            engine.dotrain()
            engine.dotest()
    file.close()
if __name__ == '__main__':
    main()