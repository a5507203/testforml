import numpy as np
import matplotlib.pyplot as plt
import os

def stringToList(line):

    a = line.replace('[','').replace(']','').split(',')
    return a

dir = 'data_and_results_ovr_v3/'
fileNames = []
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith(".txt") and ("t_test" not in file):
            fileNames.append(os.path.join(root, file))

# fileName = "new_data_and_results"
for fileName in fileNames:

    with open(fileName, 'r') as f:
        data = f.readlines()
        rates = stringToList(data[0])
        predictions = stringToList(data[1])

        corrects = []
        errors = []

        for i in range(len(predictions)):
            if(int(predictions[i]) == 1):
                corrects.append(float(rates[i]))
            elif(int(predictions[i]) == 0):
                errors.append(float(rates[i]))

        # [32,2]
        if 'dnn' in fileName:
            title = 'MR violation rate vs CV'
            xlabel = 'CV prediction'
        else:
            title = 'MR violation rate vs LOOCV'
            xlabel = 'LOOCV prediction'
        plt.boxplot([corrects,errors], labels=['Yes','No'],showmeans=True, sym='s',meanline=True)
        
        plt.title(title, fontsize=15)
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel('MR violation rate', fontsize=15)
        plt.savefig(fileName.replace(".txt","").replace(" = ","_").replace('[','').replace(']','')+".png", format="png")
        plt.clf()
        plt.cla()
        plt.close()