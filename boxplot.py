import numpy as np
import matplotlib.pyplot as plt
import os

def stringToList(line):

    a = line.replace('[','').replace(']','').split(',')
    return a

dir = 'data_and_results/Frogs_MFCCs'
fileNames = []
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith(".txt") and ("t_test" not in file):
            fileNames.append(os.path.join(root, file))

# fileName = "data_and_results\Frogs_MFCCs\/0.00000000e+00\gaussianNB\Frogs_MFCCs_label3_2_attribute3_gaussianNB_sigmas = [0.5]_std = 0.16032757095257638_mr_and_loocv_result.txt"
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
        plt.boxplot([corrects,errors], labels=['Yes','No'],showmeans=True, sym='s',meanline=True)
        plt.title('MR violation rate vs LOOCV', fontsize=10)
        plt.xlabel('LOOCV prediction')
        plt.ylabel('MR violation rate')
        plt.savefig(fileName.replace(".txt","")+".svg", format="svg")
        plt.clf()
        plt.cla()
        plt.close()