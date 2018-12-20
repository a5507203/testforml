import numpy as np
import matplotlib.pyplot as plt
import os

def stringToList(line):

    a = line.replace('[','').replace(']','').split(',')
    return a

dir = 'data_and_results/sensor_readings_24/7.19713968e-01/gaussianNB'
fileNames = []
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith(".txt") and ("t_test" not in file) and ("[0.5]" not in file):
            fileNames.append(os.path.join(root, file))
print (fileNames)

# fileName = "data_and_results\Frogs_MFCCs\/0.00000000e+00\gaussianNB\Frogs_MFCCs_label3_2_attribute3_gaussianNB_sigmas = [0.5]_std = 0.16032757095257638_mr_and_loocv_result.txt"
graphData = []
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


    
#         # [32,2]
    print(graphData)


title = 'MR violation rate vs LOOCV'
xlabel = 'LOOCV predictions with noise ~ N(0,(0.2\u03C3)^2), N(0,(1\u03C3)^2),N(0,(2\u03C3)^2)'
plt.boxplot(graphData, labels=['Yes','No','Yes','No','Yes','No'],showmeans=True, sym='s',meanline=True)
plt.legend(loc='upper right')
plt.title(title, fontsize=15)
plt.xlabel(xlabel, fontsize=15)
plt.ylabel('MR violation rate', fontsize=15)
plt.savefig('sensor_readings_24_nbc_new_combined_bar_plots.svg', format="svg")
plt.clf()
plt.cla()
plt.close()