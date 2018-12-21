import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
rcParams.update({'font.weight': 'bold'})
rcParams.update({'font.size': 10})
rcParams.update({'axes.titleweight': 'bold'})
rcParams.update({'axes.labelweight': 'bold'})
def stringToList(line):

    a = line.replace('[','').replace(']','').split(',')
    return a


dir = 'data_and_results/sensor_readings_24/7.19713968e-01/10nn/'
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
        graphData.append([corrects,errors])



fig, ax = plt.subplots()

bp1 = ax.boxplot(graphData[0], widths=0.2, positions=[0.25,0.75], sym='s')

bp2 = ax.boxplot(graphData[1], widths=0.2, positions=[1.5,2], sym='s')

bp3 = ax.boxplot(graphData[2],  positions=[2.75,3.25], widths=0.2,  sym='s')
alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}

alignment2 = {'horizontalalignment': 'left', 'verticalalignment': 'baseline'}
ax.set_xticks([0.25,0.75,1.5,2,2.75,3.25])
ax.set_xticklabels(["yes","no","yes","no","yes","no"])
ax.set_xlim(0,5)
ax.set_ylim(-0.1,1.1)


font0 = FontProperties()
font0.set_size = 10
font0.set_weight = 'normal'
t = plt.text(0.5, -0.25, '(1)', fontproperties=font0,**alignment)
t = plt.text(1.75, -0.25, '(2)', fontproperties=font0,**alignment)
t = plt.text(3, -0.25, '(3)', fontproperties=font0,**alignment)

#### lagend
t = plt.text(3.57, 0.9, '(1) n~N(0,0.04\u03C3\u00B2)', fontproperties=font0,**alignment2)
t = plt.text(3.57, 0.7, '(2) n~N(0,1\u03C3\u00B2)', fontproperties=font0,**alignment2)
t = plt.text(3.57, 0.5, '(3) n~N(0,4\u03C3\u00B2)', fontproperties=font0,**alignment2)

title = 'MRVR vs LOOCV'
xlabel = 'LOOCV predictions (error: 0.140)'
fig.subplots_adjust(bottom=0.2)
plt.title(title, fontsize=15)
plt.xlabel(xlabel, fontsize=15)
plt.ylabel('MRVR', fontsize=15)
ax.xaxis.set_label_coords(0.5, -0.15)


# plt.show()

plt.savefig(dir+'sensor_readings_24_label1_2_attribute1_10nn_new_combined_bar_plots.svg', format="svg")
plt.savefig(dir+'sensor_readings_24_label1_2_attribute1_10nn_new_combined_bar_plots.pdf', format="pdf")
plt.clf()
plt.cla()
plt.close()