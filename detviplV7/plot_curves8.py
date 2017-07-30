#!/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import cPickle



def plot_roc(data, id):

    linestyles = ['-', '--', ':']
    colors = ('b', 'g', 'r', 'c')

    plt.figure()
    for i in range(4):
        for j in range(0, 1):
            r = data[j * 4 + i]
            fppiIdx = sum(r['fppi'] < 0.1) 
            plt.plot(r['fppi'], r['recall'], linestyle=linestyles[j], color=colors[i], label=id[j * 4 + i] + '_' + str(r['recall'][fppiIdx])[:5], linewidth=2.0)

            plt.annotate('{:.3f}'.format(r['recall'][fppiIdx]), \
                xy = (r['fppi'][fppiIdx] + 0.01, r['recall'][fppiIdx]), \
                textcoords='data', color=colors[i%7]) 
               # color='green')

    plt.draw()
    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_xscale('log')

   
    
    plt.xlabel('FPPI', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.legend(loc='lower right', fontsize=10)
    plt.title('ROC Curve')
    plt.grid(b=True, which='major', color='b', linestyle='-')
    plt.grid(b=True, which='minor', color='b', linestyle=':')

def plot_curves(eval_result, curve_id):
    plot_roc(eval_result, curve_id)
    plt.savefig('curves8') 


if __name__ == '__main__':
    curvesName = []
    aliasCurvesName = []

    curvesName.append('detviplV7d5-pvanet-ohem-DRoiAlignX-Ge50-4.pkl')

    componentList = ['pedestrain', 'vehicle', 'riding_tool', 'riding_whole']
    
    aliasCurvesName.append('detviplV7.5-ge50')

    newAliasCurvesName = []
    for alias in aliasCurvesName:
        for component in componentList:
            newAliasCurvesName.append(component + '-' + alias)

    aliasCurvesName = newAliasCurvesName

    print(newAliasCurvesName)
    curves = []
    for name in curvesName:
        print(name)
        if os.path.exists(name):
            with open(name, 'rb') as fid:
                print(name)
                eval_result = cPickle.load(fid)
                curves += eval_result

    plot_curves(curves, aliasCurvesName)

