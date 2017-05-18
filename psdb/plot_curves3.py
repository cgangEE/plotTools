#!/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import cPickle



def plot_roc(data, id):

    linestyles = ['-', '--', ':']
    colors = ('b', 'g', 'r', 'c')

    plt.figure()
    for i in range(2):
        for j in range(0, 2, 1):
            r = data[j * 4 + i]
            fppiIdx = sum(r['fppi'] < 0.1) 
            plt.plot(r['fppi'], r['recall'], linestyle=linestyles[j], color=colors[i], label=id[j * 4 + i], linewidth=2.0)

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
    plt.savefig('curves3') 


if __name__ == '__main__':
    curvesName = []
    aliasCurvesName = []

    curvesName.append('psdb-pvanet-ohem-DRoiAlignX-10.pkl')
    curvesName.append('psdbHead-pvanet-DRoiAlignX-10.pkl')


    curves = []
    nameToCurves = {}

    for name in curvesName:
        print(name)
        if os.path.exists(name):
            with open(name, 'rb') as fid:
                print(name)
                
                eval_result = cPickle.load(fid)
                curves += eval_result
                nameToCurves[name] = eval_result 

    componentList = ['pedestrain', 'head', 'head-shoulder', 'upperbody']
    
    aliasCurvesName.append('pvanet-ohem-D-RoiAlign-X')
    aliasCurvesName.append('2boxes-pvanet-D-RoiAlign-X')

    newAliasCurvesName = []
    for i, alias in enumerate(aliasCurvesName):
        name = curvesName[i]
        curve = nameToCurves[name]
        for j in xrange(len(curve)):
            component = componentList[j]
            newAliasCurvesName.append(component + '-' + alias)

    aliasCurvesName = newAliasCurvesName

    print('')
    for alias in newAliasCurvesName:
        print(alias)

    plot_curves(curves, aliasCurvesName)

