#!/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import cPickle



def plot_roc(data, id):

    linestyles = ['-', '--', ':']
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

    plt.figure()
    for i, r in enumerate(data):
        plt.plot(r['fppi'], r['recall'], linestyle=linestyles[i/7], color=colors[i%7], label=id[i], linewidth=2.0)
        fppiIdx = sum(r['fppi'] < 0.1) 
        plt.annotate('{:.3f}'.format(r['recall'][fppiIdx]), \
            xy = (r['fppi'][fppiIdx] + 0.01, r['recall'][fppiIdx] + i * 0.02 - 0.02), \
            textcoords='data') 
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
    plt.savefig('curvesXX') 


if __name__ == '__main__':
    curvesName = []
    aliasCurvesName = []

    curvesName.append('detviplV4d2XX-pvanet-ohem-DRoiAlignX-10.pkl')
    curvesName.append('detviplV4d2XX-pvanet-ohem-DRoiAlignX-Ge50-10.pkl')
    #curvesName.append('detviplV4d2XX-pvanet-ohem-DRoiAlignX-Ge50-(2)10.pkl')

    aliasCurvesName.append('detviplV4d2XX-ohem-D-RoiAlignX-54ms')
    aliasCurvesName.append('detviplV4d2XX-ohem-D-RoiAlignX-Ge50-54ms')
    #aliasCurvesName.append('detviplV4d2XX-ohem-D-RoiAlignX-Ge50(2)-54ms')

    curves = []
    for name in curvesName:
        if os.path.exists(name):
            with open(name, 'rb') as fid:
                eval_result = cPickle.load(fid)
                curves += eval_result

    plot_curves(curves, aliasCurvesName)

