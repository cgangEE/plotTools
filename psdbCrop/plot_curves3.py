#!/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import cPickle



def plot_roc(data, id, componentList):

    num_id = len(id)
   
    idList = [0, 1, 2, 3, 4, 5]


    for j in xrange(num_id):
        r = data[idList[j]]

        fppiIdx = sum(r['fppi'] < 0.1) 
        print(r['recall'][fppiIdx])
        plt.plot(r['fppi'], r['recall'], label= id[j] + '_' + str(r['recall'][fppiIdx])[:5], 
                linewidth=2.0)

        plt.annotate('{:.3f}'.format(r['recall'][fppiIdx]), \
                xy = (r['fppi'][fppiIdx] + 0.01, r['recall'][fppiIdx]), \
                textcoords='data') 

    print('')

    plt.draw()
    ax = plt.gca()
    ax.set_ylim([0, 1])
    ax.set_xscale('log')

    
    plt.xlabel('FPPI', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.legend(loc='upper left', fontsize=10)
    plt.title('ROC Curve')
    plt.grid(b=True, which='major', color='b', linestyle='-')
    plt.grid(b=True, which='minor', color='b', linestyle=':')
                
    plt.savefig('curves3', 
            boxes_inches = 'tight', pad_inches = 0) 


def plot_curves(eval_result, curve_id, componentList):
    plot_roc(eval_result, curve_id, componentList)


if __name__ == '__main__':
    curvesName = []
    aliasCurvesName = []

    
    curvesName.append('psdbCrop-psdbFourParts-fppi0.1-Ohem-pvanet-DRoiAlignX-10.pkl')
    curvesName.append('psdbCrop-psdbFourParts-Mul-fppi0.1-Ohem-pvanet-DRoiAlignX-10.pkl')
    curvesName.append('psdbCrop-psdbFourParts-fppi1-Ohem-pvanet-DRoiAlignX-10.pkl')
    curvesName.append('psdbCrop-psdbFourParts-Mul-fppi1-Ohem-pvanet-DRoiAlignX-10.pkl')
    curvesName.append('psdbCrop-psdbFourParts-All-Ohem-pvanet-DRoiAlignX-10.pkl')
    curvesName.append('psdbCrop-psdbFourParts-Mul-All-Ohem-pvanet-DRoiAlignX-10.pkl')

    
    aliasCurvesName.append('conditional-fppi0.1')
    aliasCurvesName.append('full-fppi0.1')
    aliasCurvesName.append('conditional-fppi1')
    aliasCurvesName.append('full-fppi1')
    aliasCurvesName.append('conditional-All')
    aliasCurvesName.append('full-All')


    componentList = []


    curves = []
    for name in curvesName:
        print(name)
        if os.path.exists(name):
            with open(name, 'rb') as fid:
                print(name)
                eval_result = cPickle.load(fid)
                curves += eval_result

    plot_curves(curves, aliasCurvesName, componentList)

