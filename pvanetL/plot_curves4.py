#!/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import cPickle



def plot_roc(data, id, componentList):

    num_part = len(componentList)
    num_id = len(id)
    

    for i in xrange(num_part):
        plt.figure()

        for j in xrange(num_id):
            print(i, j, j * 4 + (j >= num_id - 2 and i or i * 3))
            if j >= num_id - 2:
                r = data[ (num_id - 2) * 4 + (j - num_id + 2) * 2 + i]
            else:
                r = data[j * 4 + i * 3] 

            fppiIdx = sum(r['fppi'] < 0.1) 
            print(r['recall'][fppiIdx])
            plt.plot(r['fppi'], r['recall'], 
                    label=id[j] + '_' + str(r['recall'][fppiIdx])[:5], 
                    linewidth=2.0)

            print(id[j])

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
        plt.legend(loc='lower right', fontsize=10)
        plt.title('ROC Curve - ' + componentList[i])
        plt.grid(b=True, which='major', color='b', linestyle='-')
        plt.grid(b=True, which='minor', color='b', linestyle=':')
                
        plt.savefig('curves4' + componentList[i], 
                boxes_inches = 'tight', pad_inches = 0) 


def plot_curves(eval_result, curve_id, componentList):
    plot_roc(eval_result, curve_id, componentList)


if __name__ == '__main__':
    curvesName = []
    aliasCurvesName = []

    
    curvesName.append('psdb-pvanet-1.pkl')
    curvesName.append('psdb-pvanet-ohem-D-10.pkl')
    curvesName.append('psdb-pvanet-ohem-DRoiAlign-10.pkl')
    curvesName.append('psdb-pvanet-ohem-DRoiAlignX-10.pkl')
    curvesName.append('psdbFourParts-Ohem-pvanet-DRoiAlignX-10.pkl')
    curvesName.append('psdbUpperBody-pvanet-DRoiAlignX-10.pkl')
    curvesName.append('psdbUpperBody-Ohem-pvanet-DRoiAlignX-10.pkl')

    componentList = ['Pedestrain', 'Upperbody']
    
    aliasCurvesName.append('pvanet-RoiPooling')
    aliasCurvesName.append('pvanet-ohem-D-RoiPooling')
    aliasCurvesName.append('pvanet-ohem-D-RoiAlign')
    aliasCurvesName.append('pvanet-ohem-D-RoiAlign-X')
    aliasCurvesName.append('pvanet-ohem-D-RoiAlign-X-FourParts')
    aliasCurvesName.append('pvanet-D-RoiAlign-X-roiHalf')
    aliasCurvesName.append('pvanet-ohem-D-RoiAlign-X-roiHalf')


    curves = []
    for name in curvesName:
        if os.path.exists(name):
            with open(name, 'rb') as fid:
                print(name)
                eval_result = cPickle.load(fid)
                curves += eval_result

    plot_curves(curves, aliasCurvesName, componentList)

