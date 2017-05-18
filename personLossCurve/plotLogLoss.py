#!/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt


def getLoss(filename):
    f = open(filename, 'r')
    ret = {}
    for line in f:
        if line.find(' solver.cpp') != -1 and line.find('loss') != -1:
            line = line.strip().split(' = ')
            name = line[0].strip().split()[-1]
            loss = line[1].strip().split()[0]

            if line[0].find("Iteration") != -1:
                iteration = line[0].strip().split(',')[0].split()[-1]
                if 'iteration' not in ret:
                    ret['iteration'] = []
                ret['iteration'].append(iteration)    


            if name not in ret:
                ret[name] = []

            ret[name].append(loss)

    return ret



def plot_curves(curve, curve_id):
    plt.figure()

    for i, curve in enumerate(curves):
        plt.plot(curve['iteration'], curve['loss'], \
                    label=curve_id[i], linewidth=0.5)

    plt.draw()
    ax = plt.gca()
#    ax.set_ylim([0, 2])
    ax.set_yscale('log')
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc='upper left', fontsize=10)
    plt.title('Loss Curve')
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='k', linestyle=':')
    plt.savefig('curves') 
    
if __name__ == '__main__':
    curvesName = ['log_detvipl', 'log_detvipl_ohem', 'log_detvipl_ohem_v2']
    curves = []
    for name in curvesName:
        if os.path.exists(name):
            curves.append(getLoss(name))
            print(name)

    aliasCurvesName = ['detvipl-V3-pvanet', 'detvipl-V3-pvanet-ohem', \
                        'detvipl-V3-pvanet-ohem-batchsize-2000']

    plot_curves(curves, aliasCurvesName)

