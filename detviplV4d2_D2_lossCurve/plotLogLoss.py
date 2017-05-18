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
                iteration = line[0].strip().split('Iteration ')[1].split(',')[0]
                if 'iteration' not in ret:
                    ret['iteration'] = []
                ret['iteration'].append(iteration)    


            if name not in ret:
                ret[name] = []

            ret[name].append(loss)

    
    return ret



def plot_curves(curve, curve_id):

    for i, curve in enumerate(curves):
        for key in curve.keys():
            plt.figure()

            '''
            for x in curve[key]:
                print(x)
                print(float(x))
            print(curve_id[i], key)

            for x in curve['iteration']:
                print(x)
                print(float(x))
            '''

            plt.plot(curve['iteration'], curve[key], \
                    label=key + '_' + curve_id[i] , linewidth=0.5)

            plt.draw()
            ax = plt.gca()
            #    ax.set_ylim([0, 2])
            #ax.set_yscale('log')
            plt.xlabel('Iteration', fontsize=16)
            plt.ylabel('Loss', fontsize=16)
            plt.legend(loc='upper left', fontsize=10)
            plt.title(key + '_' + curve_id[i])
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='k', linestyle=':')
            plt.savefig(key + '_' + curve_id[i]) 
    
if __name__ == '__main__':
    curvesName = ['log_detviplV4d2_D2']
    curves = []
    for name in curvesName:
        if os.path.exists(name):
            curves.append(getLoss(name))
            print(name)

    aliasCurvesName = ['detviplV4d2-D']

    plot_curves(curves, aliasCurvesName)

