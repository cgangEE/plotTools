#!/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import cPickle


def plot_roc(data, id):
    plt.figure()
    for i, r in enumerate(data):
        plt.plot(r['fppi'], r['recall'], label=id[i], linewidth=2.0)
        fppiIdx = sum(r['fppi'] < 0.1) 
        plt.annotate('{:.3f}'.format(r['recall'][fppiIdx]), \
            xy = (r['fppi'][fppiIdx] + 0.01, r['recall'][fppiIdx]), \
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
    plt.savefig('curves_tmp') 


if __name__ == '__main__':
    curvesName = [ \
                    'detviplV4-pvanet-ohem-12000.pkl', \
                    'detviplV4-pvanet-ohem-640x1000.pkl', \
                    'detviplV4-pvanet-ohem-finetune.pkl', \
                    ]
    curves = []
    for name in curvesName:
        if os.path.exists(name):
            with open(name, 'rb') as fid:
                eval_result = cPickle.load(fid)
                curves += eval_result

    aliasCurvesName = [ 
                    'detviplV4-pvanet-ohem-12000',\
                    'detviplV4-pvanet-ohem',\
                    'detviplV4-pvanet-ohem-finetune', \
                    ]

    plot_curves(curves, aliasCurvesName)

