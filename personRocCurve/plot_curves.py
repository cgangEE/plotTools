#!/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
import cPickle


def plot_roc(data, id):
    plt.figure()
    for i, r in enumerate(data):
        plt.plot(r['fppi'], r['recall'], label=id[i], linewidth=2.0)
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

def plot_curves(eval_result, curve_id):
    plot_roc(eval_result, curve_id)
    plt.savefig('curves') 


if __name__ == '__main__':
    curvesName = [  'sjtu-pvanet.pkl', 'sjtu-pvanet-Removed50.pkl',
                    'detvipl-pvanet.pkl','detvipl-pvanet-Removed50.pkl',
                    'detvipl-pvanet-ohem.pkl', 'detvipl-pvanet-ohem-Removed50.pkl',
                    'detviplV4-pvanet-ohem.pkl',
                    ]
    curves = []
    for name in curvesName:
        if os.path.exists(name):
            with open(name, 'rb') as fid:
                eval_result = cPickle.load(fid)
                curves += eval_result

    aliasCurvesName = ['detvipl-V2-pvanet', 'detvipl-V2-pvanet-Removed50',
                        'detvipl-V3-pvanet', 'detvipl-V3-pvanet-Removed50',
                        'detvipl-V3-pvanet-ohem', 
                        'detvipl-V3-pvanet-ohem-Removed50',
                        'detvipl-V4-test-pvanet-ohem', ]

    plot_curves(curves, aliasCurvesName)

