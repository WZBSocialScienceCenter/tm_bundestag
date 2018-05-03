# -*- coding: utf-8 -*-
"""
Generate plots for the evaluation results. Use the same parameters as in `tm_eval.py`.

Markus Konrad <markus.konrad@wzb.eu>
"""

import sys

import matplotlib.pyplot as plt
from tmtoolkit.utils import unpickle_file
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.topicmod.visualize import plot_eval_results

#%%

if len(sys.argv) != 4:
    print('run script as: %s  <tokens preprocessing pipeline> <eta> <alpha factor>' % sys.argv[0])
    print('<tokens preprocessing pipeline> must be 0, 1 or 2')
    exit(1)

toks = int(sys.argv[1])
eta = float(sys.argv[2])
alpha_mod = float(sys.argv[3])

#%%

picklefile = 'data/tm_eval_results_tok%d_eta_%.2f_alphamod_%.2f.pickle' % (toks, eta, alpha_mod)
print('loading pickle file with evaluation results from `%s`' % picklefile)

eval_results = unpickle_file(picklefile)
eval_results_by_n_topics = results_by_parameter(eval_results, 'n_topics')

n_metrics = len(eval_results_by_n_topics[0][1])

#%%

fig, axes = plot_eval_results(eval_results_by_n_topics,
                              title='Evaluation results for alpha=%.2f/k, beta=%.2f' % (alpha_mod, eta),
                              xaxislabel='num. topics (k)')

plot_file_eval_res = 'fig/tm_eval_results_tok%d_eta_%.2f_alphamod_%.2f.png' % (toks, eta, alpha_mod)
print('saving plot to file `%s`' % plot_file_eval_res)
plt.savefig(plot_file_eval_res)
plt.show()

print('done.')
