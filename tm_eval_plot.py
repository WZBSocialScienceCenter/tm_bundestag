import sys

import matplotlib.pyplot as plt
from tmtoolkit.utils import unpickle_file
from tmtoolkit.lda_utils.common import results_by_parameter
from tmtoolkit.lda_utils.visualize import plot_eval_results

if len(sys.argv) != 4:
    print('run script as: %s  <tokens preprocessing pipeline> <eta> <alpha factor>' % sys.argv[0])
    print('<tokens preprocessing pipeline> must be 0, 1 or 2')
    exit(1)

toks = int(sys.argv[1])
eta = float(sys.argv[2])
alpha_mod = float(sys.argv[3])

picklefile = 'data/tm_eval_results_tok%d_eta_%.2f_alphamod_%.2f.pickle' % (toks, eta, alpha_mod)
print('loading pickle file with evaluation results from `%s`' % picklefile)

eval_results = unpickle_file(picklefile)
eval_results_by_n_topics = results_by_parameter(eval_results, 'n_topics')

fig, axes = plot_eval_results(eval_results_by_n_topics, title='Evaluation results for alpha=%.2f/k, beta=%.2f' % (alpha_mod, eta),
                              metric=('arun_2010', 'cao_juan_2009', 'griffiths_2004'),
                              xaxislabel='num. topics (k)', figsize=(8, 6))

alpha_mod_int = int(alpha_mod)
if alpha_mod_int > 10:
    for ax in axes.flatten():
        ax.plot([alpha_mod_int, alpha_mod_int], ax.get_ylim(), color='k', linestyle='--')

plot_file_eval_res = 'fig/tm_eval_results_tok%d_eta_%.2f_alphamod_%.2f.png' % (toks, eta, alpha_mod)
print('saving plot to file `%s`' % plot_file_eval_res)
plt.savefig(plot_file_eval_res)
plt.show()

print('done.')
