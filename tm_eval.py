# -*- coding: utf-8 -*-

from __future__ import division
import logging
import sys
from pprint import pprint

import matplotlib.pyplot as plt

from tmtoolkit.utils import unpickle_file, pickle_data
from tmtoolkit.lda_utils import tm_lda
from tmtoolkit.lda_utils.common import results_by_parameter
from tmtoolkit.lda_utils.visualize import plot_eval_results


DATA_PICKLE_DTM = 'data/speeches_tokens_nosalut.pickle'

logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('tmtoolkit')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True

print('loading DTM...')
doc_labels, vocab, dtm = unpickle_file(DATA_PICKLE_DTM)
assert len(doc_labels) == dtm.shape[0]
assert len(vocab) == dtm.shape[1]
print('loaded DTM with %d documents, %d vocab size, %d tokens' % (len(doc_labels), len(vocab), dtm.sum()))

print('evaluating topic models...')
eta = float(sys.argv[1]) if len(sys.argv) >= 2 else 0.01  # lda default: 0.01
alpha_mod = float(sys.argv[2]) if len(sys.argv) >= 3 else 1.0
n_iter = int(sys.argv[3]) if len(sys.argv) >= 4 else 2000  # lda default: 2000
constant_params = dict(n_iter=n_iter,
#                       random_state=1,
                       eta=eta)
print('constant parameters:')
pprint(constant_params)
varying_num_topics = list(range(20, 100, 10)) + list(range(100, 200, 20)) + list(range(200, 501, 50))
#varying_num_topics = list(range(5,11))
varying_alpha = [alpha_mod/k for k in varying_num_topics]
varying_params = [dict(n_topics=k, alpha=a) for k, a in zip(varying_num_topics, varying_alpha)]
print('varying parameters:')
pprint(varying_params)

eval_results = tm_lda.evaluate_topic_models(dtm, varying_params, constant_params)

pickle_file_eval_res = 'data/tm_eval_results2_eta_%f.pickle' % eta
print('saving results to file `%s`' % pickle_file_eval_res)
pickle_data(eval_results, pickle_file_eval_res)

eval_results_by_n_topics = results_by_parameter(eval_results, 'n_topics')

fig, ax = plt.subplots()
plot_eval_results(fig, ax, eval_results_by_n_topics,
                  xaxislabel='num. topics k', yaxislabel='normalized metric result',
                  title='Evaluation results for alpha=%d/k, beta=%.2f' % (alpha_mod, eta))
plot_file_eval_res = 'fig/tm_eval_results2_eta_%f.png' % eta
print('saving plot to file `%s`' % plot_file_eval_res)
plt.tight_layout()
plt.savefig(plot_file_eval_res)
plt.show()

print('done.')
