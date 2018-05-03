# -*- coding: utf-8 -*-
"""
Evaluation of topic models: Generate several models in parallel with different hyperparameters and use evaluation
metrics to find the "best" combination of hyperparameters.

As parameters, pass:
  - which data to take (i.e. from which preprocessing pipeline)
  - fixed value for eta (aka beta)
  - a factor X for alpha: X/K where K is the number of topics
  - the number of sampling iterations

Markus Konrad <markus.konrad@wzb.eu>
"""

from __future__ import division
import logging
import sys
from pprint import pprint

from tmtoolkit.utils import unpickle_file, pickle_data
from tmtoolkit.topicmod import tm_lda


DATA_PICKLE_DTM = 'data/speeches_tokens_%d.pickle'

logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('tmtoolkit')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True


if len(sys.argv) != 5:
    print('call script as: %s <tokens preprocessing pipeline> <eta> <alpha factor> <num. iterations>' % sys.argv[0])
    print('<tokens preprocessing pipeline> must be 0, 1 or 2')
    exit(1)

preproc_mode = int(sys.argv[1])
assert 0 <= preproc_mode <= 2
eta = float(sys.argv[2])
assert 0 < eta < 1
alpha_mod = float(sys.argv[3])
assert alpha_mod > 0
n_iter = int(sys.argv[4])
assert n_iter > 0

dtm_pickle = DATA_PICKLE_DTM % preproc_mode
print('loading DTM from file `%s`...' % dtm_pickle)
doc_labels, vocab, dtm, doc_tokens = unpickle_file(dtm_pickle)
assert len(doc_labels) == dtm.shape[0]
assert len(vocab) == dtm.shape[1]
tokens = list(doc_tokens.values())
del doc_tokens
assert len(tokens) == len(doc_labels)
print('loaded DTM with %d documents, %d vocab size, %d tokens' % (len(doc_labels), len(vocab), dtm.sum()))

print('evaluating topic models...')
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

eval_results = tm_lda.evaluate_topic_models(dtm, varying_params, constant_params,
                                            metric=('griffiths_2004', 'cao_juan_2009', 'arun_2010',
                                                    'coherence_mimno_2011', 'coherence_gensim_c_v'),
                                            coherence_gensim_vocab=vocab,
                                            coherence_gensim_texts=tokens)

pickle_file_eval_res = 'data/tm_eval_results_tok%d_eta_%.2f_alphamod_%.2f.pickle' % (preproc_mode, eta, alpha_mod)
print('saving results to file `%s`' % pickle_file_eval_res)
pickle_data(eval_results, pickle_file_eval_res)

print('done.')

