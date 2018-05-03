# -*- coding: utf-8 -*-
"""
Topic Modeling for 18th German Bundestag debates. Model generation with best model parameters found by model selection
in tm_eval.py.

As parameter, pass which data should be loaded, i.e. with which preprocessing pipeline it was generated:

  1 -> use merged speeches, default pipeline
  2 -> use merged speeches, remove salutatory addresses, default pipeline


Markus Konrad <markus.konrad@wzb.eu>
"""

from __future__ import division
import sys
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from lda import LDA
from tmtoolkit.topicmod.model_io import print_ldamodel_doc_topics, print_ldamodel_topic_words, \
    save_ldamodel_summary_to_excel
from tmtoolkit.utils import unpickle_file, pickle_data

#%% input args

if len(sys.argv) != 2:
    print('run script as: %s  <tokens preprocessing pipeline>' % sys.argv[0])
    print('<tokens preprocessing pipeline> must be 1 or 2')
    exit(1)

toks = int(sys.argv[1])

assert toks in (1, 2)

#%% model hyperparameters


if toks == 1:
    K = 130
    alpha_mod = 10.0
    beta = 0.1
elif toks == 2:
    K = 130
    alpha_mod = 10.0
    beta = 0.1
else:
    print('<tokens preprocessing pipeline> must be 1 or 2')
    exit(2)

LDA_PARAMS = dict(
    n_topics=K,
    alpha=alpha_mod/K,
    eta=beta,
    n_iter=2000
)

# other parameters
BURNIN = 5   # with a default of refresh=10 this means 50 burnin iterations

# paths to data files

DATA_PICKLE_DTM = 'data/speeches_tokens_%d.pickle' % toks
LDA_MODEL_PICKLE = 'data/model%d.pickle' % toks
LDA_MODEL_LL_PLOT = 'data/model%d_logliks.png' % toks
LDA_MODEL_EXCEL_OUTPUT = 'data/model%d_results.xlsx' % toks

#%% load
print('input tokens from preprocessing pipeline %d' % toks)

print('loading DTM from `%s`...' % DATA_PICKLE_DTM)
doc_labels, vocab, dtm, tokens = unpickle_file(DATA_PICKLE_DTM)
assert len(doc_labels) == dtm.shape[0]
assert len(vocab) == dtm.shape[1]
print('loaded DTM with %d documents, %d vocab size, %d tokens' % (len(doc_labels), len(vocab), dtm.sum()))

#%% compute model

print('generating model with parameters:')
pprint(LDA_PARAMS)

model = LDA(**LDA_PARAMS)
model.fit(dtm)

#%% output

print('saving model to `%s`' % LDA_MODEL_PICKLE)
pickle_data((doc_labels, vocab, dtm, model), LDA_MODEL_PICKLE)

print('saving results to `%s`' % LDA_MODEL_EXCEL_OUTPUT)
save_ldamodel_summary_to_excel(LDA_MODEL_EXCEL_OUTPUT, model.topic_word_, model.doc_topic_, doc_labels, vocab, dtm=dtm)

#%%
print('displaying loglikelihoods...')
plt.plot(np.arange(BURNIN, len(model.loglikelihoods_)) * 10, model.loglikelihoods_[BURNIN:])
plt.xlabel('iterations')
plt.ylabel('log likelihood')
plt.savefig(LDA_MODEL_LL_PLOT)
plt.show()

#%%
print('topic-word distribution:')
print('-----')
print_ldamodel_topic_words(model.topic_word_, vocab)
print('-----')

print('document-topic distribution:')
print('-----')
print_ldamodel_doc_topics(model.doc_topic_, doc_labels)
print('-----')
