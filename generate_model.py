# -*- coding: utf-8 -*-
"""
Topic Modeling for 18th German Bundestag debates. Model generation with best model parameters found by model selection
in tm_eval.py

Best model parameters:
- k = 80 topics
- alpha = 50/k = 0.625
- eta = 0.5
- 1500 iterations
"""

from __future__ import division
from pprint import pprint
import logging

import matplotlib.pyplot as plt
from lda import LDA
from tmtoolkit.lda_utils.eval_metrics import metric_arun_2010, metric_cao_juan_2009, metric_griffiths_2004
from tmtoolkit.lda_utils.common import print_ldamodel_doc_topics, print_ldamodel_topic_words, \
    save_ldamodel_summary_to_excel
from tmtoolkit.utils import unpickle_file, pickle_data


# model hyperparameters

K = 80
LDA_PARAMS = dict(
    n_topics=K,
    alpha=50/K,
    eta=0.5,
    n_iter=1500
)

# other parameters
BURNIN = 5   # with a default of refresh=10 this means 50 burnin iterations

# paths to data files

DATA_PICKLE_DTM = 'data/speeches_tokens.pickle'
LDA_MODEL_PICKLE = 'data/final_model.pickle'
LDA_MODEL_LL_PLOT = 'data/final_model_logliks.png'
LDA_MODEL_EXCEL_OUTPUT = 'data/final_model_results.xlsx'


print('loading DTM...')
doc_labels, vocab, dtm = unpickle_file(DATA_PICKLE_DTM)
assert len(doc_labels) == dtm.shape[0]
assert len(vocab) == dtm.shape[1]
print('loaded DTM with %d documents, %d vocab size, %d tokens' % (len(doc_labels), len(vocab), dtm.sum()))

print('generating model with parameters:')
pprint(LDA_PARAMS)

model = LDA(**LDA_PARAMS)
model.fit(dtm)

print('saving model to `%s`' % LDA_MODEL_PICKLE)
pickle_data((doc_labels, vocab, dtm, model), LDA_MODEL_PICKLE)

print('saving results to `%s`' % LDA_MODEL_EXCEL_OUTPUT)
save_ldamodel_summary_to_excel(LDA_MODEL_EXCEL_OUTPUT, model.topic_word_, model.doc_topic_, doc_labels, vocab, dtm=dtm)

print('displaying loglikelihoods...')
plt.plot(model.loglikelihoods_[BURNIN:])
plt.savefig(LDA_MODEL_LL_PLOT)
plt.show()

print('calculating model evaluation metrics...')
metric_arun = metric_arun_2010(model.topic_word_, model.doc_topic_, dtm.sum(axis=1))
metric_cao = metric_cao_juan_2009(model.topic_word_)
metric_griffiths = metric_griffiths_2004(model.loglikelihoods_[BURNIN:])
print('Arun: %f, Cao: %f, Griffiths: %d' % (metric_arun, metric_cao, metric_griffiths))

print('topic-word distribution:')
print('-----')
print_ldamodel_topic_words(model.topic_word_, vocab)
print('-----')

print('document-topic distribution:')
print('-----')
print_ldamodel_doc_topics(model.doc_topic_, doc_labels)
print('-----')
