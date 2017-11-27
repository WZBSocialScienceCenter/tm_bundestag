# -*- coding: utf-8 -*-

import logging

import pandas as pd
from tmtoolkit.preprocess import TMPreproc
from tmtoolkit.utils import pickle_data


SPEECHES_PICKLE_PATH = 'data/speeches_merged.pickle'
DATA_PICKLE_DTM = 'data/speeches_tokens.pickle'

CUSTOM_STOPWORDS = [
    u'dass',
    u'dafür',
    u'daher',
    u'dabei',
    u'ab',
    u'zb',
    u'sowie',
    u'seit',
    u'hierfür',
    u'oft',
    u'mehr',
]

logging.basicConfig(level=logging.INFO)
tmtoolkit_log = logging.getLogger('tmtoolkit')
tmtoolkit_log.setLevel(logging.INFO)
tmtoolkit_log.propagate = True

print('loading speeches from `%s`' % SPEECHES_PICKLE_PATH)
speeches_df = pd.read_pickle(SPEECHES_PICKLE_PATH)
print('loaded %d speeches' % len(speeches_df))

corpus = {}
for speech_id, speech in speeches_df.iterrows():
    doc_label = '%d_sess%d_top%d_spk%d_seq%d' % (speech_id, speech.sitzung, speech.top_id, speech.speaker_key, speech.sequence)
    corpus[doc_label] = speech.text

assert len(corpus) == len(speeches_df)

print('preparing corpus...')
preproc = TMPreproc(corpus, language='german')
preproc.add_stopwords(CUSTOM_STOPWORDS)

print('running preprocessing pipeline...')
preproc.tokenize()\
       .pos_tag()\
       .lemmatize()\
       .tokens_to_lowercase()\
       .remove_special_chars_in_tokens()\
       .clean_tokens(remove_shorter_than=2)\
       .remove_common_tokens(0.9)\
       .remove_uncommon_tokens(3, absolute=True)

print('generating DTM...')
doc_labels, vocab, dtm = preproc.get_dtm()

pickle_data((doc_labels, vocab, dtm), DATA_PICKLE_DTM)
print('done.')
