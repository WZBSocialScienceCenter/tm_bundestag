# -*- coding: utf-8 -*-

import logging
import re
import string

import pandas as pd
from tmtoolkit.preprocess import TMPreproc
from tmtoolkit.utils import pickle_data


SPEECHES_PICKLE_PATH = 'data/speeches_merged.pickle'
DATA_PICKLE_DTM = 'data/speeches_tokens.pickle'

CUSTOM_STOPWORDS = [    # those will be removed
    u'dass',
    u'dafür',
    u'daher',
    u'dabei',
    u'ab',
    u'zb',
    u'usw',
    u'schon',
    u'sowie',
    u'sowieso',
    u'seit',
    u'bereits',
    u'hierfür',
    u'oft',
    u'mehr',
    u'na',
    u'000',
]

CUSTOM_SPECIALCHARS = [   # those will be removed
    u'\u2019',     # ’
    u'\u2018',     # ‘
    u'\u201a',     # ‚
    u'\u201c',     # “
    u'\u201d',     # ”
    u'\u201e',     # „
    u'\u2026',     # …
    u'\u00ad',     # ­
    u'\u00b4',     # ´
    u'\u02bc',     # ʼ
    u'\ufffd',     # �
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
preproc.add_special_chars(CUSTOM_SPECIALCHARS)

print('tokenizing...')
preproc.tokenize()

vocab = preproc.vocabulary
pttrn_token_w_specialchar = re.compile(u'[^A-Za-z0-9ÄÖÜäöüß' + re.escape(string.punctuation) + u']', re.UNICODE)
pttrn_token_w_specialchar_inv = re.compile(u'[A-Za-z0-9ÄÖÜäöüß' + re.escape(string.punctuation) + u']', re.UNICODE)
tokens_w_specialchars = [t for t in vocab if pttrn_token_w_specialchar.search(t)]
uncommon_special_chars = set([pttrn_token_w_specialchar_inv.sub('', t) for t in tokens_w_specialchars])
uncommon_special_chars = set(sum([[c for c in cs] for cs in uncommon_special_chars], []))

print('detected the following uncommon special characters:')
for c in uncommon_special_chars:
    print('%04x -> %s' % (ord(c), c))


print('running preprocessing pipeline...')
preproc.pos_tag()\
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
