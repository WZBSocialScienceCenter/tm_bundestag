# -*- coding: utf-8 -*-

import sys
import logging
import re
import string

import pandas as pd
from tmtoolkit.preprocess import TMPreproc
from tmtoolkit.utils import pickle_data


DATA_PICKLE_DTM = 'data/speeches_tokens_%d.pickle'

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

logging.basicConfig(level=logging.DEBUG)
tmtoolkit_log = logging.getLogger('tmtoolkit')
tmtoolkit_log.setLevel(logging.DEBUG)
tmtoolkit_log.propagate = True

if len(sys.argv) == 2:
    preproc_mode = int(sys.argv[1])
else:
    preproc_mode = None

if preproc_mode is None or not 0 <= preproc_mode <= 2:
    print('call script as: %s <preprocessing pipeline>' % sys.argv[0])
    print('where preprocessing pipeline is:')
    print('  0 -> use separate speech parts, default pipeline')
    print('  1 -> use merged speeches, default pipeline')
    print('  2 -> use merged speeches, remove salutatory addresses, default pipeline')
    exit(1)

print('preprocessing mode %d' % preproc_mode)

if preproc_mode == 0:
    speeches_pickle = 'data/speeches_separate.pickle'
else:
    speeches_pickle = 'data/speeches_merged.pickle'

print('loading speeches from `%s`' % speeches_pickle)
speeches_df = pd.read_pickle(speeches_pickle)
print('loaded %d speeches' % len(speeches_df))

if preproc_mode == 2:
    # remove salutatory address:
    # "Herr Präsident! Sehr geehrte Kolleginnen und Kollegen! Meine Damen und Herren! Ich will zum Schluss ..."
    # -> "Ich will zum Schluss ..."

    print('removing salutations...')

    pttrn_salutation = re.compile(r'^.+!\s+')
    def remove_salutations(text):
        m = pttrn_salutation.match(text)
        if m:
            text = text[m.end(0):]
            assert text
        return text

    speeches_df['text'] = speeches_df.text.apply(remove_salutations)

    CUSTOM_STOPWORDS += [u'sagen', u'geben', u'm\xfcssen', u'stehen', u'sehen', u'gehen', u'nat\xfcrlich', u'ganz',
                         u'lassen', u'h\xf6ren', u'gerade', u'daran', u'eben', u'denen', u'immer', u'deshalb',
                         u'finden', u'tun', u'geben', u'genau', u'sollen', u'deutlich', u'kommen', u'n\xe4mlich',
                         u'sprechen', u'legen', u'halten', u'bringen', u'f\xfchren', u'darauf', u'darum', u'dar\xfcber',
                         u'gro\xdf', u'diskutieren', u'denken', u'davon', u'vielmehr', u'letzter', u'insbesondere',
                         u'glauben', u'vielleicht', u'bleiben', u'gar', u'genug', u'erst', u'schauen', u'\xfcbrig',
                         u'zeigen', u'teil', u'teilweise', u'sicht', u'einfach', u'fallen',
                         u'entscheidend', u'stellen', u'wesentlich', u'd\xfcrfen',
                         u'weder', u'kaum', u'reden', u'sicherlich', u'liegen', u'angehen',
                         u'wort', u'wissen', u'bisher', u'bestehen', u'trotzdem', u'klar',
                         u'wichtig', u'sogar', u'deswegen', u'l\xe4sst',
                         u'kennen', u'genauso', u'sowohl', u'ausdr\xfccklich', u'zumindest',
                         u'wirklich', u'kurz', u'brauchen', u'\xfcberhaupt',
                         u'unserer', u'nehmen', u'setzen', u'm\xf6glich',
                         u'gesamt', u'wenig', u'jedenfalls', u'viel',
                         u'ansprechen', u'besonders', u'nennen',
                         u'erster', u'au\xdferdem', u'versuchen',
                         u'allein', u'angesichts', u'hoffen', u'viele', u'fast',
                         u'vorstellen', u'aufgrund', u'eigentlich', u'hinaus',
                         u'gleichzeitig', u'laufen', u'wenige', u'notwendig',
                         u'nachdenken', u'vieles', u'lange', u'deren', u'statt',
                         u'daneben', u'beispielsweise',
                         u'ebenfalls', u'vielen', u'ganze', u'au\xdfer', u'zur\xfcck', u'ziemlich',
                         u'weiterhin', u'm\xf6chten', u'dagegen', u'beispiel', u'\xfcbrigens', u'einzig', u'beim',
                         u'darin', u'innerhalb', u'daraus', u'dadurch', u'allerdings']

print('preparing corpus...')
corpus = {}
for speech_id, speech in speeches_df.iterrows():
    doc_label = '%d_sess%d_top%d_spk_%s_seq%d' % (speech_id, speech.sitzung, speech.top_id,
                                                 speech.speaker_fp, speech.sequence)
    corpus[doc_label] = speech.text

assert len(corpus) == len(speeches_df)

print('starting preprocessing...')
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
    print('%04x' % ord(c))


print('running preprocessing pipeline...')
preproc.pos_tag()\
       .lemmatize()\
       .tokens_to_lowercase()\
       .remove_special_chars_in_tokens()\
       .clean_tokens(remove_shorter_than=2)\
       .remove_common_tokens(0.9)\
       .remove_uncommon_tokens(3, absolute=True)

print('retrieving tokens...')
tokens = preproc.tokens

print('generating DTM...')
doc_labels, vocab, dtm = preproc.get_dtm()

output_dtm_pickle = DATA_PICKLE_DTM % preproc_mode

print('writing DTM to `%s`...' % output_dtm_pickle)
pickle_data((doc_labels, vocab, dtm, tokens), output_dtm_pickle)
print('done.')
