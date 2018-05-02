# -*- coding: utf-8 -*-

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tmtoolkit.utils import unpickle_file
from tmtoolkit.topicmod.model_stats import get_most_relevant_words_for_topic, get_topic_word_relevance, \
    get_doc_lengths, exclude_topics


pd.set_option('display.width', 180)

#%% load data

# model and DTM
doc_labels, vocab, dtm, model = unpickle_file('data/model2.pickle')

n_docs, n_topics = model.doc_topic_.shape
_, n_vocab = model.topic_word_.shape

assert n_docs == len(doc_labels) == dtm.shape[0]
assert n_topics == model.topic_word_.shape[0]
assert n_vocab == len(vocab) == dtm.shape[1]

print('loaded model with %d documents, vocab size %d, %d tokens and %d topics'
      % (n_docs, n_vocab, dtm.sum(), n_topics))

# raw speeches
speeches_merged = unpickle_file('data/speeches_merged.pickle')

# MDB data
mdb = pd.read_csv('data/offenesparlament-mdb.csv', usecols=['id', 'first_name', 'last_name', 'party'])

first_name_unicode = mdb.first_name.str.lower().str.decode('utf-8').values.astype(np.unicode_)
last_name_unicode = mdb.last_name.str.lower().str.decode('utf-8').values.astype(np.unicode_)

mdb['speaker_fp'] = np.core.defchararray.add(np.core.defchararray.add(first_name_unicode, u'-'), last_name_unicode)
for c1, c2 in zip(u'äöüé ', u'aoue-'):
    mdb['speaker_fp'] = mdb['speaker_fp'].str.replace(c1, c2)

#mdb['speaker_fp'] = mdb.profile_url.str.extract(r'/([a-z0-9-]+)$')

assert sum(mdb['speaker_fp'].isna()) == 0

#%% prepare model

exclude_topic_indices = np.array([2, 19, 72, 9, 114, 18, 87, 30, 16, 6])
print('excluding %d topics: %s' % (len(exclude_topic_indices), exclude_topic_indices+1))
theta, phi = exclude_topics(exclude_topic_indices, model.doc_topic_, model.topic_word_)

n_topics = theta.shape[1]
assert phi.shape[0] == n_topics
del model

#%% meta data belonging to speeches

# this is not so straight-forward because unfortunately there are many missing speaker_key values for the speeches,
# hence it is difficult to match speeches with speakers from the MDB data

pttrn_doc_label = re.compile(u'(\d+)_sess(\d+)_top([\d-]+)_spk_([a-zß0-9-]+)_seq(\d+)', re.UNICODE)

doc_meta = []
for dl in doc_labels:
    m = pttrn_doc_label.search(dl)

    doc_meta.append({
        'doc_label': dl,
        'merged_speech_id': int(m.group(1)),
        'sess_id': int(m.group(2)),
        'top_id': int(m.group(3)),
        'speaker_fp': m.group(4),
        'seq_id': int(m.group(5))
    })

doc_meta = pd.DataFrame(doc_meta)

# left join with raw speeches to get "speaker_key"
doc_meta = pd.merge(doc_meta, speeches_merged[['sequence', 'speaker_key']],
                    left_on='seq_id', right_on='sequence', how='left')

# left join using "speaker_fp" will work in most cases
doc_meta = pd.merge(doc_meta, mdb, on='speaker_fp', how='left')
doc_meta['mdb_speaker_key'] = doc_meta.id.fillna(-1, downcast='infer')
del doc_meta['id']

# leaves about 1100 speeches not merged -> try to use the "speaker_key" from the raw speeches data now
doc_meta_gaps = doc_meta.loc[doc_meta.party.isna() & (doc_meta.speaker_key != -1), ['merged_speech_id', 'speaker_key']]
doc_meta_gaps = pd.merge(doc_meta_gaps, mdb, left_on='speaker_key', right_on='id', how='left')
del doc_meta_gaps['id']

doc_meta_gaps.set_index('merged_speech_id', verify_integrity=True, inplace=True)
doc_meta.set_index('merged_speech_id', verify_integrity=True, inplace=True)

doc_meta.update(doc_meta_gaps)

n_no_mdb_data = sum(doc_meta.party.isna())

print('no MDB data found for %d speeches from %d overall speeches' % (n_no_mdb_data, len(doc_meta)))


#%% calculate average topic proportion per party

stats_per_party = {}
for party, grp in doc_meta.groupby('party'):
    party_speeches_ind = np.where(np.isin(doc_labels, grp.doc_label))[0]

    party_doc_topic = theta[party_speeches_ind, :]
    avg_party_theta = np.sum(party_doc_topic, axis=0) / len(party_doc_topic)
    assert np.isclose(avg_party_theta.sum(), 1)

    stats_per_party[party] = (avg_party_theta, len(grp))


#%% plot average topic proportion per party

n_parties = len(stats_per_party)
n_top_topics = 5
n_top_words = 8
fig, axes = plt.subplots(n_parties, 1, sharex=True, figsize=(8, 2 * n_parties), constrained_layout=True)

fig.suptitle(u'Top %d average topic proportions per party' % n_top_topics, fontsize='large')
fig.subplots_adjust(top=0.925)

topic_word_rel_mat = get_topic_word_relevance(phi, theta, get_doc_lengths(dtm), lambda_=0.6)

for i, (party, ax) in enumerate(zip(sorted(stats_per_party.keys()), axes)):
    theta_party, n_speeches_party = stats_per_party[party]
    top_topics_ind = np.argsort(theta_party)[::-1][:n_top_topics]
    ypos = np.arange(n_top_topics)[::-1]

    ax.barh(ypos, theta_party[top_topics_ind], color='lightgray')
    ax.set_yticks(ypos)
    ax.set_yticklabels([u'topic %d' % (t+1) for t in top_topics_ind])
    ax.set_title(u'%s (N=%d)' % (party.decode('utf-8'), n_speeches_party), fontsize='medium')

    for y, t in zip(ypos, top_topics_ind):
        most_rel_words = get_most_relevant_words_for_topic(vocab, topic_word_rel_mat, t, n_top_words)
        ax.text(0.002, y-0.15, ', '.join(most_rel_words), fontsize='xx-small')

    if i == n_parties-1:
        ax.set_xlabel(u'average proportion')

plt.savefig('fig/top_topics_per_party.png')
plt.show()