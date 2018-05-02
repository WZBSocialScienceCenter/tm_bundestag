# -*- coding: utf-8 -*-

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tmtoolkit.utils import unpickle_file

pd.set_option('display.width', 180)

#%%

doc_labels, vocab, dtm, model = unpickle_file('data/model2.pickle')

speeches_merged = unpickle_file('data/speeches_merged.pickle')

mdb = pd.read_csv('data/offenesparlament-mdb.csv', usecols=['id', 'first_name', 'last_name', 'party'])

first_name_unicode = mdb.first_name.str.lower().str.decode('utf-8').values.astype(np.unicode_)
last_name_unicode = mdb.last_name.str.lower().str.decode('utf-8').values.astype(np.unicode_)


mdb['speaker_fp'] = np.core.defchararray.add(np.core.defchararray.add(first_name_unicode, u'-'), last_name_unicode)
for c1, c2 in zip(u'äöüé ', u'aoue-'):
    mdb['speaker_fp'] = mdb['speaker_fp'].str.replace(c1, c2)

#mdb['speaker_fp'] = mdb.profile_url.str.extract(r'/([a-z0-9-]+)$')

assert sum(mdb['speaker_fp'].isna()) == 0

#%%
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

#%%

doc_meta = pd.merge(doc_meta, speeches_merged[['sequence', 'speaker_key']],
                    left_on='seq_id', right_on='sequence', how='left')

doc_meta = pd.merge(doc_meta, mdb, on='speaker_fp', how='left')
doc_meta['mdb_speaker_key'] = doc_meta.id.fillna(-1, downcast='infer')
del doc_meta['id']

doc_meta_gaps = doc_meta.loc[doc_meta.party.isna() & (doc_meta.speaker_key != -1), ['merged_speech_id', 'speaker_key']]
doc_meta_gaps = pd.merge(doc_meta_gaps, mdb, left_on='speaker_key', right_on='id', how='left')
del doc_meta_gaps['id']

doc_meta_gaps.set_index('merged_speech_id', verify_integrity=True, inplace=True)
doc_meta.set_index('merged_speech_id', verify_integrity=True, inplace=True)

doc_meta.update(doc_meta_gaps)

n_no_mdb_data = sum(doc_meta.party.isna())

print('no MDB data found for %d speeches from %d overall speeches' % (n_no_mdb_data, len(doc_meta)))


#%%
avg_topics_per_party = {}
for party, grp in doc_meta.groupby('party'):
    party_speeches_ind = np.where(np.isin(doc_labels, grp.doc_label))[0]

    party_doc_topic = model.doc_topic_[party_speeches_ind, :]
    avg_topics_per_party[party] = np.sum(party_doc_topic, axis=0) / len(party_doc_topic)
    assert np.isclose(avg_topics_per_party[party].sum(), 1)

#%%
n_parties = len(avg_topics_per_party)
n_top = 5
fig, axes = plt.subplots(n_parties, 1, sharex=True, figsize=(8, 2 * n_parties))

for i, (party, ax) in enumerate(zip(sorted(avg_topics_per_party.keys()), axes)):
    #print(party)
    theta = avg_topics_per_party[party]
    top_topics_ind = np.argsort(theta)[::-1][:n_top]
    ypos = np.arange(n_top)[::-1]
    ax.barh(ypos, theta[top_topics_ind], color='lightgray')
    ax.set_yticks(ypos)
    ax.set_yticklabels([u'topic %d' % (t+1) for t in top_topics_ind])
    ax.set_title(party.decode('utf-8'))

    if i == n_parties-1:
        ax.set_xlabel(u'proportion')

plt.show()