# -*- coding: utf-8 -*-
"""
Prepare the raw data: Load the CSV files for each session and merge the speeches for each speaker.

Markus Konrad <markus.konrad@wzb.eu>
"""

import os

import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_SEPARATE_PICKLE_PATH = 'data/speeches_separate.pickle'
OUTPUT_MERGED_PICKLE_PATH = 'data/speeches_merged.pickle'

RAW_DATA_PATH = 'data/offenesparlament-sessions-csv'
SESS_COLUMNS = (
    'sequence',
    'sitzung',
#    'speaker_cleaned',   # not reliable. # TODO: load from MDB data?
    'speaker_fp',
    'speaker_key',        # not reliable (many NAs)
#    'speaker_party',     # not reliable. # TODO: load from MDB data?
    'text',
    'top',
    'top_id',
    'type'
)


#
# load raw data: CSV files with parlament debates
#

parl_speeches_parts = []
for fname in sorted(os.listdir(RAW_DATA_PATH)):
    fpath = os.path.join(RAW_DATA_PATH, fname)
    if fname.endswith('.csv') and os.path.isfile(fpath):
        print('reading CSV file `%s`' % fpath)
        sess_df = pd.read_csv(fpath, index_col='id', usecols=range(1, 15), encoding='utf-8')

        # filter observations:
        # - only speeches
        # - only those with text (2 times missing text -- probably an error in the data)
        # - exclude session 191 (this session was not coded!  -- probably an error in the data)
        # filter variables: use only columns defined in SESS_COLUMNS
        sess_df = sess_df.loc[(sess_df.type == 'speech') & (~sess_df.text.isnull() & (sess_df.sitzung != 191)),
                              SESS_COLUMNS]
        # sess_df = sess_df.loc[(~sess_df.text.isnull() & (sess_df.sitzung != 191)), SESS_COLUMNS]


        parl_speeches_parts.append(sess_df)

parl_speeches_df = pd.concat(parl_speeches_parts)

# set missing TOP IDs to -1
parl_speeches_df.top_id.fillna(-1, inplace=True, downcast='infer')

# set missing speaker IDs to -1
parl_speeches_df.speaker_key.fillna(-1, inplace=True, downcast='infer')

# check NAs
assert sum(parl_speeches_df.sitzung.isnull()) == 0
assert sum(parl_speeches_df.top_id.isnull()) == 0
assert sum(parl_speeches_df.type.isnull()) == 0
assert sum(parl_speeches_df.text.isnull()) == 0
assert sum(parl_speeches_df.speaker_key.isnull()) == 0
assert sum(parl_speeches_df.speaker_fp.isnull()) == 0

print('loaded %d speech records' % len(parl_speeches_df))

print('top_id missings: %d' % sum(parl_speeches_df.top_id == -1))
print('speaker_key missings: %d' % sum(parl_speeches_df.speaker_key == -1))

speech_lengths = parl_speeches_df.text.str.len()

print('speeches length properties:')
print('> range %d - %d' % (speech_lengths.min(), speech_lengths.max()))
print('> mean %f' % speech_lengths.mean())
print('> median %f' % speech_lengths.median())

plt.figure()
speech_lengths.plot('hist', title='Unprocessed speech lengths in num. chars', bins=100)
plt.show(block=False)

plt.figure()
speech_lengths.plot('hist', title='Unprocessed speech lengths in num. chars between [0, 2000]', bins=400,
                    xlim=(0, 2000))
plt.show(block=False)

# merge speeches

def only_value_from_series(ser):
    uniques = ser.unique()
    assert len(uniques) == 1
    return uniques[0]


def typed_series(name, val, dtype=None):
    return pd.Series(data=[val] if type(val) not in (list, tuple) else val, name=name, dtype=dtype)


speeches_groups = []
# grouping with speaker_fp instead of speaker ID because of many missings in speaker ID
parl_speeches_grouped = parl_speeches_df.groupby(('sitzung', 'speaker_fp', 'top_id'), sort=False)
for seq, (gname, gspeech) in enumerate(parl_speeches_grouped):
    sitzung, speaker_fp, top_id = gname
    print('merging speech %d (sitzung %d, speaker %s, top %d)' % (seq+1, sitzung, speaker_fp, top_id))
    pars = '\n\n'.join(gspeech.text)

    speeches_groups.append(pd.DataFrame([
        typed_series('sequence', seq+1, int),
        typed_series('orig_sequences', ','.join(map(str, gspeech.sequence))),
        typed_series('n_interruptions', len(gspeech.sequence)-1, int),
        typed_series('sitzung', sitzung, int),
#        typed_series('speaker_cleaned', only_value_from_series(gspeech.speaker_cleaned), str),
#        typed_series('speaker_fp', only_value_from_series(gspeech.speaker_fp), str),
        typed_series('speaker_fp', speaker_fp),
        typed_series('speaker_key', only_value_from_series(gspeech.speaker_key)),
#        typed_series('speaker_party', only_value_from_series(gspeech.speaker_party), str),
        typed_series('text', pars),
        typed_series('top_id', top_id, int),
        typed_series('top', only_value_from_series(gspeech.top))
    ]))


# no idea why "pd.concat(speeches_groups, ignore_index=True)" does not work but this works:
speeches_merged_df = pd.concat(speeches_groups, ignore_index=True, axis=1).T

print('%d merged speeches' % len(speeches_merged_df))

speeches_merged_lengths = speeches_merged_df.text.str.len()

print('merged speeches length properties:')
print('> range %d - %d' % (speeches_merged_lengths.min(), speeches_merged_lengths.max()))
print('> mean %f' % speeches_merged_lengths.mean())
print('> median %f' % speeches_merged_lengths.median())

plt.figure()
speeches_merged_lengths.plot('hist', title='Lengths of merged speeches in num. chars', bins=100)
plt.show(block=False)

plt.figure()
speeches_merged_df.n_interruptions.plot('hist', title='Num. of interruptions', bins=50)
plt.show(block=False)

print('saving separate (original) speeches to `%s`' % OUTPUT_MERGED_PICKLE_PATH)
parl_speeches_df.to_pickle(OUTPUT_SEPARATE_PICKLE_PATH)

print('saving merged speeches to `%s`' % OUTPUT_MERGED_PICKLE_PATH)
speeches_merged_df.to_pickle(OUTPUT_MERGED_PICKLE_PATH)

plt.show()  # block

print('done.')