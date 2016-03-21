import os
import itertools
import random
import cPickle as pickle
import numpy as np
import matplotlib.pylab as plt
import pretty_midi
import glob
from helpers import getImmediateSubdirectories, openCSV
from helpers import getDistanceFromKey, encodeChromagram
from helpers import unichr2notes, segmentSequence
from pattern_graph import PatternGraph
from pattern_graph_plotting import plotInitialStates, plotEndingStates
from pattern_graph_plotting import plotPatternGraph
from pattern_mining import minePatterns
from params import data_folder, missing_key_csv, broken_csv
plt.ion()

genres = getImmediateSubdirectories(data_folder)
corrupted_files = openCSV(missing_key_csv) + openCSV(broken_csv)

# prune corrupted files
gen_trk_dict = {}
for genre in genres:
    gen_trk_dict[genre] = [x for x in glob.glob(os.path.join(data_folder,
                                                             genre)+'/*.mid')
                           if [x] not in corrupted_files]

rnd_gen = random.choice(genres)
rnd_idx = np.random.randint(0, len(gen_trk_dict[rnd_gen]))

fullpath = gen_trk_dict[rnd_gen][rnd_idx]
fullpath = ('/Users/rafaelvalle/Desktop/research/music_pattern_graphs/'
            'data/midi/Rock - Metal/311 - Down.mid')
print fullpath

# extract and plot chromagram from random file
data = pretty_midi.PrettyMIDI(fullpath)
dist_c = getDistanceFromKey(data.key_signature_changes[0].key_number, 0)

# get chroma as bool and transpose to C
chroma = np.roll(data.get_chroma(times=data.get_beats()).astype(bool),
                 dist_c, axis=0)
segment_ids = segmentSequence(chroma.T, window=4)
plt.figure(figsize=(15, 3))
plt.imshow(chroma,
           interpolation='nearest',
           origin='low',
           aspect='auto',
           cmap=plt.cm.Oranges)

for i in segment_ids:
    plt.axvline(i, linewidth=1, color='b')

plt.savefig('images/chromagram.png')

chroma_unichr = encodeChromagram(chroma.astype(int).T)
chroma_alphabet = set()
[chroma_alphabet.add(i) for i in chroma_unichr]
chroma_unichr_str = ''.join(chroma_unichr)

# instantiate pattern graph and add first symbol as initial state
patt_graph = PatternGraph()
patt_graph.addInitialState(chroma_unichr_str[0])

# instantiate regular expressions and respective connectors
patt_symbs = {'unt': 'T', 'sur': 'S', 'fol': 'F'}
conns = ('|', '|', '|', '|', '|', '')

# iterate through symbol combinations and update pattern graph
for x, y in itertools.combinations(chroma_alphabet, 2):
    patts = ((x, 'unt', y),
             (x, 'sur', y),
             (x, 'fol', y),
             (y, 'unt', x),
             (y, 'sur', x),
             (y, 'fol', x))

    patts_dict = minePatterns(patts, conns, chroma_unichr_str)
    for (a, patt_str, b), count in patts_dict.items():
        patt_graph.addTransition(a, patt_symbs[patt_str], b, count)


# add last symbol as ending state
patt_graph.addEndingState(chroma_unichr[-1])

# parameters for plotting and saving pattern graph
name = 'test'
feature = 'chroma'
min_prob, max_prob = 0.0, 1.0

plotInitialStates(patt_graph, '{}_initial_{}.dot'.format(name, feature),
                  min_prob, max_prob, unichr2notes)

plotEndingStates(patt_graph, '{}_ending_{}.dot'.format(name, feature),
                 min_prob, max_prob, unichr2notes)

plotPatternGraph(patt_graph, '{}_pattern_graph_{}.dot'.format(name, feature),
                 True, min_prob, max_prob, unichr2notes)

pickle.dump(patt_graph, open('{}_{}.pg'.format(name, feature), 'wb'))
