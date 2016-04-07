from __future__ import division
import os
import ntpath
import itertools
import cPickle as pickle
import numpy as np
import pretty_midi
import glob
from sklearn.decomposition import NMF
from helpers import getImmediateSubdirectories, openCSV
from helpers import getDistanceFromKey, encodeChromagram
from helpers import unichr2notes, decodeChromagram
from helpers import buildHistogram, cropEdges
from pattern_graph import PatternGraph
from pattern_graph_plotting import plotInitialStates, plotEndingStates
from pattern_graph_plotting import plotPatternGraph, plotPatternSequence
from pattern_mining import minePatterns, mineUntSurFol
from params import data_folder, patt_gph_folder
from params import patt_gph_img_folder, patt_seq_img_folder
from params import missing_key_csv, broken_csv, special_chrs
from params import train_dic_name, test_dic_name


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

genres = getImmediateSubdirectories(data_folder)
corrupted_files = openCSV(broken_csv)
missing_key_files = np.array(openCSV(missing_key_csv))
train_dic = pickle.load(open(train_dic_name, "rb"))

# prune corrupted files
for genre in train_dic.keys():
    patt_graph = None
    for fullpath in train_dic[genre]:
        print '\n{}\n\tAnalyzing'.format(fullpath)
        # load file and extract chromagram as bool and transpose to C
        data = pretty_midi.PrettyMIDI(fullpath)
        if fullpath in missing_key_files[:, 0]:
            idx = np.where(missing_key_files[:, 0] == fullpath)[0][0]
            dist_c = getDistanceFromKey(int(missing_key_files[idx, 1]), 0)
        else:
            dist_c = getDistanceFromKey(
                                data.key_signature_changes[0].key_number, 0)
        chroma = np.roll(data.get_chroma(times=data.get_beats()).astype(bool),
                         dist_c, axis=0).T

        # remove silence from beginning and end
        chroma = cropEdges(chroma)

        # convert chromagram to unichr alphabet
        chroma_unichr = encodeChromagram(chroma.astype(int))
        chroma_histogram = buildHistogram(chroma_unichr, special_chrs)
        chroma_unichr_str = ''.join(chroma_unichr)

        # instantiate pattern graph and add first symbol as initial state
        if patt_graph is None:
            patt_graph = PatternGraph()

        # update node counts and add initial and ending states and transitions
        patt_graph.updateNodeCounts(chroma_histogram)
        patt_graph.addInitialState(chroma_unichr_str[0])
        patt_graph.addEndingState(chroma_unichr[-1])

        """
        # instantiate regular expressions and respective connectors
        patt_symbs = {'unt': 'T', 'sur': 'S', 'fol': 'F'}
        conns = ('|', '|', '|', '|', '|', '')
        # greedy search, more abstract mining code but slower
        # iterate through symbol combs and update pattern graph transitions
        for x, y in itertools.combinations(chroma_histogram.keys(), 2):
            patts = ((x, 'unt', y),
                    (x, 'sur', y),
                    (x, 'fol', y),
                    (y, 'unt', x),
                    (y, 'sur', x),
                    (y, 'fol', x))
            patts_dict = minePatterns(patts, conns, chroma_unichr_str)
            for (a, patt_str, b), count in patts_dict.items():
                # symbols with length 2 are escaped special chars, e. g. \$
                if len(a) == 2:
                    a = a[1]
                if len(b) == 2:
                    b = b[1]
                patt_graph.addPattern(a, patt_symbs[patt_str], b, count)
        # routine for mine self-transitions in the end
        """
        # linear search
        patt_seq, patt_dic = mineUntSurFol(chroma_unichr_str)

        # update pattern graph
        patt_graph.addInitialPattern(tuple(patt_seq[0]))
        patt_graph.addEndingPattern(tuple(patt_seq[-1]))
        for key, count in patt_dic.items():
            patt_graph.addPattern(key[0], key[1], key[2], count)

        # plot songs pattern sequence
        n_singv = 3  # RGBA

        def decoder(x):
            return decodeChromagram(x).T

        def dim_red(x):
            global n_singv
            return NMF(n_singv).fit_transform(
                x.T.astype(float)).reshape(1, x.shape[1], n_singv)

        print "\tPlotting Chromagram and Pattern Sequence"
        plotPatternSequence(chroma.T, patt_seq, decoder, dim_red,
                            filename=ntpath.basename(fullpath),
                            directory=patt_seq_img_folder)

    print "\tSaving and plotting Pattern Graph"
# parameters for plotting and saving pattern graph
    name = genre
    feature = 'chroma'
    min_prob, max_prob = 0.0, 1.0
    pickle.dump(patt_graph,
                open('{}{}_{}.pg'.format(patt_gph_folder, name, feature), 'wb'))
    plotInitialStates(patt_graph,
                      '{}_initial_{}.dot'.format(name, feature),
                      min_prob, max_prob, unichr2notes,
                      directory=patt_gph_img_folder)

    plotEndingStates(patt_graph,
                     '{}_ending_{}.dot'.format(name, feature),
                     min_prob, max_prob, unichr2notes,
                     directory=patt_gph_img_folder)

    plotPatternGraph(patt_graph,
                     '{}_pattern_graph_{}.dot'.format(name, feature), True,
                     min_prob, max_prob, unichr2notes,
                     directory=patt_gph_img_folder,
                     render=False)
