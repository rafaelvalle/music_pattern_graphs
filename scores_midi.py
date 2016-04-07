from __future__ import division
import os
import ntpath
import cPickle as pickle
import numpy as np
import pretty_midi
import glob
from helpers import getImmediateSubdirectories, openCSV
from helpers import getDistanceFromKey, encodeChromagram
from helpers import buildHistogram, cropEdges
from metrics import computeLikelihood, computeViolationRatio
from pattern_mining import mineUntSurFol
from pattern_graph_plotting import plotViolationMatrix, plotLikelihoodMatrix
from params import data_folder, patt_gph_folder, patt_gph_ext
from params import images_folder
from params import missing_key_csv, broken_csv, special_chrs
from params import test_dic_name


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

genres = getImmediateSubdirectories(data_folder)
corrupted_files = openCSV(broken_csv)
missing_key_files = np.array(openCSV(missing_key_csv))

# load pattern graphs
pg_likelihoods = {}  # {pg genre, file genre}: [[file1,score1], [file2,score2]]
pg_vratios = {}  # {pg genre, file genre}: [[file1,score1], [file2,score2]]
pgs = {}

max_ntrans = 0.0
for fullpath in glob.glob(patt_gph_folder+"*."+patt_gph_ext):
    genre_feature = os.path.splitext(ntpath.basename(fullpath))[0]
    pgs[genre_feature] = pickle.load(open(fullpath, "rb"))
    max_ntrans = max(max_ntrans, pgs[genre_feature].n_trans)

test_dic = pickle.load(open(test_dic_name, "rb"))
for genre in test_dic.keys():
    """
    max_len = 0
    for fullpath in test_dic[genre]:
        max_len = max(len(pretty_midi.PrettyMIDI(fullpath).get_beats()),
                      max_len)
    penalty = 1
    lowest_prob = np.power(1.0 / (max_ntrans + penalty), max_len)
    """

    for fullpath in test_dic[genre]:
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

        # linear search
        patt_seq, patt_dic = mineUntSurFol(chroma_unichr_str)
        print 'Computing Scores (likelihood and violation ratio)'
        for pg_name, pg in pgs.items():
            if (genre, pg_name) not in pg_likelihoods:
                pg_likelihoods[(genre, pg_name)] = []

            if (genre, pg_name) not in pg_vratios:
                pg_vratios[(genre, pg_name)] = []

            pg_vratios[(genre,
                        pg_name)].append((computeViolationRatio(pg, patt_seq),
                                        fullpath))
            pg_likelihoods[(genre,
                            pg_name)].append((computeLikelihood(pg,
                                                                patt_seq),
                                            fullpath))

name = 'midi_files_lklhoods'
pickle.dump(pg_likelihoods,
            open('{}.pgl'.format(name), 'wb'))

name = 'midi_files_vratios'
pickle.dump(pg_vratios,
            open('{}.pgl'.format(name), 'wb'))

plotViolationMatrix(pg_vratios,  'Violation Ratio', filename='vratio_matrix',
                    directory=images_folder)

plotLikelihoodMatrix(pg_likelihoods, 'Log Likelihood',
                     filename='likelihood_matrix',
                     directory=images_folder)
