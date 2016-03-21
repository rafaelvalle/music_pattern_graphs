import os
import csv
from fractions import Fraction
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances as manhat_dist
from sklearn.metrics.pairwise import euclidean_distances as euclid_dist
from sklearn.metrics.pairwise import chi2_kernel as chi2_dist


##############
#  ENCODING  #
##############
def notes2unichr(x):
    notes_dict = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6,
                  'G': 7, 'G#': 8, 'A': 9, 'Bb': 10, 'B': 11}
    bin_array = [0]*12
    for i in x:
        bin_array[notes_dict[i]] = 1

    return unichr(binaryString2int(array2str(bin_array)))


def chroma_str2notes(x):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
    return [notes[i] for i in xrange(12) if x[i] == '1']


def unichr2notes(x, as_string=True):
    if as_string:
        return ' '.join(chroma_str2notes(unichr2chroma_str(x)))
    return chroma_str2notes(unichr2chroma_str(x))


def unichr2chroma_str(x):
    return int2chroma_str(ord(x))


def binaryString2int(x):
    return int(x, 2)


def int2chroma_str(x):
    """Converts an int to a binary Chroma encoding

    Parameters
    ----------
    x : int

    Returns
    -------
    chroma_str : str
        String representing a binary chroma
    """

    chroma_str = "{:b}".format(x)
    return "0"*(12-len(chroma_str))+chroma_str


def chroma_str2chroma(x):
    """Converts an int to a binary Chroma encoding

    Parameters
    ----------
    x : list of strings
        List of chroma strings, e.g. '010101010101'

    Returns
    -------
    chroma : np.array
        Array representing a binary chroma
    """
    return np.array([bool(int(i)) for i in x])


def array2str(x):
    return ''.join(map(str, x))


def encodeChromagram(chromagram):
    return map(unichr, map(binaryString2int, map(array2str, chromagram)))


def decodeChromagram(chromagram, as_str=False):
    """Decodes an list of int encoded chromagram to binary chromagram"

    Parameters
    ----------
    chromaint : int list
        List of chromagrams encoded as int using the encodeChromagram method
    is_unichar : bool
        Is chromagram encoded as unichar
    as_str : boolean
        Decode into list of strings instead of traditional chromagrams

    Returns
    -------
    chroma : str list
        List of string binary encoding of chromagram
    """
    chroma = [int2chroma_str(j) for j in [ord(i) for i in chromagram]]

    if as_str:
        return chroma

    return np.array([chroma_str2chroma(i) for i in chroma]).T


#######################
#  UTILITY FUNCTIONS  #
#######################
def getFractions(vals):
    return np.array([Fraction(val) for val in vals])


def round2nearest(vals, fraction):
    return np.round(np.array(vals, dtype=float) / fraction) * fraction


def getImmediateSubdirectories(folder_path):
    return [name for name in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, name))]


###########
#  MUSIC  #
###########
def getDistanceFromKey(key_num, key=0):
    if key_num > 11:
        return key - ((key_num + 3) % 12)
    return key - key_num


def segmentSequence(seq, window=4, dist_func=euclid_dist, thresh=2.5):
    if len(seq) > window:
        prv_txtr = seq[0:window].ravel()

    segment_ids = []
    for i in range(0+window, (len(seq)/window) * window, window):
        cur_txtr = seq[i:i+window].ravel()
        if euclid_dist(prv_txtr, cur_txtr, squared=True) > thresh:
            segment_ids.append(i)
            prv_txtr = cur_txtr

    # do last window if necessary
    if len(seq) % window != 0:
        cur_txtr = seq[(len(seq)/window) * window]

    return segment_ids


def getPhraseStarts(vals, window=4):
    """Naive implementation of phrase segmentation
    Parameters
    ----------
    vals : numpy array
        Rows as features cols as timesteps
    window : int
        Window size in timesteps
    """

    results = np.zeros(vals.shape[1], dtype=bool)
    results[0] = True

    for i in xrange(1, vals.shape[1] - window):
        results[i] = abs(vals[:, i].mean() - vals[:, i:i+window].mean()) > \
                         vals[:, i:window].std() and not results[i-1]
    return results


##########
#  MISC  #
##########
def Zero():
    return 0.0


def writeCSV(filename, items, delimiter=','):
    with open(filename, 'wb') as f:
        wr = csv.writer(f, delimiter=delimiter, quoting=csv.QUOTE_ALL)
        for item in items:
            wr.writerow([item])


def openCSV(filename):
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        return list(reader)