import re
from collections import defaultdict
import numpy as np
from helpers import Zero


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

# valid patterns
re_dict = {}
re_dict['fol'] = lambda symbs: ur"({x}{y})".format(x=symbs[0], y=symbs[1])
re_dict['sur'] = lambda symbs: ur"({x}{y}{x})".format(x=symbs[0], y=symbs[1])
re_dict['unt'] = lambda symbs: ur"({x}{x}+{y})".format(x=symbs[0], y=symbs[1])


def mineUntSurFol(seq):
    patt_seq = []
    patt_dic = defaultdict(Zero)

    if len(seq) == 2:
        patt_dic[(seq[0], 'F', seq[1])] += 1
        patt_seq.append((seq[0], 'F', seq[1]))
        return [key], patt_dic

    i = 0
    while i < len(seq) - 1:
        if seq[i] == seq[i+1]:
            if i < len(seq)-2:
                if seq[i+2] != seq[i]:
                    patt_seq.append((seq[i], 'T', seq[i+2]))
                    patt_dic[(seq[i], 'T', seq[i+2])] += 1
                    i += 2
                else:
                    i += 1
            else:
                patt_seq.append((seq[i], 'T', seq[i+1]))
                patt_dic[(seq[i], 'T', seq[i+1])] += 1
                i += 1
        else:
            if i < len(seq) - 2 and seq[i] == seq[i+2]:
                patt_seq.append((seq[i], 'S', seq[i+1]))
                patt_dic[(seq[i], 'S', seq[i+1])] += 1
                i += 2
            else:
                patt_seq.append((seq[i], 'F', seq[i+1]))
                patt_dic[(seq[i], 'F', seq[i+1])] += 1
                i += 1
    return (np.array(patt_seq, dtype=object), patt_dic)


def minePatterns(patts, conns, word):
    global re_dict
    """ Given a list of non-instantiated regular expression and their respective
    symbols, instantiate a regular expression and returns all matches in word

    Parameters
    ----------
    patts : str tuple (origin, pattern, target)
        Tuple containing symbol of origin, string representing pattern to mine
        and symbol of target, e. g. ('x', 'fol', 'y')


    conns: list of strings
        List of strings connectors to connect the regular expressions

    word : str
        Sequence of characters in which search will be performed

    Returns
    -------
        int np.ndarray,
        Array where indices represent the index of the regular
        expression mined and values indicate number of matches found

    """

    re_str = r""
    matches_dict = {}
    for i in range(len(patts)):
        re_str += re_dict[patts[i][1]]((patts[i][0], patts[i][2]))
        re_str += conns[i]

    matches = np.sum(np.array(re.findall(re_str, word, re.UNICODE),
                              dtype=bool), axis=0)

    # return empty dictionary if no matches are found
    if not isinstance(matches, np.ndarray):
        return {}

    for i in range(len(patts)):
        matches_dict[(patts[i][0], patts[i][1], patts[i][2])] = matches[i]

    return matches_dict
