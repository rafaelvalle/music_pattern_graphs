import re
import numpy as np


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

# valid patterns
re_dict = {}
re_dict['fol'] = lambda symbs: u"({x}{y})".format(x=symbs[0], y=symbs[1])
re_dict['sur'] = lambda symbs: u"({x}{y}{x})".format(x=symbs[0], y=symbs[1])
re_dict['unt'] = lambda symbs: u"({x}{x}+{y})".format(x=symbs[0], y=symbs[1])


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
