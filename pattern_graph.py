from collections import defaultdict
from helpers import Zero

def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


class Node(object):
    def __init__(self, label, count=.0, count_initial=.0, count_ending=.0):
        self.label = label
        self.count = float(count)
        self.count_initial = float(count_initial)
        self.count_ending = float(count_ending)
        self.transitions = defaultdict(Zero)  # (pattern, symbol) -> count

    def addTransition(self, patt, state, cnt=1.0):
        if cnt > 0:
            self.incrementCount(cnt)
            self.transitions[(patt, state)] += cnt

    def incrementCount(self, amount=1):
        self.count += amount

    def getPriorProbability(self, n_trans):
        return self.count / n_trans

    def getTransitionProbability(self, pattern, state):
        return self.transitions[(pattern, state)] / self.count


class PatternGraph(object):
    def __init__(self):
        self.initial_states = set()
        self.ending_states = set()

        # number of events seen so far
        self.n_trans = 0

        # dictionary to store node objects
        self.nodes_dict = {}

    def __repr__(self):
        rep = 'PatternGraph'
        rep += '\nInitial states Q0: %s' % str(self.initial_states)
        rep += '\nEnding states F : %s' % str(self.ending_states)
        rep += '\nTransitions: (state, pattern, state)'
        for node_lbl, node in self.nodes_dict.items():
            for patt_lbl, trans_lbl, in node.transitions.keys():
                rep += ('\n  ({} {} {})'
                        '%.5f'.format(node_lbl, patt_lbl, trans_lbl,
                                      node.getTransitionProbability(patt_lbl,
                                                                    trans_lbl)))
        return rep

    def addTransition(self, state_orig, patt, state_dest, cnt=1.0):
        if cnt > 0:
            self.n_trans += 1.0
            self.addState(state_orig)
            self.addState(state_dest)
            self.nodes_dict[state_orig].addTransition(patt, state_dest, cnt)

    def addState(self, state):
        if state not in self.nodes_dict:
            self.nodes_dict[state] = Node(state)

    def addInitialState(self, state):
        self.initial_states.add(state)

    def addEndingState(self, state):
        self.addState(state)
        self.nodes_dict[state].incrementCount()
        self.ending_states.add(state)
