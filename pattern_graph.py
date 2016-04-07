from __future__ import division


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


class Node(object):
    def __init__(self, label, count=.0, n_trans=.0):
        self.label = label
        self.count = count
        self.n_trans = n_trans
        self.transitions = {}  # (pattern, state) -> count

    def addTransition(self, patt, state, cnt=1.0):
        if cnt > 0:
            self.incrementTransCount(cnt)
            if (patt, state) not in self.transitions:
                self.transitions[(patt, state)] = 0
            self.transitions[(patt, state)] += cnt

    def incrementCount(self, amount=1):
        self.count += amount

    def incrementTransCount(self, cnt=1.0):
        self.n_trans += cnt

    def getNodeProbability(self, n_events):
        return self.count / n_events

    def getTransitionProbability(self, pattern, state):
        if (pattern, state) not in self.transitions:
            return 0
        return self.transitions[(pattern, state)] / self.n_trans

    def getPatternCount(self, pattern, state):
        if (pattern, state) not in self.transitions:
            return 0
        return self.transitions[(pattern, state)]

    def hasTransition(self, pattern, state):
        return (pattern, state) in self.transitions


class PatternGraph(object):
    def __init__(self):
        self.initial_states = set()
        self.ending_states = set()

        self.initial_patterns = PatternCollection()
        self.ending_patterns = PatternCollection()

        # number of events seen so far
        self.n_trans = 0
        self.n_events = 0

        # dictionary to store node objects
        self.nodes_dict = {}

        self.cur_state = None
        self.prev_state = None

    def __repr__(self):
        rep = 'PatternGraph'
        rep += '\nInitial states Q0: {}'.format(self.initial_states)
        rep += '\nEnding states F : {}'.format(self.ending_states)
        rep += '\nNumber of transitions : {}'.format(self.n_trans)
        rep += '\nNumber of events: {}'.format(self.n_events)
        """
        rep += '\nNodes {}'.format(self.nodes_dict.keys())
        rep += '\nPatterns: (state, pattern, state)'
        for node_lbl, node in self.nodes_dict.items():
            for patt_lbl, trans_lbl, in node.transitions.keys():
                rep += (u'\n  ({} {} {}) '
                        '{:.3f}'.format([node_lbl], patt_lbl, [trans_lbl],
                                        node.getTransitionProbability(patt_lbl,
                                                                      trans_lbl)
                                        )
                        )
        """
        return rep

    def addState(self, state):
        if state not in self.nodes_dict:
            self.nodes_dict[state] = Node(state)

    def addInitialState(self, state):
        self.addState(state)
        self.initial_states.add(state)

    def addEndingState(self, state):
        self.addState(state)
        self.ending_states.add(state)

    def addPattern(self, state_orig, patt, state_dest, cnt=1):
        if cnt > 0:
            self.n_trans += cnt
            self.addState(state_orig)
            self.addState(state_dest)
            self.nodes_dict[state_orig].addTransition(patt, state_dest, cnt)

    def addInitialPattern(self, patt):
        self.initial_patterns.addPattern(patt)

    def addEndingPattern(self, patt):
        self.ending_patterns.addPattern(patt)

    def getPatternProbability(self, patt):
        if self.hasState(patt[0]):
            return (self.nodes_dict[patt[0]].getPatternCount(patt[1], patt[2]) /
                    self.n_trans)
        return 0

    def getInitialPatternProbability(self, patt):
        return self.initial_patterns.getPatternProbability(patt)

    def getEndingPatternProbability(self, patt):
        return self.ending_patterns.getPatternProbability(patt)

    def incrementEventCount(self, amount=1):
        self.n_events += amount

    def updateNodeCounts(self, histogram):
        for state, amount in histogram.items():
            self.addState(state)
            self.nodes_dict[state].incrementCount(amount)
            self.incrementEventCount(amount)

    def hasState(self, state):
        return state in self.nodes_dict

    def hasInitialState(self, patt):
        return patt in self.initial_states

    def hasEndingState(self, patt):
        return patt in self.ending_states

    def hasPattern(self, patt):
        if self.hasState(patt[0]) and \
                self.nodes_dict[patt[0]].hasTransition(patt[1], patt[2]):
            return True
        return False

    def hasInitialPattern(self, patt):
        return patt in self.initial_patterns

    def hasEndingPattern(self, patt):
        return patt in self.ending_patterns


class PatternCollection(object):
    def __init__(self):
        self.count = 0
        self.patterns = {}

    def addPattern(self, pattern, cnt=1):
        if cnt > 0:
            self.count += cnt
            if not self.hasPattern(pattern):
                self.patterns[pattern] = 0

            self.patterns[pattern] += cnt

    def hasPattern(self, pattern):
        if pattern in self.patterns:
            return True
        return False

    def getPatternProbability(self, pattern):
        if self.hasPattern(pattern):
            return self.patterns[pattern] / self.count
        return 0
