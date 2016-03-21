import graphviz as gv  # for plotting graphs

def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def plotInitialStates(patt_graph, name='initial', min_prob=0.0, max_prob=1.0,
                      decoder=lambda x: x, directory='images/', filepath=None):
    if filepath is None:
        filepath = name

    G = gv.Digraph(name=name, format='png', directory=directory)

    for state in patt_graph.initial_states:
        prob = patt_graph.nodes_dict[state].count / patt_graph.n_trans
        if prob >= min_prob and prob <= max_prob:
            G.node(decoder(state))

    G.render(filename=name)


def plotEndingStates(patt_graph, name='ending', min_prob=0.0, max_prob=1.0,
                     decoder=lambda x: x, directory='images/', filepath=None):
    ANY_STATE = 'ANY'
    if filepath is None:
        filepath = name

    G = gv.Digraph(name=name, format='png', directory=directory)

    for state in patt_graph.ending_states:
        G.node(decoder(state), shape='doublecircle')

    for state in patt_graph.ending_states:
        prob = patt_graph.nodes_dict[state].count / patt_graph.n_trans
        if prob >= min_prob and prob <= max_prob:
            G.edge(ANY_STATE, decoder(state), label=str(prob))

    G.render(filename=name)


def plotPatternGraph(patt_graph, name='transition', normalize=True,
                     min_prob=0.0, max_prob=1.0, decoder=lambda x: x,
                     directory='images/', filepath='None'):

    if filepath is None:
        filepath = name

    # instantiate directed multigraph
    G = gv.Digraph(name=name, format='png', directory=directory)

    # add starting states as square circle
    for state in patt_graph.initial_states:
        G.node(decoder(state), shape='invtriangle')

    # add ending states as double circles
    for state in patt_graph.ending_states:
        G.node(decoder(state), shape='doublecircle')

    # add other nodes and transitions
    for node_lbl, node in patt_graph.nodes_dict.items():
        for patt_lbl, trans_lbl, in node.transitions.keys():
            prob = node.getTransitionProbability(patt_lbl, trans_lbl)
            if prob >= min_prob and prob <= max_prob:
                G.edge(decoder(node_lbl), decoder(trans_lbl),
                       color='1.000 {:.3f} 0.900'.format(prob),
                       label='{}'.format(patt_lbl))

    G.render(filename=name)
