import sys
import numpy as np
import graphviz as gv
import matplotlib as mpl
import matplotlib.pylab as plt
plt.ioff()


def set_trace():
    from IPython.core.debugger import Pdb
    import sys
    Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)


def plotScores(pg_scores, titles, figsize=(30, 5), normalize=False, title='',
               filename='scores', directory=''):

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.subplots_adjust(wspace=.3)

    for i in range(len(pg_scores)):
        pg_score = pg_scores[i]
        x_lbls = {}
        y_lbls = {}

        j = 0
        for lbl in sorted(np.unique(np.array(pg_score.keys())[:, 0])):
            x_lbls[lbl] = j
            j += 1

        j = 0
        for lbl in sorted(np.unique(np.array(pg_score.keys())[:, 1])):
            y_lbls[lbl] = j
            j += 1

        score_matrix = np.zeros((len(x_lbls), len(y_lbls)))
        for (a, b), vals in pg_score.items():
            score_matrix[x_lbls[a], y_lbls[b]] = np.mean(np.array(
                                                    vals)[:, 0].astype(float))
        if normalize:
            score_matrix = np.exp(score_matrix)
            score_matrix /= max(score_matrix)

        im = axes[i].imshow(score_matrix,
                            interpolation='nearest',
                            origin='low',
                            aspect='auto',
                            cmap=plt.cm.magma)

        tick_marks = np.arange(len(x_lbls))
        axes[i].set_xticks(tick_marks)
        axes[i].set_yticks(tick_marks)
        axes[i].set_xticklabels(sorted(x_lbls.keys()), rotation=90)
        axes[i].set_yticklabels(sorted(y_lbls.keys()))
        axes[i].set_xlabel('Genre')
        axes[i].set_ylabel('Pattern Graph')
        axes[i].set_title(titles[i])
        plt.colorbar(im, ax=axes[i])

    plt.savefig('{}{}.png'.format(directory, filename),
                bbox_inches='tight')


def plotViolationMatrix(pg_vratios, title, figsize=(10, 6), normalize=False,
                        filename='likelihood_matrix', directory=''):
    x_lbls = {}
    y_lbls = {}

    i = 0
    for lbl in sorted(np.unique(np.array(pg_vratios.keys())[:, 0])):
        x_lbls[lbl] = i
        i += 1

    i = 0
    for lbl in sorted(np.unique(np.array(pg_vratios.keys())[:, 1])):
        y_lbls[lbl] = i
        i += 1

    vratio_matrix = np.zeros((len(x_lbls), len(y_lbls)))

    for (a, b), vals in pg_vratios.items():
        vratio_matrix[x_lbls[a], y_lbls[b]] = np.mean(np.array(
                                                   vals)[:, 0].astype(float))
    if normalize:
        vratio_matrix = np.exp(vratio_matrix)
        vratio_matrix /= max(vratio_matrix)

    plt.figure(figsize=figsize)
    plt.imshow(vratio_matrix,
               interpolation='nearest',
               origin='low',
               aspect='auto',
               cmap=plt.cm.magma)

    tick_marks = np.arange(len(x_lbls))
    plt.xticks(tick_marks, sorted(x_lbls.keys()), rotation=90)
    plt.yticks(tick_marks, sorted(y_lbls.keys()))
    plt.tight_layout()
    plt.ylabel('Pattern Graph')
    plt.xlabel('Genre')
    plt.colorbar()
    plt.title(title)
    plt.savefig('{}{}.png'.format(directory, filename),
                bbox_inches='tight')


def plotLikelihoodMatrix(pg_likelihoods, title='', figsize=(10, 6),
                         normalize=False, filename='likelihood_matrix',
                         directory=''):
    x_lbls = {}
    y_lbls = {}

    i = 0
    for lbl in sorted(np.unique(np.array(pg_likelihoods.keys())[:, 0])):
        x_lbls[lbl] = i
        i += 1

    i = 0
    for lbl in sorted(np.unique(np.array(pg_likelihoods.keys())[:, 1])):
        y_lbls[lbl] = i
        i += 1

    lkl_matrix = np.zeros((len(x_lbls), len(y_lbls)))

    for (a, b), vals in pg_likelihoods.items():
        lkl_matrix[x_lbls[a], y_lbls[b]] = np.mean(np.array(
                                                   vals)[:, 0].astype(float))

    if normalize:
        lkl_matrix = np.exp(lkl_matrix)
        lkl_matrix /= max(lkl_matrix)

    plt.figure(figsize=figsize)
    plt.imshow(lkl_matrix,
               interpolation='nearest',
               origin='low',
               aspect='auto',
               cmap=plt.cm.magma)

    tick_marks = np.arange(len(x_lbls))
    plt.xticks(tick_marks, sorted(x_lbls.keys()), rotation=90)
    plt.yticks(tick_marks, sorted(y_lbls.keys()))
    plt.tight_layout()
    plt.ylabel('Pattern Graph')
    plt.xlabel('Genre')
    plt.colorbar()
    plt.title(title)
    plt.savefig('{}{}.png'.format(directory, filename),
                bbox_inches='tight')


def plotPatternSequence(raw_feature, patt_seq, decoder, dim_red=lambda x: x,
                        figsize=(30, 5), filename='patt_seq', directory=''):
    """
    Parameters
    ----------
    raw_feature : np.ndarray
        Raw feature used to create the pattern sequence
    patt_seq : list
        Contains a list where even numbered indices are patterns to be decoded
        and odd numbered indices are annotations
    decoder: method
        Method to decode patterns into matrix representation for image plotting

    Returns
    ------
        Plot including raw feature, dimensionality reduced pattern sequence,
        Plotted pattern sequence
    """

    img_mat = decoder([patt_seq[0, 0]] + patt_seq[:, 2].tolist())
    img_mat_red = dim_red(img_mat)

    fig, axes = plt.subplots(3, figsize=figsize, sharex=False)
    fig.suptitle(filename, y=1)
    axes[0].set_title('Raw Feature')
    axes[0].imshow(raw_feature,
                   interpolation='nearest',
                   origin='low',
                   aspect='auto',
                   cmap=plt.cm.Oranges)

    axes[1].set_title('Pattern Sequence Feature')
    axes[1].imshow(img_mat,
                   interpolation='nearest',
                   origin='low',
                   aspect='auto',
                   cmap=plt.cm.Oranges)

    axes[2].set_title('Dimensionality Reduced Feature')
    axes[2].imshow(img_mat_red,
                   interpolation='nearest',
                   origin='low',
                   aspect='auto')

    colors = {'F': plt.cm.gray(0.0),
              'T': plt.cm.gray(0.5),
              'S': plt.cm.gray(1.0)}

    for i in range(0, len(patt_seq)):
        axes[2].axvline(i,
                        ymin=0,
                        ymax=0.5,
                        linewidth=4,
                        color=colors[patt_seq[i, 1]],
                        ls='solid',
                        label=patt_seq[i, 1])

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]

    fig.subplots_adjust(hspace=.5)
    plt.legend(handles, labels,
               framealpha=0.5,
               bbox_to_anchor=(1.05, 1))
    plt.savefig('{}{}.png'.format(directory, filename), bbox_inches='tight')


def plotInitialStates(patt_graph, name='initial', min_prob=0.0, max_prob=1.0,
                      decoder=lambda x: x, directory='images/', filepath=None):
    PRE_STATE = 'Q0'
    if filepath is None:
        filepath = name

    G = gv.Digraph(name=name, format='svg', directory=directory, engine='dot',
                   graph_attr={'rankdir': 'TB'})

    for state in patt_graph.initial_states:
        G.node(decoder(state), shape='invtriangle')

    for state in patt_graph.initial_states:
        prob = patt_graph.nodes_dict[state].count / patt_graph.n_events
        if prob >= min_prob and prob <= max_prob:
            G.edge(PRE_STATE,
                   decoder(state),
                   color='1.000 {:.3f} 0.900'.format(prob),
                   label=str('{:.3f}'.format(prob)))

    G.graph_attr['rankdir'] = 'TB'
    G.render(filename=name, view=False)


def plotEndingStates(patt_graph, name='ending', min_prob=0.0, max_prob=1.0,
                     decoder=lambda x: x, directory='images/', filepath=None):
    ANY_STATE = 'ANY'
    if filepath is None:
        filepath = name

    G = gv.Digraph(name=name, format='svg', directory=directory, engine='dot',
                   graph_attr={'rankdir': 'TB'})

    for state in patt_graph.ending_states:
        G.node(decoder(state), shape='doublecircle')

    for state in patt_graph.ending_states:
        prob = patt_graph.nodes_dict[state].count / patt_graph.n_events
        if prob >= min_prob and prob <= max_prob:
            G.edge(ANY_STATE,
                   decoder(state),
                   color='1.000 {:.3f} 0.900'.format(prob),
                   label=str('{:.3f}'.format(prob)))

    G.graph_attr['rankdir'] = 'TB'
    G.render(filename=name, view=False)


def plotPatternGraph(patt_graph, name='transition', normalize=True,
                     min_prob=0.0, max_prob=1.0, decoder=lambda x: x,
                     directory='images/', filepath='None', render=False):

    if filepath is None:
        filepath = name

    # instantiate directed multigraph
    G = gv.Digraph(name=name, format='svg', directory=directory, engine='dot')

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
    if render:
        G.render(filename=name, view=False)
    else:
        G.save(filename=name)
