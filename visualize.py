# Note that  the code for plot_stats_new_trial() and plot_best is my own

from __future__ import print_function

import copy
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import neat


FONTSIZE = 16


def plot_stats_new_trial(a, b, c, d, filename=''):
    """ This function combines the statitistic reporters of the predator and prey species and produces graphs"""

    o_t_best, o_t_avg, o_t_stdev, o_y_best, o_y_avg, o_y_stdev = [], [], [], [], [], []

    for i in range(len(a)):
        t_stats = neat.StatisticsReporter()
        t_stats.most_fit_genomes = a[i]
        t_stats.generation_statistics = b[i]

        o_t_best.append([c.fitness for c in t_stats.most_fit_genomes])
        o_t_avg.append((t_stats.get_fitness_mean()))
        o_t_stdev.append(t_stats.get_fitness_stdev())

        y_stats = neat.StatisticsReporter()
        y_stats.most_fit_genomes = c[i]
        y_stats.generation_statistics = d[i]

        o_y_best.append([c.fitness for c in y_stats.most_fit_genomes])
        o_y_avg.append((y_stats.get_fitness_mean()))
        o_y_stdev.append(y_stats.get_fitness_stdev())

    t_best_fitness = np.mean(o_t_best, axis=0)
    t_avg_fitness = np.mean(o_t_avg, axis=0)
    t_stdev_fitness = np.mean(o_t_stdev, axis=0)

    y_best_fitness = np.mean(o_y_best, axis=0)
    y_avg_fitness = np.mean(o_y_avg, axis=0)
    y_stdev_fitness = np.mean(o_y_stdev, axis=0)

    best_fitness = (t_best_fitness + y_best_fitness) / 2.0
    plot_best(best_fitness, filename+"overall_best_fitness.png")

    avg_fitness = (t_avg_fitness + y_avg_fitness) / 2.0
    plot_best(avg_fitness, filename + "overall_avg_fitness.png")

    stdev_fitness = (t_stdev_fitness + y_stdev_fitness) / 2.0
    plot_best(stdev_fitness, filename + "overall_stdev_fitness.png")


def plot_best(best, file):
    """This function plots a single array"""
    plt.ylabel("Fitness (#)", fontsize=FONTSIZE)
    plt.xlabel("Epochs (#)", fontsize=FONTSIZE)
    plt.plot([x for x in range(len(best))], best)
    plt.savefig(file)
    plt.show()


def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    #plt.title("Speciation")
    plt.ylabel("Size per Species", fontsize=FONTSIZE)
    plt.xlabel("Generations", fontsize=FONTSIZE)

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add(cg.key)

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            #print(pending, used_nodes)
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot