"""
This file contains the code that runs the project using NEAT.
Date:
    March 23, 2020
Project:
    Simulating the Red Queen's Hypothesis
Author:
    name: Vineel Nagisetty
    contact: vineel.nagisetty@uwaterloo.ca
"""

# Imports
from make_env import make_env
import neat
from gym import wrappers
import numpy as np
from utils import random_action, process_output
import argparse
import visualize


# Create Environment
env = make_env('simple_tag_one')

# load configs
prey_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                          neat.DefaultStagnation, "./configs/prey_config")

predator_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                              neat.DefaultStagnation, "./configs/predator_config")


# Declare variables
prey_brain = None
predator_brain = None
best_prey = None
best_predator = None

# Declare constants
STEPS = 100


def get_predator_action(obs):
    """ This function returns an action for a predator"""
    if predator_brain is None:
        return random_action()
    else:
        return process_output(np.array(predator_brain.activate(obs)))


def get_prey_action(obs):
    """ This function returns an action for the prey"""
    if prey_brain is None:
        return random_action()
    else:
        return process_output(np.array(prey_brain.activate(obs)))


def eval_individual(steps=STEPS, is_prey=True):
    """This function calculate fitness for an individual using the environment"""
    reward = 0
    obs = env.reset()
    # run the simulation
    for i in range(steps):
        action = [get_predator_action(obs[0]), get_prey_action(obs[1])]
        obs, rewards, _, _ = env.step(np.array(action))
        if is_prey:
            reward += rewards[1]
        else:
            reward += rewards[0]
    return reward


def eval_prey_population(genomes, config):
    """This function evaluates fitness of the entire prey population"""
    global prey_brain
    for genome_id, genome in genomes:
        prey_brain = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = eval_individual(is_prey=True)


def eval_predator_population(genomes, config):
    """This function evaluates fitness of the entire predator population"""
    global predator_brain
    for genome_id, genome in genomes:
        predator_brain = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = eval_individual(is_prey=False)


def run_experiment(num_trials, num_epochs, num_runs, reports=False, plots=True, view=True) -> None:
    """This function runs the entire experiment for the desired number of trials"""

    # single trial
    def single_trial(epochs, runs):
        """This function runs one trial for the given number of epochs"""

        global best_prey, prey_brain, best_predator, predator_brain
        gen_count = 0

        predator_most_fit_genomes, predator_generation_statistics = [], []
        prey_most_fit_genomes, prey_generation_statistics = [], []

        predator_checkpoint_f = './checkpoints/predator/checkpoint-'
        prey_checkpoint_f = './checkpoints/prey/checkpoint-'

        # inner loop to run each trial
        for i in range(epochs):
            print("Starting epoch {}".format(i))

            ########### Train predator
            if i == 0:
                predator_population = neat.Population(predator_config)
            else:
                predator_population = neat.Checkpointer.restore_checkpoint(predator_checkpoint_f + str(gen_count-1))

            predator_population.generation = gen_count
            predator_checkpoint = neat.Checkpointer(runs, None, predator_checkpoint_f)
            predator_checkpoint.last_generation_checkpoint = gen_count - 1
            predator_population.add_reporter(predator_checkpoint)

            if reports:
                predator_stdout = neat.StdOutReporter(False)
                predator_stdout.generation = gen_count
                predator_population.add_reporter(predator_stdout)

            if plots:
                predator_stats = neat.StatisticsReporter()
                predator_stats.most_fit_genomes = predator_most_fit_genomes
                predator_stats.generation_statistics = predator_generation_statistics
                predator_population.add_reporter(predator_stats)

            best_predator = predator_population.run(eval_predator_population, runs)
            predator_brain = neat.nn.FeedForwardNetwork.create(best_predator, predator_config)

            ########### Train prey
            if i == 0:
                prey_population = neat.Population(prey_config)
            else:
                prey_population = neat.Checkpointer.restore_checkpoint(prey_checkpoint_f + str(gen_count-1))

            prey_population.generation = gen_count
            prey_checkpoint = neat.Checkpointer(runs, None, prey_checkpoint_f)
            prey_checkpoint.last_generation_checkpoint = gen_count - 1
            prey_population.add_reporter(prey_checkpoint)

            if reports:
                prey_stdout = neat.StdOutReporter(False)
                prey_stdout.generation = gen_count
                prey_population.add_reporter(prey_stdout)

            if plots:
                prey_stats = neat.StatisticsReporter()
                prey_stats.most_fit_genomes = prey_most_fit_genomes
                prey_stats.generation_statistics = prey_generation_statistics
                prey_population.add_reporter(prey_stats)

            best_prey = prey_population.run(eval_prey_population, runs)
            prey_brain = neat.nn.FeedForwardNetwork.create(best_prey, prey_config)

            # update generation count
            gen_count += runs

            # update stats as required
            if plots:
                predator_most_fit_genomes = predator_stats.most_fit_genomes
                predator_generation_statistics = predator_stats.generation_statistics
                prey_most_fit_genomes = prey_stats.most_fit_genomes
                prey_generation_statistics = prey_stats.generation_statistics

        if plots:
            # plot predator stats
            visualize.plot_stats(predator_stats, ylog=False, view=False, ptype=True, filename="./results_experiment/plots/predator/epochs={}_runs={}_".format(num_epochs, num_runs))
            visualize.plot_species(predator_stats, filename="./results_experiment/plots/predator/epochs={}_runs={}_.species.svg".format(num_epochs, num_runs))
            visualize.draw_net(predator_config, best_predator,
                               filename="./results_experiment/plots/predator/epochs={}_runs={}_.net".format(num_epochs, num_runs))

            # plot prey stats
            visualize.plot_stats(prey_stats, ylog=False, view=False, ptype=False, filename="./results_experiment/plots/prey/epochs={}_runs={}".format(num_epochs, num_runs))
            visualize.plot_species(prey_stats, filename="./results_experiment/plots/prey/epochs={}_runs={}_.species.svg".format(num_epochs, num_runs))
            visualize.draw_net(prey_config, best_prey,
                               filename="./results_experiment/plots/prey/epochs={}_runs={}_.net".format(num_epochs, num_runs))

        if view:
            view_run(steps=STEPS, record=True,
                     info="epochs={}_runs={}_current_gens={}".format(num_epochs, num_runs, gen_count))

        return

    # outer loop to run trials
    for j in range(num_trials):
        print("Starting Trial {}".format((j+1)))
        single_trial(num_epochs, num_runs)

    # Completed experiment
    print('\n Completed Experiment')
    return


def view_run(steps, record=False, info="") -> None:
    """ This function runs the simulation, rendering it so we can see"""
    if record:
        env_local = wrappers.Monitor(env, './results_experiment/videos/' + info + '/')

    obs = env_local.reset()
    # run the simulation
    for i in range(steps):
        action = [get_predator_action(obs[0]), get_prey_action(obs[1])]
        obs, rewards, _, _ = env_local.step(np.array(action))
        env_local.render(mode='rgb_array')


def main():
    """
    The main function that parses arguments
    :return:
    """
    parser = argparse.ArgumentParser(description="run the experiment using the given values")
    parser.add_argument("-e", "--num_epochs", type=int,
                        default=2, metavar='num_epochs',
                        help="the number of epochs to run")
    parser.add_argument("-t", "--num_trials", type=int,
                        default=1, metavar='num_trials',
                        help="the number of trials to run")
    parser.add_argument("-r", "--num_runs", type=int,
                        default=2, metavar="num_runs",
                        help="the number of runs for each epoch")
    parser.add_argument("-s", "--steps", type=int,
                        default=100, metavar="steps",
                        help="the number of steps to run for each simulation")

    args = parser.parse_args()
    #run_experiment(args.num_trials, args.num_epochs, args.num_runs, view=True)

    combinations = {
        (1, 200, 1),
        (1, 100, 2),
        (1, 50, 4),
        (1, 40, 5),
        (1, 25, 8),
        (1, 20, 10),
        (1, 8, 25),
        (1, 5, 40),
        (1, 4, 50),
        (1, 2, 100),
        (1, 1, 200)
    }
    for combination in combinations:
        print("using trials={}, epochs={}, runs={}".format(*combination))
        #run_experiment(*combination)
    run_experiment(1, 1, 1)


if __name__ == "__main__":
    main()
