import sys
import os

from EvolutionaryMusic.midi import Midi
from EvolutionaryMusic.generator import Generator
from EvolutionaryMusic.data_generator import DataGenerator
from EvolutionaryMusic.cli import parse
from EvolutionaryMusic.song_name_generator import random_name, timestamp_name

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(color_codes=True)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():
    # args, kwargs, conf = parse(sys.argv[1:])
    args, kwargs, conf = parse(
        [
            "--fitness=combination",
            "--subrater-path=../songs/training/",
            # "--subrater-path=../maestro-v3.0.0/2006",
            "--population-size=20",
            "--sequence-length=20",
            "--training-cycles=s0:i2:s0:i0",
            "--number-of-generations=30",
            "--epochs=100",
            "--early-stopping=20:25",
            # "--save-every-individual",
        ]
    )

    # If we want to plot data run the data generating generator
    if conf.plot:
        generator = DataGenerator(*args, **kwargs)
    else:
        generator = Generator(*args, **kwargs)

    generator.run()
    best_individual = generator.get_best_individual()

    if conf.plot:
        # Save generator variables into easier to use variables
        pop_size = generator.population_size
        nmb_of_gens = generator.number_of_generations
        time_data = generator.generator_data
        fitness_data = generator.fitness_data
        cross_data = generator.cross_data

        # Calculate average fitness for each generation
        fitness_data["Average"] = fitness_data.sum(axis=1) / pop_size

        # Plot Fitness over time
        plt.subplot(2, 1, 1)
        X = range(nmb_of_gens)
        plt.plot(X, fitness_data["Average"], label="Average fitness")
        plt.plot(X, fitness_data[pop_size - 1], label="Best fitness")
        plt.plot(X, fitness_data[0], label="Worst fitness")
        plt.xlabel("Number of generations")
        plt.ylabel("Fitness score")
        plt.legend(loc="best")

        # Plot Fitness over time but kinda regression ish
        plt.subplot(2, 1, 2)
        X = [i for i in range(nmb_of_gens)]
        columns = ["Average", pop_size - 1, 0]
        labels = ["Average fitness", "Best fitness", "Worst fitness"]
        multi_regplot(
            X,
            fitness_data,
            "Number of generations",
            "Fitness score",
            columns=columns,
            labels=labels,
            scatter=False,
        )
        plt.show()

        # Plot time consumtion vs number of nodes
        plt.subplot(2, 2, 1)
        X = time_data["avg number of nodes"]
        Y = time_data
        columns = ["fitness", "generators", "deeptime", "crosstime", "mutation"]
        multi_regplot(X, Y, "Average number of nodes in individual", "Time (seconds)", columns)

        # Plot time consumtion vs Size
        plt.subplot(2, 2, 2)
        X = time_data["avg length"]
        multi_regplot(X, time_data, "Average length individual", "Time (seconds)", columns)

        # Plot time connsumption vs average fitness
        plt.subplot(2, 2, 4)
        X = fitness_data["Average"]
        multi_regplot(X, time_data, "Average fitness of generation", "Time (seconds)", columns)
        plt.show()

        # More detailed plots for time of copy in crossover
        plt.subplot(2, 2, 1)
        X = cross_data[[1, 3]].max(axis=1)
        Y = cross_data[0]
        single_regplot(X, Y, "Max length of parent", "Time (seconds)")

        plt.subplot(2, 2, 2)
        X = cross_data[[2, 4]].max(axis=1)
        single_regplot(X, Y, "Max number of nodes of parents", "Time (seconds)")

        plt.subplot(2, 2, 3)
        X = cross_data[1] + cross_data[3]
        single_regplot(X, Y, "Added length of parent", "Time (seconds)")

        plt.subplot(2, 2, 4)
        X = cross_data[2] + cross_data[4]
        single_regplot(X, Y, "Added nmb of nodes of parent", "Time (seconds)")
        plt.show()

        print(cross_data)

    # Play song with pyo
    if conf.play:
        midi_player = Midi()
        midi_player.play(best_individual)

    # show best individual
    best_individual.root_node.print_tree()
    print("Best individual")
    print(best_individual)
    print("Fitness: " + str(best_individual.fitness_score))

    # Write to a midi file
    if not conf.no_save:
        if conf.save_every_individual:

            songs_path = timestamp_name()[:-4]
            os.makedirs(songs_path)

            individuals = generator.individuals()
            leading_zeros = len(str(len(individuals)))

            for i, individual in enumerate(individuals):
                name = f"{(i+1)}".zfill(leading_zeros)
                path_name = f"{songs_path}/{name}.mid"
                print(f"Saving {path_name}")
                Midi.save(individual, path_name)
        else:
            name = random_name() if conf.random_song_name else timestamp_name()
            print(f"Saving {name}")
            Midi.save(best_individual, name)


def single_regplot(X, Y, x_label, y_label):
    sns.regplot(x=X, y=Y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def multi_regplot(X, Y, x_label, y_label, columns, labels=None, scatter=True):
    if labels is None:
        labels = columns
    for i in range(len(columns)):
        sns.regplot(x=X, y=Y[columns[i]], label=labels[i], scatter=scatter)
    plt.legend(loc="best")
    plt.xlabel(x_label)
    plt.ylabel(y_label)


if __name__ == "__main__":
    main()
