import argparse

from EvolutionaryMusic.fitness import (
    Fitness,
    FitnessSubrater,
    FitnessNeural,
    calculate_fitness_only_C,
    calculate_set_note_fitness,
)
from EvolutionaryMusic.parent_selection import (
    tournament_selection,
    roulette_selection,
    sus_selection,
    rank_selection,
    elitism,
    parent_generator_selection,
    survivor_selection,
)


def parse(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--fitness",
        choices=["subrater", "c", "set_note", "neural", "combination"],
        help="The fitness function to use",
        default="subrater",
    )

    parser.add_argument(
        "-n",
        "--no-save",
        help="Do not save the generated song",
        action="store_true",
    )

    parser.add_argument(
        "-p",
        "--play",
        help="Play the best generated song",
        action="store_true",
    )

    parser.add_argument(
        "--parent",
        help="Set parent selection",
        choices=["tournament", "roulette", "sus", "rank", "elitism", "random"],
        default="random",
    )

    parser.add_argument(
        "--survivor",
        help="Set survivor selection",
        choices=["tournament", "roulette", "sus", "rank", "elitism", "random"],
        default="random",
    )

    parser.add_argument(
        "-g",
        "--number-of-generations",
        help="Set the number of generations",
        type=int,
        default=30,
    )

    parser.add_argument(
        "-m",
        "--mutation-probability",
        help="Set the probability of mutation",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--population-size",
        help="Set the population size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--sequence-length",
        help="Set the sequence length for nerual network data",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--epochs",
        help="Set the number of times the training set will be processed by the nerual network",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--training-cycles",
        help=(
            "Specify which generations we are going to switch fitness function."
            " E.g. i0:s5:i0:a0:i5; (s) for subrater, (i) for interactive, (a) for automatic."
            " Default is subrater."
        ),
        default="none",
    )

    parser.add_argument(
        "--subrater-path",
        help=(
            "Specify the path of midi songs which the "
            "subrater fitness function should calibrate from."
        ),
        default=None,
    )

    parser.add_argument(
        "--save-interactive-samples",
        help=("Specify whether you want to save interactive midi songs"),
        action="store_true",
    )

    parser.add_argument(
        "--train-existing-songs",
        help=("Specify if you want to train the model on midi files from ../songs/training"),
        action="store_true",
    )

    parser.add_argument(
        "--load-neural-model",
        help=("Specify if you want to load a trained model before running the program"),
        action="store_true",
    )

    parser.add_argument(
        "--plot",
        help="Bool if you want to plot generator data or not",
        action="store_true",
    )

    parser.add_argument(
        "--random-song-name",
        help="Use random name for the saved best individual",
        action="store_true",
    )

    parser.add_argument(
        "--save-every-individual",
        help="Instead of just saving the 'best' individual in terms of fitness score"
        ", save every individal in the final generation",
        action="store_true",
    )

    parser.add_argument(
        "--early-stopping",
        help="To enable previews after n..m generations and the option "
        "to stop if the individuals are performing well",
        default=None,
    )

    conf = parser.parse_args(argv)

    kwargs = {
        "starting_population": None,
        "number_of_generations": conf.number_of_generations,
        "mutation_probability": conf.mutation_probability,
        "population_size": conf.population_size,
        "early_stopping": conf.early_stopping,
    }

    args = [
        __get_fitness(conf),
        __get_parent_selection_generator(conf.parent),
        __get_survivor_selection_generator(conf.survivor),
    ]

    return args, kwargs, conf


def __get_fitness(conf):
    splits = [] if conf.training_cycles == "none" else conf.training_cycles.split(":")

    if conf.fitness == "subrater":
        return FitnessSubrater(conf.subrater_path)
    elif conf.fitness == "c":
        return calculate_fitness_only_C
    elif conf.fitness == "set_note":
        return calculate_set_note_fitness
    elif conf.fitness == "neural":
        return FitnessNeural(
            conf.population_size,
            conf.sequence_length,
            conf.epochs,
            splits,
            conf.save_interactive_samples,
            conf.train_existing_songs,
            conf.load_neural_model,
            False,
        )
    elif conf.fitness == "combination":
        return Fitness(
            conf.population_size,
            conf.sequence_length,
            conf.epochs,
            splits,
            conf.save_interactive_samples,
            conf.train_existing_songs,
            conf.load_neural_model,
            conf.subrater_path,
        )

    return None


def __get_parent_selection_generator(selector):
    SELECTION_GENERATORS = {
        "tournament": tournament_selection,
        "roulette": roulette_selection,
        "sus": sus_selection,
        "rank": rank_selection,
        "elitism": elitism,
        "random": parent_generator_selection,
    }

    return SELECTION_GENERATORS[selector]


def __get_survivor_selection_generator(selector):
    SELECTION_GENERATORS = {
        "tournament": tournament_selection,
        "roulette": roulette_selection,
        "sus": sus_selection,
        "rank": rank_selection,
        "elitism": elitism,
        "random": survivor_selection,
    }

    return SELECTION_GENERATORS[selector]
