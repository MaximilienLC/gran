import argparse

from grans.grads import 
from grans.rands import ga

def main():

    for epoch in args.nb_epochs:

        if epoch == 0:
            random_behaviour_in_emulator()
        else:
            evolve_in_emulator()

        autoencoder.train()
        autoregressor.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs",
        "-o",
        type=int,
        default=1,
        help="Number of epochs to run th.",
    )

    parser.add_argument(
        "--env_path",
        "-e",
        type=str,
        required=True,
        help="Path to the Env class file.",
    )
