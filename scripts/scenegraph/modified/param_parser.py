"""Parameter parser to set the model hyperparameters."""

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the partial NCI1 graph dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by ID.
    """
    parser = argparse.ArgumentParser(description="Run Graph2Vec.")

    parser.add_argument("--input-path",
                        nargs="?",
                        default="./dataset/",
	                help="Input folder with jsons.")

    parser.add_argument("--output-path",
                        nargs="?",
                        default="./features/nci1.csv",
	                help="Embeddings path.")

    parser.add_argument("--dimensions",
                        type=int,
                        default=128,
	                help="Number of dimensions. Default is 128.")

    parser.add_argument("--workers",
                        type=int,
                        default=4,
	                help="Number of workers. Default is 4.")

    parser.add_argument("--epochs",
                        type=int,
                        default=10,
	                help="Number of epochs. Default is 10.")

    parser.add_argument("--epochsave",
                        type=int,
                        default=10,
	                help="When to save the model.")

    parser.add_argument("--min-count",
                        type=int,
                        default=5,
	                help="Minimal structural feature count. Default is 5.")

    parser.add_argument("--wl-iterations",
                        type=int,
                        default=2,
	                help="Number of Weisfeiler-Lehman iterations. Default is 2.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.025,
	                help="Initial learning rate. Default is 0.025.")

    parser.add_argument("--down-sampling",
                        type=float,
                        default=0.0001,
	                help="Down sampling rate of features. Default is 0.0001.")

    parser.add_argument("--valsize",
                        type=int,
                        default=10,
	                help="Size of validation dataset.")

    parser.add_argument("--valeval",
                        type=int,
                        default=5,
	                help="When to evaluate model on validation dataset.")

    parser.add_argument("--evaltopk",
                        type=int,
                        default=100,
	                help="Consider topk retrieved image-similarity pairs for computing evaluation scores for each evaluation image.")

    parser.add_argument("--plotepoch",
                        type=int,
                        default=20,
	                help="Epoch at which to plot the loss log.")

    return parser.parse_args()
