import argparse
import logging
import os

from . import visualize
from .util import files

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

DEFAULT_HUMAN_JUDGMENTS = {
    "reddit": "https://s3.ca-central-1.amazonaws.com/ehsk-research/thred/eval/human_mer_reddit.pkl",
    "opensubtitles": "https://s3.ca-central-1.amazonaws.com/ehsk-research/thred/eval/human_mer_opensubtitles.pkl"
}

DEFAULT_RESPONSES = {
    "reddit": "https://s3.ca-central-1.amazonaws.com/ehsk-research/thred/eval/responses_reddit.txt",
    "opensubtitles": "https://s3.ca-central-1.amazonaws.com/ehsk-research/thred/eval/responses_opensubtitles.txt"
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--response_file', type=str, default="reddit", help='Generated response file')
    parser.add_argument('--human_judgment', type=str, default="reddit", help='Human judgment file')
    parser.add_argument('--bert_dir', type=str, help='Fine-tuned BERT model directory')
    parser.add_argument('--esim_model', type=str, help='Pre-trained ESIM model archive')
    parser.add_argument('--embedding', type=str, default="elmo",
                        choices=("elmo", "glove", "word2vec", 'bert-base-uncased', 'bert-large-uncased', 'bert-base-cased'),
                        help='Embedding method for semantic similarity and word-level metrics')
    parser.add_argument('--generator_types', type=str, nargs='*',
                        default=("THRED", "HRED", "Seq2Seq", "TA-Seq2Seq"),
                        help="The name of generative models to fare against each other")
    parser.add_argument('--plot_dir', type=str, help='Directory to store plots')

    args = parser.parse_args()
    print(args)

    response_file = DEFAULT_RESPONSES.get(args.response_file, args.response_file)
    human_judgment_file = DEFAULT_HUMAN_JUDGMENTS.get(args.human_judgment, args.human_judgment)

    if args.plot_dir is None:
        plots_dir = os.path.join(files.get_parent_dir(__file__), "plots")
    else:
        plots_dir = args.plot_dir

    visualize.plot(response_file, human_judgment_file, args.generator_types,
                   args.bert_dir, args.esim_model, args.embedding, plots_dir)


if __name__ == '__main__':
    main()
