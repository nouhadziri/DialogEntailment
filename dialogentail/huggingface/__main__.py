import argparse

from .finetune_bert import run as bert_run


def set_bert_defaults(args):
    args.bert_model = args.model
    args.do_lower_case = "uncased" in args.model

    if args.learning_rate is None:
        args.learning_rate = 5e-5

    if args.warmup_proportion is None:
        args.warmup_proportion = 0.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        choices=("bert-base-uncased", "bert-large-uncased"),
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='')
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=None,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=None,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    # BERT specific parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="(BERT) Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="(BERT) The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="(BERT) local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="(BERT) Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="(BERT) Number of updates steps to accumulate before performing a backward/update pass.")

    args = parser.parse_args()

    set_bert_defaults(args)
    bert_run(args)


if __name__ == '__main__':
    main()
