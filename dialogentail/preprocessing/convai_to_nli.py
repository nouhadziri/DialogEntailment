import codecs
import logging
import os
import random
import subprocess
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from dialogentail.preprocessing.dnli import read_dnli_file
from dialogentail.reader.convai_reader import ConvAIReader
from dialogentail.reader.multinli_reader import MultiNLIReader
from dialogentail.util import rand, nlp, stopwatch, files

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('ConvAI2NLI')

dull_responses = [
    "i don't know .",
    "i don't know what you're talking about .",
    "i don't know what you mean .",
    "i don't know what you mean by that .",
    "i don't know what you mean by that answer .",
    "i'm not sure .",
    "i'm not sure what you're saying .",
    "i'm not sure what you're talking about .",
    "i'm not sure what you're trying to say .",
    "i'm not sure what you mean by that .",
    "i'm not sure what you mean by this .",
    "i'm not sure what you mean .",
    "i don't understand what you mean .",
]


# def _fix_contractions(txt):
#     refined_txt = txt
#     refined_txt = re.sub(r"\bi'm\b", "i 'm", refined_txt)
#     refined_txt = re.sub(r"\byou're\b", "you 're", refined_txt)
#     refined_txt = re.sub(r"\bhe's\b", "he 's", refined_txt)
#     refined_txt = re.sub(r"\bthey're\b", "they 're", refined_txt)
#     refined_txt = re.sub(r"\bi've\b", "i 've", refined_txt)
#     refined_txt = re.sub(r"\bi'd\b", "i 'd", refined_txt)
#     return refined_txt


def generate_dull_samples(size=1):
    return random.sample(dull_responses, size)


def generate_poor_samples_from_utterances(utterances, lb, ub, size=1):
    other_utterances = utterances[:lb]
    other_utterances.extend(utterances[ub + 1:])
    other_utterances = [u for u in other_utterances if u != ""]
    return random.sample(other_utterances, size)


def generate_poor_samples_from_facts(facts, size=1):
    return random.sample(facts, size)


def generate_broken_samples(utterances, lb, ub, size=1):
    broken_samples = []
    for _ in range(size):
        rand_tokens = []
        response_tokens = nlp.omit_punctuations(utterances[ub])
        if response_tokens:
            rand_tokens.extend(random.sample(response_tokens, random.randint(min(len(response_tokens) - 1, 3),
                                                                             min(len(response_tokens) - 1, 5))))
        context_tokens = nlp.omit_punctuations(" ".join(utterances[lb:ub]))
        if context_tokens:
            rand_tokens.extend(random.sample(context_tokens, random.randint(min(len(context_tokens) - 1, 3),
                                                                            min(len(context_tokens) - 1, 5))))

        other_utterances = utterances[:lb]
        other_utterances.extend(utterances[ub + 1:])
        other_tokens = nlp.omit_punctuations(" ".join(other_utterances))
        if other_tokens:
            rand_tokens.extend(random.sample(other_tokens, random.randint(min(len(other_tokens) - 1, 3),
                                                                          min(len(other_tokens) - 1, 6))))

        if rand_tokens:
            broken_samples.append(" ".join(rand_tokens))

    return broken_samples


def flush_conversations(conversations, context_size, tsv_file,
                        prob_dull_samples=0.75, n_dull_samples=1,
                        prob_broken_samples=0.75, max_broken_samples=2, max_poor_samples=2,
                        start_index=0):
    num_turns = context_size + 1
    idx = start_index

    for dialogue, speaker_facts, partner_facts in conversations:
        facts = list(speaker_facts)
        facts.extend(partner_facts)

        utterances = [utterance.replace('__SILENCE__', '').strip() for utterance in dialogue]
        empty_utterance_indices = {i for i, utterance in enumerate(utterances) if utterance == ""}

        for i in range(len(dialogue)):
            lb = i
            ub = min(i + context_size, len(dialogue))

            for x in range(lb, ub):
                if x in empty_utterance_indices:
                    context_has_empty_utterance = True
                    break
            else:
                context_has_empty_utterance = False

            if context_has_empty_utterance:
                continue

            context = ' '.join(utterances[lb:ub])
            response = utterances[min(i + context_size, len(dialogue) - 1)]

            if not response:
                continue

            id = rand.generate_random_string(5)
            promptId = rand.generate_random_digits()
            tsv_file.write(
                "{index}\t{promptId}\t{pairId}\tdialogue\t()\t()\t()\t()\t{s1}\t{s2}\t{label}\t{label}\n".format(
                    index=idx, promptId=promptId, pairId=f"{id}0", s1=context, s2=response,
                    label="entailment"))
            idx += 1

            j = 1
            if random.random() < prob_dull_samples:
                for dull_response in generate_dull_samples(size=n_dull_samples):
                    promptId = rand.generate_random_digits()
                    tsv_file.write(
                        "{index}\t{promptId}\t{pairId}\tdialogue\t()\t()\t()\t()\t{s1}\t{s2}\t{label}\t{label}\n".format(
                            index=idx, promptId=promptId, pairId=f"{id}D{j}", s1=context, s2=dull_response,
                            label="neutral"))
                    j += 1
                    idx += 1

            if max_poor_samples > 0:
                poor_samples = generate_poor_samples_from_utterances(utterances, lb, ub,
                                                                     size=random.randint(1, max_poor_samples))

                for poor_response in poor_samples:
                    promptId = rand.generate_random_digits()
                    tsv_file.write(
                        "{index}\t{promptId}\t{pairId}\tdialogue\t()\t()\t()\t()\t{s1}\t{s2}\t{label}\t{label}\n".format(
                            index=idx, promptId=promptId, pairId=f"{id}P{j}", s1=context, s2=poor_response,
                            label="neutral"))
                    j += 1
                    idx += 1

            if random.random() < prob_broken_samples:
                for bad_response in generate_broken_samples(utterances, lb, ub,
                                                            size=random.randint(1, max_broken_samples)):
                    promptId = rand.generate_random_digits()
                    tsv_file.write(
                        "{index}\t{promptId}\t{pairId}\tdialogue\t()\t()\t()\t()\t{s1}\t{s2}\t{label}\t{label}\n".format(
                            index=idx, promptId=promptId, pairId=f"{id}P{j}", s1=context, s2=bad_response,
                            label="contradiction"))
                    j += 1
                    idx += 1

            if i >= len(dialogue) - num_turns:
                break

    return idx


@stopwatch.profile
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('convai_file', type=str, help='ConvAI no_cands file')
    parser.add_argument('-s', "--context_size", default=2, type=int,
                        help='context size (i.e., num of utterances included as context)')
    parser.add_argument("--prob_dull_samples", default=0.33, type=float,
                        help='the probability of generating dull samples per entailment sample')
    parser.add_argument("--n_dull_samples", default=2, type=int,
                        help='if determined to generate dull samples, the number of samples generated')
    parser.add_argument("--prob_broken_samples", default=0.30, type=float,
                        help='the probability of generating broken samples per entailment sample')
    parser.add_argument("--max_broken_samples", default=3, type=int,
                        help='if determined to generate broken samples, number of samples generated would be '
                             'random between 1 and max_broken_samples (inclusive)')
    parser.add_argument("--max_poor_samples", default=3, type=int,
                        help='number of poor samples (i.e., randomly chosen utterances) generated would be '
                             'random between 1 and max_poor_samples (inclusive). 0 to exclude poor samples.')
    parser.add_argument("--mnli_entailments", default=0, type=int,
                        help='number of entailment samples randomly selected from MultiNLI')
    parser.add_argument("--mnli_neutrals", default=0, type=int,
                        help='number of neutral samples randomly selected from MultiNLI')
    parser.add_argument("--mnli_contradictions", default=180000, type=int,
                        help='number of contraditions samples randomly selected from MultiNLI')
    parser.add_argument('--mnli_file', type=str, help='MultiNLI v1.0 file to draw random samples from')
    parser.add_argument("--dnli_positives", default=20000, type=int,
                        help='number of positive samples randomly selected from DialogueNLI')
    parser.add_argument("--dnli_neutrals", default=4000, type=int,
                        help='number of neutral samples randomly selected from DialogueNLI')
    parser.add_argument("--dnli_negatives", default=110000, type=int,
                        help='number of contraditions samples randomly selected from Dialogue NLI')
    parser.add_argument('--dnli_file', type=str, help='DialogueNLI file to draw random samples from')

    args = parser.parse_args()
    print(args)

    if args.prob_dull_samples < 0 or args.prob_dull_samples > 1:
        raise ValueError("prob_dull_samples must be within [0.0, 1.0]")

    if args.prob_broken_samples < 0 or args.prob_broken_samples > 1:
        raise ValueError("prob_broken_samples must be within [0.0, 1.0]")

    if args.n_dull_samples < 1:
        raise ValueError("n_dull_samples must be >= 1")

    if args.max_broken_samples < 1:
        raise ValueError("max_broken_samples must be >= 1")

    if args.max_poor_samples < 0:
        raise ValueError("max_poor_samples must be >= 0")

    basedir, filename = os.path.split(args.convai_file)
    filename = files.get_file_name(filename)

    if args.mnli_file and args.dnli_file:
        suffix = "_MDnli"
    elif args.mnli_file:
        suffix = "_mnli"
    elif args.dnli_file:
        suffix = "_dnli"
    else:
        suffix = ""

    tsv_output = os.path.join(basedir, f"convai_nli_{filename}_ctx{args.context_size}{suffix}.tsv")
    jsonl_output = os.path.join(basedir, f"convai_nli_{filename}_ctx{args.context_size}{suffix}.jsonl")

    with codecs.getwriter("utf-8")(open(tsv_output, "wb")) as tsv_file:

        tsv_file.write("index\tpromptID\tpairID\tgenre\tsentence1_binary_parse\tsentence2_binary_parse\t"
                       "sentence1_parse\tsentence2_parse\tsentence1\tsentence2\tlabel1\tgold_label\n")

        conversations = []
        idx = 1
        for convai_sample in tqdm(ConvAIReader(args.convai_file), desc="ConvAI"):
            conversations.append(convai_sample)

            if len(conversations) > 10000:
                idx = flush_conversations(conversations, args.context_size, tsv_file,
                                          args.prob_dull_samples, args.n_dull_samples,
                                          args.prob_broken_samples, args.max_broken_samples,
                                          args.max_poor_samples,
                                          start_index=idx)
                conversations = []

        flush_conversations(conversations, args.context_size, tsv_file,
                            args.prob_dull_samples, args.n_dull_samples,
                            args.prob_broken_samples, args.max_broken_samples,
                            args.max_poor_samples,
                            start_index=idx)

        if args.mnli_file and os.path.exists(args.mnli_file):
            idx = _generate_from_nli(MultiNLIReader(args.mnli_file),
                                     {"entailment": args.mnli_entailments,
                                      "neutral": args.mnli_neutrals,
                                      "contradiction": args.mnli_contradictions},
                                     idx,
                                     tsv_file,
                                     desc="MultiNLI")

        if args.dnli_file and os.path.exists(args.dnli_file):
            _generate_from_nli(read_dnli_file(args.dnli_file),
                               {"entailment": args.dnli_positives,
                                "neutral": args.dnli_neutrals,
                                "contradiction": args.dnli_negatives},
                               idx,
                               tsv_file,
                               desc="DialogueNLI")

    is_windows = sys.platform.startswith("win")
    if is_windows:
        logger.warning("Please shuffle the output file or run the program in a Unix-based system.")
    else:
        logger.info("Shuffling...")
        script_file = os.path.join(files.get_containing_dir(__file__), "fin.sh")
        subprocess.run([script_file, tsv_output], stdout=subprocess.PIPE)

    logger.info("Converting to jsonl...")
    _convert_to_jsonl(jsonl_output, tsv_output)


def _convert_to_jsonl(jsonl_output, tsv_output):
    with codecs.getreader("utf-8")(open(tsv_output, "rb")) as tsv_file, \
            codecs.getwriter("utf-8")(open(jsonl_output, "wb")) as jsonl_file:
        for i, line in enumerate(tsv_file):
            if i == 0:
                continue

            line = line.strip()
            tokens = line.split("\t")
            pairId = tokens[2]
            genre = tokens[3]
            gold_label = tokens[-1]
            s1 = tokens[8].replace('\\', '\\\\').replace('"', '\\"')
            s2 = tokens[9].replace('\\', '\\\\').replace('"', '\\"')
            jsonl_file.write(
                f'{{"pairID": "{pairId}", "genre": "{genre}", "gold_label": "{gold_label}", "sentence1": "{s1}", "sentence2": "{s2}"}}\n')


def _normalize(text, spacy):
    return " ".join(spacy.word_tokenize(text.replace('I\\m', "I'm"))).lower().strip()


def _generate_from_nli(nli_iterator, n_samples, idx, tsv_file, desc):
    logger.info(f"Drawing random samples from {desc}...")

    nli_samples = defaultdict(lambda: [])
    for s1, s2, promptId, pairId, genre, gold_label in tqdm(nli_iterator, desc=desc):

        if gold_label not in n_samples or n_samples[gold_label] == 0:
            continue

        nli_samples[gold_label].append((s1, s2, promptId, pairId, genre))

    for gold_label, samples in nli_samples.items():
        indices = np.arange(len(samples))
        np.random.shuffle(indices)
        random_indices = indices[:n_samples[gold_label]]

        for r_i in random_indices:
            s1, s2, promptId, pairId, genre = samples[r_i]
            tsv_file.write(
                f"{idx}\t{promptId}\t{pairId}\t{genre}\t()\t()\t()\t()\t{s1}\t{s2}\t{gold_label}\t{gold_label}\n")
            idx += 1

    return idx


if __name__ == "__main__":
    main()
