import codecs
import os
from random import randint, shuffle, sample, random

from tqdm import tqdm

from dialogentail.reader.swag_reader import SwagReader
from dialogentail.util import stopwatch, nlp

dull_responses = [
    "I don't know.",
    "I don't know what you're talking about.",
    "I don't know what you mean.",
    "I don't know what you mean by that.",
    "I don't know what you mean by that answer.",
    "I'm not sure.",
    "I'm not sure what you're saying.",
    "I'm not sure what you're talking about.",
    "I'm not sure what you're trying to say.",
    "I'm not sure what you mean by that.",
    "I'm not sure what you mean by this.",
    "I'm not sure what you mean.",
    "I don't understand what you mean.",
]


@stopwatch.profile
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('swagfile', type=str, help='swag csv file')

    args = parser.parse_args()

    basedir, filename = os.path.split(args.swagfile)
    output = os.path.join(basedir, f"dial_{filename}.jsonl")

    spacy = nlp.Spacy()

    with codecs.getwriter("utf-8")(open(output, "wb")) as out_file:
        line, rand_line, dull_line = 0, 0, 0
        for id, sentence1, distractors, sentence2 in tqdm(SwagReader(args.swag_file)):
            if sentence2:
                out_file.write('{{"pairID": "{}0", "gold_label": "coherent", '
                               '"sentence1": "{}", "sentence2": "{}"}}\n'.format(id, sentence1, sentence2))

            for p, poor_sent in enumerate(distractors):
                out_file.write('{{"pairID": "{id}{seq}", "gold_label": "{gold}", '
                               '"sentence1": "{s1}", "sentence2": "{s2}"}}\n'.format(id=id, seq=p + 1,
                                                                                     gold="" if sentence2 is None else "poor",
                                                                                     s1=sentence1, s2=poor_sent))

            if sentence2:
                bad_size = randint(0, len(distractors) // 2)
                rand_line += bad_size
                if bad_size > 0:
                    tokens = set()
                    for p in distractors:
                        tokens.update(spacy.word_tokenize(p))

                    for i in range(bad_size):
                        utter_size = randint(5, len(tokens) // 2)
                        shuffled_list = list(tokens)
                        shuffle(shuffled_list)
                        out_file.write('{{"pairID": "{}1{}", "gold_label": "bad", '
                                       '"sentence1": "{}", "sentence2": "{}"}}\n'
                                       .format(id, i + 1, sentence1, ' '.join(shuffled_list[:utter_size])))

                dull_size = 0 if random() <= 0.5 else 1
                dull_line += dull_size
                if dull_size > 0:
                    dull_utterances = sample(dull_responses, dull_size)

                    for i, dull_utter in enumerate(dull_utterances):
                        out_file.write('{{"pairID": "{}3{}", "gold_label": "poor", '
                                       '"sentence1": "{}", "sentence2": "{}"}}\n'
                                       .format(id, i + 1, sentence1, dull_utter))

        print(f"{line} lines + {rand_line} random lines + {dull_line} dull lines exported")


if __name__ == "__main__":
    main()
