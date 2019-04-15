import codecs
import json
import os

from tqdm import tqdm

from dialogentail.util import rand, stopwatch, files


def read_dnli_file(dnli_file):
    with codecs.getreader("utf-8")(open(dnli_file, "rb")) as dnli_reader:
        dnli_json = json.load(dnli_reader)

    for sample in dnli_json:
        label = sample['label']
        if label == 'negative':
            label = 'contradiction'
        elif label == 'positive':
            label = 'entailment'

        id = sample['id'][sample['id'].rfind('_') + 1:]
        promptId = rand.generate_random_digits()

        yield sample['sentence1'], sample['sentence2'], promptId, id, sample['dtype'], label


@stopwatch.profile
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dnli_file', type=str, help='DialogueNLI json file')

    args = parser.parse_args()

    basedir, filename = os.path.split(args.dnli_file)
    filename = files.get_file_name(filename)

    tsv_output = os.path.join(basedir, f"{filename}_mnli.tsv")
    jsonl_output = os.path.join(basedir, f"{filename}_snli.jsonl")

    with codecs.getwriter("utf-8")(open(tsv_output, "wb")) as tsv_file, \
            codecs.getwriter("utf-8")(open(jsonl_output, "wb")) as jsonl_file:
        tsv_file.write("index\tpromptID\tpairID\tgenre\tsentence1_binary_parse\tsentence2_binary_parse\t"
                       "sentence1_parse\tsentence2_parse\tsentence1\tsentence2\tlabel1\tgold_label\n")

        idx = 1
        for s1, s2, promptId, id, genre, label in tqdm(read_dnli_file(args.dnli_file)):
            tsv_file.write(
                "{index}\t{promptId}\t{pairId}\t{genre}\t()\t()\t()\t()\t{s1}\t{s2}\t{label}\t{label}\n".format(
                    index=idx, promptId=promptId, pairId=id, genre=genre,
                    s1=s1, s2=s2, label=label))
            idx += 1
            jsonl_file.write(
                f'{{"pairID": "{id}", "genre": "{genre}", "gold_label": "{label}", "sentence1": "{s1}", "sentence2": "{s2}"}}\n')


if __name__ == "__main__":
    main()
