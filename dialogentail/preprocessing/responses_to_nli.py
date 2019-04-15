import codecs

from dialogentail.reader.response_reader import ResponseFileReader
from dialogentail.util import rand, files


def get_entailment_label(score):
    if score >= 2: # coherent
        return 'entailment'
    elif score >= 1: # poor
        return 'neutral'
    else: # bad
        return 'contradiction'


def _escape(text):
    return text.replace('\\', '\\\\').replace('"', '\\"')


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('responses_file', type=str, help='Responses file. '
                                                         'For each sample: '
                                                         '(Line 1: conversation history (tab-separated), '
                                                         'Line 2: ground truth, '
                                                         'Line 3... generated responses)')
    parser.add_argument('--judgment_file', type=str,
                        help='pickle file containing the average rating of human judgment per test dialogue')
    parser.add_argument("--n_methods", default=4, type=int,
                        help='number of generation methods (i.e., num of generated responses per test dialogue)')

    args = parser.parse_args()

    if args.judgment_file is not None:
        mean_human_judgment = files.load_obj(args.judgment_file)
    else:
        mean_human_judgment = None

    filename = files.get_file_name(args.responses_file)
    responses_jsonl_output = f"nli_{filename}.jsonl"
    responses_tsv_output = f"nli_{filename}.tsv"
    ground_truth_jsonl_output = f"groundtruth_nli_{filename}.jsonl"
    ground_truth_tsv_output = f"groundtruth_nli_{filename}.tsv"

    if mean_human_judgment:
        resp_tsv = codecs.getwriter("utf-8")(open(responses_tsv_output, "wb"))
        resp_jsonl = codecs.getwriter("utf-8")(open(responses_jsonl_output, "wb"))
    else:
        resp_tsv, resp_jsonl = None, None

    id, idx = 10001, 1
    with codecs.getwriter("utf-8")(open(ground_truth_jsonl_output, "wb")) as gt_jsonl, \
            codecs.getwriter("utf-8")(open(ground_truth_tsv_output, "wb")) as gt_tsv:

        if resp_tsv:
            resp_tsv.write("index\tpromptID\tpairID\tgenre\tsentence1_binary_parse\tsentence2_binary_parse\t"
                           "sentence1_parse\tsentence2_parse\tsentence1\tsentence2\tlabel1\tgold_label\n")

        gt_tsv.write("index\tpromptID\tpairID\tgenre\tsentence1_binary_parse\tsentence2_binary_parse\t"
                     "sentence1_parse\tsentence2_parse\tsentence1\tsentence2\tlabel1\tgold_label\n")

        for i, (context_utterances, ground_truth, generated_responses) in \
                enumerate(ResponseFileReader(args.responses_file, args.n_methods)):
            premise = " ".join(context_utterances)

            promptId = rand.generate_random_digits()
            gt_tsv.write(
                "{index}\t{promptId}\t{pairId}\tdialogue\t()\t()\t()\t()\t{s1}\t{s2}\t{gold_label}\t{gold_label}\n".format(
                    index=idx, promptId=promptId, pairId=id, s1=premise, s2=ground_truth, gold_label="entailment"))
            gt_jsonl.write(
                '{{"pairID": {}, "genre": "dialogue", "gold_label": "{}", "sentence1": "{}", "sentence2": "{}"}}\n'
                    .format(id, "entailment", _escape(premise), _escape(ground_truth)))
            id += 1
            idx += 1

            if mean_human_judgment:
                for j, response in enumerate(generated_responses):
                    nli_label = get_entailment_label(mean_human_judgment[i * args.n_methods + j][2])

                    resp_jsonl.write(
                        '{{"pairID": {}, "genre": "dialogue", "gold_label": "{}", "sentence1": "{}", "sentence2": "{}"}}\n'
                            .format(id, nli_label, _escape(premise), _escape(response)))

                    promptId = rand.generate_random_digits()
                    resp_tsv.write(
                        "{index}\t{promptId}\t{pairId}\tdialogue\t()\t()\t()\t()\t{s1}\t{s2}\t{gold_label}\t{gold_label}\n".format(
                            index=idx, promptId=promptId, pairId=id, s1=premise, s2=response,
                            gold_label=nli_label))
                    id += 1
                    idx += 1

    if resp_jsonl:
        resp_jsonl.close()

    if resp_tsv:
        resp_tsv.close()


if __name__ == "__main__":
    main()
