import codecs
import csv

from ..util import files


def _esc(s):
    return s.replace('"', '\\"')


class SwagReader:
    """
    Iterates over the SWAG dataset and yields a quadruple with the format: (id, sentence1, options (a.k.a distractors), gold_sentence)
    For reading the training or validation data, "_full.csv" files should be used.
    """

    def __init__(self, swag_path):
        self._swag_path = swag_path

    def _check_if_test_file(self):
        f = files.get_file_name(self._swag_path)
        return f.startswith("test_") or f.startswith("test-") or f.endswith("test")

    def __iter__(self):
        is_test_file = self._check_if_test_file()
        id = 2 if is_test_file else 1
        s1 = 4 if is_test_file else 14
        ss = 5 if is_test_file else 15
        endings = [7, 8, 9, 10] if is_test_file else [4, 5, 6, 7]

        with codecs.getreader("utf-8")(open(self._swag_path, "rb")) as swag_file:
            csv_reader = csv.reader(swag_file, delimiter=',')

            for i, row in enumerate(csv_reader):
                if i == 0:
                    continue

                starting_sentence = _esc(row[s1].strip())

                distractors = [f"{_esc(row[ss])} {_esc(row[e])}" for e in endings if row[e].strip()]

                if not is_test_file:
                    gold_sentence = f"{_esc(row[15])} {_esc(row[3].strip())}"
                else:
                    gold_sentence = None

                yield row[id], starting_sentence, distractors, gold_sentence
