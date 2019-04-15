import codecs
import re


def _refine_sentence_end_marks(txt):
    if re.match(r".*\S\.$", txt):
        return txt[:-1] + " ."
    elif re.match(r".*\S!$", txt):
        return txt[:-1] + " !"
    else:
        return txt


class ConvAIReader:
    """
    Iterates over the ConvAI dataset and yields a conversation with its corresponding prior information (from the first interlocutor or the other).
    Each item would be a triple with the format: (utterances, self prior, partner prior)
    Note that this class only supports reading from "no_cands" files.
    """

    def __init__(self, convai_path):
        self._convai_path = convai_path

    def __iter__(self):
        with codecs.getreader("utf-8")(open(self._convai_path, "rb")) as convai_file:
            dialogue, self_prior, partner_prior = [], [], []
            last_dialogue_line = 0

            for i, line in enumerate(convai_file):
                line = line.strip()
                if not line:
                    continue

                dialog_line = int(line.split()[0])

                if last_dialogue_line > 0 and dialog_line == 1:
                    yield dialogue, self_prior, partner_prior
                    dialogue, self_prior, partner_prior = [], [], []

                if "your persona:" in line or "partner's persona:" in line:
                    fact = _refine_sentence_end_marks(' '.join(line.split()[3:]))

                    if "your persona:" in line:
                        self_prior.append(fact)
                    else:
                        partner_prior.append(fact)
                else:
                    msg, resp = tuple(line.strip().split('\t'))
                    dialogue.append(' '.join(msg.split()[1:]))
                    dialogue.append(resp)

                last_dialogue_line = dialog_line

            yield dialogue, self_prior, partner_prior
