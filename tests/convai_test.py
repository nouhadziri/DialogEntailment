import unittest
import os

from dialogentail import ConvAIReader
from dialogentail.util.files import get_containing_dir


class ConvAITest(unittest.TestCase):
    def test_reader_both(self):
        n_conversations = 0
        first_utterance, last_self_prior, last_partner_prior = None, None, None
        dataset_file = os.path.join(get_containing_dir(__file__), "test_files", "sample_both_original_no_cands.txt")
        for utterances, self_prior, partner_prior in ConvAIReader(dataset_file):
            if n_conversations == 0:
                first_utterance = utterances[0]
            n_conversations += 1
            last_self_prior = self_prior[-1]
            last_partner_prior = partner_prior[-1]

        self.assertEqual(n_conversations, 13)
        self.assertEqual(first_utterance, "hi , how are you doing ? i'm getting ready to do some cheetah chasing to stay in shape .")
        self.assertEqual(last_self_prior, "i listen to classic rock .")
        self.assertEqual(last_partner_prior, "i want to be a doctor when i graduate .")

    def test_reader_other(self):
        n_conversations = 0
        first_utterance, last_partner_prior = None, None
        self_prior_empty = True
        dataset_file = os.path.join(get_containing_dir(__file__), "test_files", "sample_other_revised_no_cands.txt")
        for utterances, self_prior, partner_prior in ConvAIReader(dataset_file):
            if n_conversations == 0:
                first_utterance = utterances[0]
            n_conversations += 1

            if self_prior:
                self_prior_empty = False

            last_partner_prior = partner_prior[-1]

        self.assertEqual(n_conversations, 41)
        self.assertEqual(first_utterance,
                         "hello what are doing today ?")
        self.assertEqual(last_partner_prior, "i eat way too much .")
        self.assertTrue(self_prior_empty)
