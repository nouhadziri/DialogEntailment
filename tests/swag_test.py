import unittest
import os

from dialogentail import SwagReader
from dialogentail.util.files import get_containing_dir


class SwagTest(unittest.TestCase):
    def test_reader_val(self):
        n_samples = 0
        first_sent, last_gold = None, None
        dataset_file = os.path.join(get_containing_dir(__file__), "test_files", "swag_sample_val.csv")
        for id, start_sent, _, gold_sentence in SwagReader(dataset_file):
            if n_samples == 0:
                first_sent = start_sent
            n_samples += 1
            last_gold = gold_sentence

        self.assertEqual(n_samples, 20)
        self.assertEqual(first_sent, "Students lower their eyes nervously.")
        self.assertEqual(last_gold, "Then, the camel stops and the woman gets down from the camel.")

    def test_reader_test(self):
        n_samples = 0
        first_sent, last_option = None, None
        dataset_file = os.path.join(get_containing_dir(__file__), "test_files", "swag_sample_test.csv")
        for id, start_sent, options, _ in SwagReader(dataset_file):
            if n_samples == 0:
                first_sent = start_sent
            n_samples += 1
            last_option = options[-1]

        self.assertEqual(n_samples, 40)
        self.assertEqual(first_sent, "A person shows the bottom of a large dust mop.")
        self.assertEqual(last_option, "A guy speaks into the microphone.")
