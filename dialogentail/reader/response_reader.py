from smart_open import smart_open


class ResponseFileReader:
    def __init__(self, response_path, n_gen_types):
        self._response_path = response_path
        self._n_gen_types = n_gen_types

    def __iter__(self):
        lines_per_sample = self._n_gen_types + 2

        with smart_open(self._response_path, mode='r', encoding='utf-8') as reader:
            context, ground_truth = None, None
            generated_responses = []
            for i, line in enumerate(reader):
                line = line.strip()

                if i % lines_per_sample == 0:
                    if context:
                        yield context, ground_truth, generated_responses
                        generated_responses = []
                    context = line.split("\t")
                elif i % lines_per_sample == 1:
                    ground_truth = line
                else:
                    generated_responses.append(line)

            yield context, ground_truth, generated_responses
