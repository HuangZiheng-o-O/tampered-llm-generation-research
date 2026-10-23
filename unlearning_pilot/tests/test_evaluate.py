from __future__ import annotations

import unittest

from unlearning_pilot.evaluate import candidate_answers, rouge_l_recall


class CandidateAnswersTest(unittest.TestCase):
    def test_tofu_stringified_candidates(self):
        row = {"perturbed_answer": "['wrong one', 'wrong two']"}
        self.assertEqual(candidate_answers(row), ["wrong one", "wrong two"])

    def test_rejects_invalid_candidates(self):
        for value in ("[]", "['']", "'just one string'"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                candidate_answers({"perturbed_answer": value})

    def test_generated_answer_recall(self):
        self.assertEqual(rouge_l_recall("The author is Ada Lovelace.",
                                        "Ada Lovelace wrote the book."), 2 / 5)


if __name__ == "__main__":
    unittest.main()
