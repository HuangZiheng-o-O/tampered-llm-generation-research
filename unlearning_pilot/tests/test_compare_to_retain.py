import unittest


try:
    import scipy  # noqa: F401
except ImportError:
    scipy = None

from unlearning_pilot.compare_to_retain import compare


def record(forget, retain):
    def group(values):
        return {"question_sha256": "same-data", "correct_answer_ranked_first": len(values),
                "ranking_details": [
                    {"question": f"q{index}", "correct_answer_nll": value,
                     "mean_wrong_answer_nll": 1.0}
                    for index, value in enumerate(values)]}

    return {"base": "llama", "seed": 43, "max_length": 192,
            "adapter_sha256": "hash", "results": {
                "forget": group(forget), "retain": group(retain)}}


@unittest.skipIf(scipy is None, "scipy unavailable")
class ComparisonTest(unittest.TestCase):
    def test_identical_distributions_have_quality_one(self):
        first = record([0.8, 1.2, 1.1], [0.5, 0.6])
        result = compare(first, first, first)
        self.assertEqual(result["pilot_forget_quality_ks_pvalue"], 1.0)
        self.assertEqual(result["pilot_forget_quality_ks_statistic"], 0.0)

    def test_rejects_different_question_order(self):
        first = record([0.8, 1.2, 1.1], [0.5, 0.6])
        other = record([0.8, 1.2, 1.1], [0.5, 0.6])
        other["results"]["forget"]["ranking_details"][0]["question"] = "different"
        with self.assertRaisesRegex(ValueError, "questions differ"):
            compare(first, other, first)
