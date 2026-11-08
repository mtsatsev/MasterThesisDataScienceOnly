import math
import unittest

from llm_bayesian_reasoning.metrics.ranking_metrics import (
    f1_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["A", "B", "C", "D", "E"]
RELEVANT = ["B", "D", "X"]
METRICS_AT_K = (precision_at_k, recall_at_k, f1_at_k, ndcg_at_k)


class TestRankingMetrics(unittest.TestCase):
    def test_when_k_is_below_one_then_error_names_k(self):
        for metric in METRICS_AT_K:
            for k in (0, -1):
                with self.subTest(metric=metric.__name__, k=k):
                    with self.assertRaisesRegex(ValueError, f"k.*{k}"):
                        metric(RANKED, RELEVANT, k)

    def test_when_ranking_has_hits_then_precision_is_hits_over_k(self):
        for k, expected in ((1, 0.0), (2, 1 / 2), (3, 1 / 3), (5, 2 / 5)):
            with self.subTest(k=k):
                self.assertAlmostEqual(precision_at_k(RANKED, RELEVANT, k), expected)

    def test_when_k_exceeds_the_ranking_then_precision_still_divides_by_k(self):
        self.assertAlmostEqual(precision_at_k(RANKED, RELEVANT, 10), 2 / 10)

    def test_when_ranking_has_hits_then_recall_is_hits_over_all_relevant(self):
        for k, expected in ((1, 0.0), (3, 1 / 3), (5, 2 / 3), (10, 2 / 3)):
            with self.subTest(k=k):
                self.assertAlmostEqual(recall_at_k(RANKED, RELEVANT, k), expected)

    def test_when_ranking_has_hits_then_f1_is_harmonic_mean_of_precision_and_recall(
        self,
    ):
        for k, expected in ((3, 1 / 3), (5, 0.5)):
            with self.subTest(k=k):
                self.assertAlmostEqual(f1_at_k(RANKED, RELEVANT, k), expected)

    def test_when_ranking_has_hits_then_ndcg_discounts_them_by_position(self):
        dcg = 1 / math.log2(3) + 1 / math.log2(5)
        ideal_dcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)

        self.assertAlmostEqual(ndcg_at_k(RANKED, RELEVANT, 5), dcg / ideal_dcg)

    def test_when_hits_are_ranked_below_k_then_ndcg_ignores_them(self):
        for k, expected in (
            (1, 0.0),
            (3, (1 / math.log2(3)) / (1 + 1 / math.log2(3) + 1 / math.log2(4))),
        ):
            with self.subTest(k=k):
                self.assertAlmostEqual(ndcg_at_k(RANKED, RELEVANT, k), expected)

    def test_when_k_is_below_the_number_of_relevant_then_ideal_ndcg_has_k_hits(self):
        ranked = ["B", "D", "A"]

        self.assertAlmostEqual(ndcg_at_k(ranked, RELEVANT, 2), 1.0)

    def test_when_first_hit_is_second_then_reciprocal_rank_is_one_half(self):
        self.assertAlmostEqual(reciprocal_rank(RANKED, RELEVANT), 0.5)

    def test_when_first_hit_is_deep_in_the_ranking_then_reciprocal_rank_is_not_cut(
        self,
    ):
        ranked = [f"Miss {index}" for index in range(59)] + ["B"]

        self.assertAlmostEqual(reciprocal_rank(ranked, RELEVANT), 1 / 60)

    def test_when_every_relevant_entity_is_ranked_first_then_metrics_are_perfect(
        self,
    ):
        ranked = ["B", "D", "X", "A", "C"]

        for metric in (recall_at_k, f1_at_k, ndcg_at_k):
            with self.subTest(metric=metric.__name__):
                self.assertAlmostEqual(metric(ranked, RELEVANT, 3), 1.0)
        self.assertAlmostEqual(precision_at_k(ranked, RELEVANT, 3), 1.0)
        self.assertAlmostEqual(reciprocal_rank(ranked, RELEVANT), 1.0)

    def test_when_no_relevant_entity_is_ranked_then_metrics_are_zero(self):
        for ranked in (["A", "C", "E"], []):
            with self.subTest(ranked=ranked):
                for metric in METRICS_AT_K:
                    with self.subTest(metric=metric.__name__):
                        self.assertEqual(metric(ranked, RELEVANT, 3), 0.0)
                self.assertEqual(reciprocal_rank(ranked, RELEVANT), 0.0)

    def test_when_nothing_is_relevant_then_error_is_raised(self):
        for metric in METRICS_AT_K:
            with self.subTest(metric=metric.__name__):
                with self.assertRaisesRegex(ValueError, "relevant"):
                    metric(RANKED, [], 3)
        with self.assertRaisesRegex(ValueError, "relevant"):
            reciprocal_rank(RANKED, [])
