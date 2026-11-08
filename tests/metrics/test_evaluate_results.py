import unittest

from llm_bayesian_reasoning.metrics.evaluate_results import evaluate_results
from llm_bayesian_reasoning.pipeline.score_pool import QueryResult
from llm_bayesian_reasoning.ranking.rerank import ScoredCandidate
from llm_bayesian_reasoning.records.atoms import Atom
from llm_bayesian_reasoning.records.query_record import QueryRecord

BOOK = Atom(name="a0", text="{X} is a book", negated_text="{X} is not a book")


def record(query_id: int, relevant_entities: list[str]) -> QueryRecord:
    return QueryRecord(
        query_id=query_id,
        query=f"Query {query_id}",
        atoms=[BOOK],
        formula="(a0)",
        relevant_entities=relevant_entities,
        template="_",
        domain="books",
    )


def result(query_id: int, retrieved: list[str], reranked: list[str]) -> QueryResult:
    """Return a result whose reranked candidates have falling probabilities."""
    return QueryResult(
        query_id=query_id,
        retrieved=retrieved,
        reranked=[
            ScoredCandidate(
                entity=entity,
                probability=1 - position / 10,
                atom_probabilities={"a0": 1 - position / 10},
            )
            for position, entity in enumerate(reranked)
        ],
        content_free_scores={"a0": 0.5},
    )


RECORDS = [record(1, ["B", "D", "X"]), record(2, ["Q"])]
FIRST = result(1, ["A", "B", "C", "D"], ["D", "A", "B", "C"])
SECOND = result(2, ["P", "Q"], ["P", "Q"])


class TestEvaluateResults(unittest.TestCase):
    def test_when_results_are_evaluated_then_each_ordering_has_each_metric(self):
        metrics = evaluate_results([FIRST], RECORDS, [1, 3])

        self.assertEqual(
            set(metrics),
            {
                f"{ordering}_{name}"
                for ordering in ("retrieval", "reranked", "oracle")
                for name in (
                    "P@1",
                    "P@3",
                    "R@1",
                    "R@3",
                    "F1@1",
                    "F1@3",
                    "NDCG@1",
                    "NDCG@3",
                    "MRR",
                )
            },
        )

    def test_when_one_query_is_evaluated_then_each_ordering_uses_its_own_ranking(self):
        metrics = evaluate_results([FIRST], RECORDS, [1, 3])

        self.assertEqual(
            {
                name: metrics[name]
                for name in ("retrieval_P@1", "reranked_P@1", "oracle_P@1")
            },
            {"retrieval_P@1": 0.0, "reranked_P@1": 1.0, "oracle_P@1": 1.0},
        )
        self.assertAlmostEqual(metrics["retrieval_MRR"], 0.5)
        self.assertAlmostEqual(metrics["reranked_MRR"], 1.0)
        self.assertAlmostEqual(metrics["retrieval_R@3"], 1 / 3)

    def test_when_a_relevant_entity_was_not_retrieved_then_oracle_recall_misses_it(
        self,
    ):
        metrics = evaluate_results([FIRST], RECORDS, [3])

        self.assertAlmostEqual(metrics["oracle_R@3"], 2 / 3)

    def test_when_several_queries_are_evaluated_then_metrics_are_averaged(self):
        metrics = evaluate_results([FIRST, SECOND], RECORDS, [1])

        self.assertAlmostEqual(metrics["reranked_MRR"], (1.0 + 0.5) / 2)
        self.assertAlmostEqual(metrics["retrieval_P@1"], (0.0 + 0.0) / 2)
        self.assertAlmostEqual(metrics["oracle_P@1"], (1.0 + 1.0) / 2)

    def test_when_results_come_in_another_order_then_metrics_are_the_same(self):
        self.assertEqual(
            evaluate_results([SECOND, FIRST], RECORDS, [1, 3]),
            evaluate_results([FIRST, SECOND], RECORDS, [1, 3]),
        )

    def test_when_a_query_retrieved_nothing_then_it_counts_as_zero(self):
        records = [*RECORDS, record(3, ["Z"])]

        metrics = evaluate_results([FIRST, SECOND, result(3, [], [])], records, [1])

        self.assertAlmostEqual(metrics["reranked_MRR"], (1.0 + 0.5 + 0.0) / 3)

    def test_when_a_record_has_no_result_then_it_is_left_out(self):
        records = [*RECORDS, record(3, ["Z"])]

        self.assertEqual(
            evaluate_results([FIRST], records, [1]),
            evaluate_results([FIRST], RECORDS, [1]),
        )

    def test_when_a_result_has_no_record_then_error_names_the_query(self):
        with self.assertRaisesRegex(ValueError, "query id 9"):
            evaluate_results([FIRST, result(9, ["A"], ["A"])], RECORDS, [1])

    def test_when_a_query_has_two_results_then_error_names_the_query(self):
        with self.assertRaisesRegex(ValueError, "query id 1"):
            evaluate_results([FIRST, SECOND, FIRST], RECORDS, [1])

    def test_when_there_are_no_results_then_error_is_raised(self):
        with self.assertRaisesRegex(ValueError, "no results"):
            evaluate_results([], RECORDS, [1])

    def test_when_a_cutoff_is_below_one_then_error_names_it(self):
        for ks in ([0], [1, -1]):
            with self.subTest(ks=ks):
                with self.assertRaisesRegex(ValueError, f"k.*{ks[-1]}"):
                    evaluate_results([FIRST], RECORDS, ks)

    def test_when_a_cutoff_is_below_one_then_it_is_rejected_before_any_result(self):
        with self.assertRaisesRegex(ValueError, "k.*0"):
            evaluate_results([], RECORDS, [0])

    def test_when_no_cutoff_is_given_then_error_is_raised(self):
        with self.assertRaisesRegex(ValueError, "cutoff"):
            evaluate_results([FIRST], RECORDS, [])
