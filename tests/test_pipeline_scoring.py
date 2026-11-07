"""Tests for candidate scoring through ProbLog and for the suite configs (no model)."""

import json
from pathlib import Path

import pytest

from llm_bayesian_reasoning.pipeline.config import (
    EstimatorConfig,
    ExperimentSuiteConfig,
    PipelineConfig,
)
from llm_bayesian_reasoning.pipeline.pipeline import (
    CONTENT_FREE_KEY,
    run_pipeline,
    score_candidate_documents,
)
from llm_bayesian_reasoning.problog_models.problog_models import (
    ProblogAtom,
    ProblogFormula,
)
from llm_bayesian_reasoning.retrievers.document import RetrievalResult, ScoredDocument


class _FixedEstimator:
    """Returns preset probabilities per atom, with 0.5 for the content-free input."""

    def __init__(self, probabilities: dict[str, float]):
        self.probabilities = probabilities

    def score_probability(self, predicates, entity):
        return [
            ProblogAtom(
                atom=atom.atom,
                probability=0.5 if entity == "N/A" else self.probabilities[atom.atom],
                context=atom.context,
            )
            for atom in predicates
        ]


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("(({X} is a) AND ({X} is b))", 0.8 * 0.5),
        ("(({X} is a) OR ({X} is b))", 1 - 0.2 * 0.5),
        ("(({X} is a) AND (NOT({X} is b)))", 0.8 * 0.5),
        ("(({X} is a) AND (NOT({X} is c)))", 0.8 * 0.9),
    ],
)
def test_scoring_composes_atoms_through_problog(formula, expected):
    probabilities = {"{X} is a": 0.8, "{X} is b": 0.5, "{X} is c": 0.1}
    atoms = [ProblogAtom(atom=a) for a in probabilities]
    atom_scores: dict = {}
    scores = score_candidate_documents(
        atoms=atoms,
        formula=ProblogFormula(formula=formula),
        candidate_documents=[ScoredDocument(title="E (film)", text="doc", score=1.0)],
        estimator=_FixedEstimator(probabilities),
        atom_scores=atom_scores,
    )
    assert scores["E (film)"] == pytest.approx(expected)
    assert atom_scores["E (film)"] == probabilities
    assert set(atom_scores[CONTENT_FREE_KEY]) == set(probabilities)


@pytest.mark.parametrize(
    "path",
    sorted(
        p
        for p in (Path(__file__).parents[1] / "scripts").glob("*.json")
        if "variants" in json.loads(p.read_text())
    ),
    ids=lambda p: p.name,
)
def test_suite_configs_are_valid(path):
    ExperimentSuiteConfig.model_validate(json.loads(path.read_text()))


class _FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k):
        documents = self.documents[:top_k]
        results = [(i, d.score) for i, d in enumerate(documents)]
        return RetrievalResult(results=results, documents=documents)


_DOCUMENTS = [
    ScoredDocument(title="Wrong", text="doc", score=2.0),
    ScoredDocument(title="Right", text="doc", score=1.0),
]


class _ByEntityEstimator:
    def score_probability(self, predicates, entity):
        return [
            ProblogAtom(atom=a.atom, probability=0.9 if entity == "Right" else 0.1)
            for a in predicates
        ]


def test_run_pipeline_reranks_and_writes_results_and_metrics(tmp_path):
    output_path = tmp_path / "run" / "results.jsonl"
    config = PipelineConfig(
        top_n=2,
        top_k=1,
        index_path=tmp_path / "index",
        output_path=output_path,
        estimator_config=EstimatorConfig(device="cpu"),
    )
    data = {
        1: {
            "query": "q",
            "atoms": [ProblogAtom(atom="{X} is a")],
            "problog_formula": ProblogFormula(formula="({X} is a)"),
        }
    }
    out = run_pipeline(
        data, _FakeRetriever(_DOCUMENTS), _ByEntityEstimator(), config, {1: ["Right"]}
    )
    assert out["results"][1]["ranked_entities"] == ["Right", "Wrong"]
    assert out["metrics"]["reranked_MRR"] == 1.0
    assert out["metrics"]["retrieval_MRR"] == 0.5

    [row] = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert row["id"] == 1 and CONTENT_FREE_KEY in row["atom_scores"]
    assert (output_path.parent / "results_metrics.json").exists()

    # A rerun resumes from the checkpoint instead of scoring the record again.
    again = run_pipeline(data, _FakeRetriever(_DOCUMENTS), None, config, {1: ["Right"]})
    assert again["results"][1]["ranked_entities"] == ["Right", "Wrong"]
    assert len(output_path.read_text().splitlines()) == 1


def test_experiment_suite_writes_variant_results_metrics_and_manifest(
    tmp_path, monkeypatch
):
    import importlib.util
    import sys

    path = Path(__file__).parents[1] / "scripts" / "run_experiment_suite.py"
    spec = importlib.util.spec_from_file_location("run_experiment_suite", path)
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    monkeypatch.setattr(
        suite, "build_or_load_retriever", lambda **_: _FakeRetriever(_DOCUMENTS)
    )
    monkeypatch.setattr(
        suite, "load_model_and_tokenizer_from_config", lambda _: (None, None)
    )
    monkeypatch.setattr(
        suite,
        "create_estimator_from_components",
        lambda *_: _ByEntityEstimator(),
    )

    record = {
        "id": 1,
        "query": "q",
        "parsed": {
            "atoms": ["{x} is a"],
            "negated_atoms": ["{x} is not a"],
            "logical query": "({x} is a)",
        },
        "metadata": {"relevance_ratings": {"Right": ["Definitely relevant"]}},
    }
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(json.dumps(record) + "\n")
    config = {
        "data_file": str(data_file),
        "results_root": str(tmp_path / "results"),
        "retrieval_cache_dir": str(tmp_path / "cache"),
        "use_metadata_ground_truth": True,
        "retrievers": [
            {
                "name": "bm25-main",
                "retriever_type": "BM25",
                "index_path": str(tmp_path / "index"),
                "documents_path": str(tmp_path / "docs.jsonl"),
                "retrieval_pool_size": 2,
            }
        ],
        "variants": [
            {
                "name": "bm25_x",
                "retriever_name": "bm25-main",
                "top_n": 2,
                "top_k": 1,
                "estimator_config": {
                    "device": "cpu",
                    "estimator_type": "YesNoContinuation",
                },
            }
        ],
    }
    config_file = tmp_path / "suite.json"
    config_file.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["x", "--suite-config", str(config_file)])
    suite.main()

    [suite_dir] = list((tmp_path / "results").iterdir())
    [row] = [
        json.loads(line)
        for line in (suite_dir / "bm25_x" / "results.jsonl").read_text().splitlines()
    ]
    assert row["ranked_entities"] == ["Right", "Wrong"]
    assert CONTENT_FREE_KEY in row["atom_scores"]
    metrics = json.loads((suite_dir / "bm25_x" / "metrics.json").read_text())
    assert metrics["reranked_MRR"] == 1.0
    manifest = json.loads((suite_dir / "manifest.json").read_text())
    assert manifest["bm25_x"]["metrics"] == metrics
    assert (suite_dir / "suite_config.json").exists()
