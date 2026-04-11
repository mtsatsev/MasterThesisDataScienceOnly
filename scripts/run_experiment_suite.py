#!/usr/bin/env python3
"""Run a grid of retrieval-and-estimation experiments from one suite config.

The suite runner persists retrieval candidate pools per retriever and query,
loads each retriever index once, and reuses a single loaded LLM across
compatible estimator variants.
"""

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from llm_bayesian_reasoning.estimators.factory import (
    create_estimator_from_components,
    load_model_and_tokenizer_from_config,
)
from llm_bayesian_reasoning.pipeline.config import (
    EstimatorType,
    ExperimentSuiteConfig,
    ExperimentVariantConfig,
    RetrieverConfig,
)
from llm_bayesian_reasoning.pipeline.metrics import compute_metrics
from llm_bayesian_reasoning.pipeline.pipeline import (
    build_record_result,
    create_logic_backend,
    score_candidate_documents,
)
from llm_bayesian_reasoning.problog_models.problog_models import (
    ProblogAtom,
    ProblogFormula,
)
from llm_bayesian_reasoning.retrievers.cache_manager import (
    RetrievalCacheManager,
    build_retriever_cache_key,
)
from llm_bayesian_reasoning.retrievers.document import RetrievalResult
from llm_bayesian_reasoning.retrievers.factory import build_or_load_retriever

logger = logging.getLogger("run_experiment_suite")


def _normalize_placeholder(value: str) -> str:
    return value.replace("{x}", "{X}")


def _build_atom(atom: str) -> ProblogAtom:
    return ProblogAtom(atom=_normalize_placeholder(atom))


def _safe_name(value: str) -> str:
    return Path(value).stem.replace("/", "_").replace(" ", "_")


def _make_suite_results_folder(root: Path) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_dir = root / f"{timestamp}_experiment_suite"
    results_dir.mkdir(parents=True, exist_ok=False)
    return results_dir


def _load_preprocessed_for_suite(
    path: Path,
    limit: int | None = None,
    use_metadata_ground_truth: bool = False,
) -> tuple[dict, dict | None]:
    """Load preprocessed records once so variants can materialize their atom view later."""
    data: dict = {}
    ground_truth: dict = {}

    with path.open(encoding="utf-8") as file_handle:
        for index, line in enumerate(file_handle):
            if limit is not None and index >= limit:
                break
            try:
                document = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_id = document.get("id")
            if record_id is None:
                continue

            query = document.get("query") or document.get("original_query") or ""
            parsed = document.get("parsed", {})
            atoms_raw = parsed.get("atoms", [])
            negated_atoms_raw = parsed.get("negated_atoms", [])
            logical = parsed.get("logical query") or parsed.get("logical_query") or ""

            atoms_text = [atom for atom in atoms_raw if isinstance(atom, str)]
            negated_atoms_text = [
                atom for atom in negated_atoms_raw if isinstance(atom, str)
            ]

            data[record_id] = {
                "query": query,
                "atoms": [_build_atom(atom) for atom in atoms_text],
                "negated_atoms": [_build_atom(atom) for atom in negated_atoms_text],
                "problog_formula": ProblogFormula(
                    formula=_normalize_placeholder(logical)
                ),
            }

            if use_metadata_ground_truth:
                metadata = document.get("metadata", {})
                relevance = metadata.get("relevance_ratings") or {}
                if isinstance(relevance, dict):
                    ground_truth[record_id] = list(relevance.keys())

    return data, ground_truth if use_metadata_ground_truth else None


def _build_variant_atoms(
    record: dict,
    estimator_type: EstimatorType,
) -> list[ProblogAtom] | list[tuple[ProblogAtom, ProblogAtom]]:
    positive_atoms: list[ProblogAtom] = record["atoms"]
    negated_atoms: list[ProblogAtom] = record["negated_atoms"]

    if estimator_type == EstimatorType.LIKELIHOOD_BASED_CONTRASTIVE:
        if not negated_atoms:
            raise ValueError(
                "Contrastive variants require parsed.negated_atoms in the dataset"
            )
        if len(positive_atoms) != len(negated_atoms):
            raise ValueError(
                "Contrastive variants require the same number of atoms and negated_atoms"
            )
        return list(zip(positive_atoms, negated_atoms))

    return positive_atoms


def _model_signature(variant: ExperimentVariantConfig) -> tuple[str, str, str]:
    estimator_config = variant.estimator_config
    return (
        estimator_config.model_name,
        estimator_config.device,
        repr(estimator_config.quantization),
    )


def _scoring_signature(
    variant: ExperimentVariantConfig,
) -> tuple[str, str, bool, str, str, float]:
    estimator_config = variant.estimator_config
    return (
        variant.logic_backend.value,
        estimator_config.estimator_type.value,
        estimator_config.include_retrieved_text,
        estimator_config.true_token,
        estimator_config.false_token,
        estimator_config.contrastive_temperature,
    )


def _write_result(file_handle, record_id, record_result: dict) -> None:
    row = {"id": record_id, **record_result}
    file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    file_handle.flush()


def _load_retrieval_pool(
    cache_manager: RetrievalCacheManager,
    retriever,
    retriever_cache_key: str,
    query: str,
    requested_pool_size: int,
    retriever_config: RetrieverConfig,
) -> RetrievalResult:
    cached = cache_manager.get(
        query=query,
        retriever_key=retriever_cache_key,
        requested_pool_size=requested_pool_size,
        retriever=retriever,
    )
    if cached is not None:
        return cached

    retrieval_result = retriever.retrieve(query, top_k=requested_pool_size)
    return cache_manager.set(
        query=query,
        retriever_key=retriever_cache_key,
        retrieval_result=retrieval_result,
        metadata={
            "retriever_name": retriever_config.name,
            "retriever_type": retriever_config.retriever_type.value,
            "index_path": str(retriever_config.index_path),
            "retriever_model_name": retriever_config.retriever_model_name,
        },
    )


def _prepare_variant_outputs(
    results_dir: Path,
    variants: list[ExperimentVariantConfig],
) -> tuple[dict[str, dict], dict[str, dict]]:
    variant_states: dict[str, dict] = {}
    manifest: dict[str, dict] = {}

    for variant in variants:
        variant_dir = results_dir / _safe_name(variant.name)
        variant_dir.mkdir(parents=True, exist_ok=False)
        results_path = variant_dir / "results.jsonl"
        metrics_path = variant_dir / "metrics.json"
        variant_states[variant.name] = {
            "handle": results_path.open("w", encoding="utf-8"),
            "results": {},
            "results_path": results_path,
            "metrics_path": metrics_path,
            "variant": variant,
        }
        manifest[variant.name] = {
            "results_path": str(results_path),
            "metrics_path": str(metrics_path),
            "retriever_name": variant.retriever_name,
            "top_n": variant.top_n,
            "top_k": variant.top_k,
            "logic_backend": variant.logic_backend.value,
            "estimator_type": variant.estimator_config.estimator_type.value,
            "include_retrieved_text": variant.estimator_config.include_retrieved_text,
            "model_name": variant.estimator_config.model_name,
        }
    return variant_states, manifest


def _get_candidate_pool_size(
    retriever_config: RetrieverConfig,
    variants: list[ExperimentVariantConfig],
) -> int:
    requested_top_n = max(variant.top_n for variant in variants)
    if retriever_config.retrieval_pool_size is None:
        return requested_top_n
    if retriever_config.retrieval_pool_size < requested_top_n:
        logger.info(
            "Expanding retrieval pool for %s from configured %d to required %d",
            retriever_config.name,
            retriever_config.retrieval_pool_size,
            requested_top_n,
        )
    return max(retriever_config.retrieval_pool_size, requested_top_n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a multi-variant retrieval + scoring experiment suite"
    )
    parser.add_argument("--suite-config", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_payload = json.loads(args.suite_config.read_text(encoding="utf-8"))
    suite_config = ExperimentSuiteConfig.model_validate(config_payload)

    suite_config.results_root.mkdir(parents=True, exist_ok=True)
    suite_results_dir = _make_suite_results_folder(suite_config.results_root)
    (suite_results_dir / "suite_config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )

    logger.info("Loading preprocessed data from %s", suite_config.data_file)
    data, ground_truth = _load_preprocessed_for_suite(
        suite_config.data_file,
        limit=suite_config.limit,
        use_metadata_ground_truth=suite_config.use_metadata_ground_truth,
    )
    ground_truth_map: dict[int | str, list[str]] = ground_truth or {}
    logger.info("Loaded %d records for the experiment suite", len(data))

    variant_states, manifest = _prepare_variant_outputs(
        suite_results_dir,
        suite_config.variants,
    )

    variants_by_retriever: dict[str, list[ExperimentVariantConfig]] = defaultdict(list)
    for variant in suite_config.variants:
        variants_by_retriever[variant.retriever_name].append(variant)

    try:
        for retriever_config in suite_config.retrievers:
            retriever_variants = variants_by_retriever.get(retriever_config.name, [])
            if not retriever_variants:
                continue

            retrieval_pool_size = _get_candidate_pool_size(
                retriever_config,
                retriever_variants,
            )
            logger.info(
                "Preparing retriever %s (%s) with retrieval pool size %d",
                retriever_config.name,
                retriever_config.retriever_type.value,
                retrieval_pool_size,
            )
            retriever = build_or_load_retriever(
                documents_path=retriever_config.documents_path,
                index_path=retriever_config.index_path,
                retriever_type=retriever_config.retriever_type,
                batch_size=retriever_config.batch_size,
                limit=retriever_config.index_limit,
                retriever_model_name=retriever_config.retriever_model_name,
            )
            cache_manager = RetrievalCacheManager(
                suite_config.retrieval_cache_dir / _safe_name(retriever_config.name)
            )
            retriever_cache_key = build_retriever_cache_key(
                retriever_config.retriever_type.value,
                retriever_config.index_path,
                retriever_config.retriever_model_name,
            )
            retrieval_pool_cache: dict[int | str, RetrievalResult] = {}

            variants_by_model: dict[
                tuple[str, str, str], list[ExperimentVariantConfig]
            ] = defaultdict(list)
            for variant in retriever_variants:
                variants_by_model[_model_signature(variant)].append(variant)

            for model_variants in variants_by_model.values():
                reference_variant = model_variants[0]
                logger.info(
                    "Loading shared model resources for %s",
                    reference_variant.estimator_config.model_name,
                )
                model, tokenizer = load_model_and_tokenizer_from_config(
                    reference_variant.estimator_config
                )

                variants_by_scoring: dict[
                    tuple[str, str, bool, str, str, float],
                    list[ExperimentVariantConfig],
                ] = defaultdict(list)
                for variant in model_variants:
                    variants_by_scoring[_scoring_signature(variant)].append(variant)

                for scoring_variants in variants_by_scoring.values():
                    scoring_variant = scoring_variants[0]
                    scoring_top_n = max(variant.top_n for variant in scoring_variants)
                    logger.info(
                        "Scoring variant family rooted at %s (retriever=%s, top_n=%d)",
                        scoring_variant.name,
                        retriever_config.name,
                        scoring_top_n,
                    )
                    estimator = create_estimator_from_components(
                        scoring_variant.estimator_config,
                        model,
                        tokenizer,
                    )
                    logic_backend = create_logic_backend(scoring_variant.logic_backend)

                    for record_id, record in data.items():
                        query = record["query"]
                        if record_id not in retrieval_pool_cache:
                            retrieval_pool_cache[record_id] = _load_retrieval_pool(
                                cache_manager=cache_manager,
                                retriever=retriever,
                                retriever_cache_key=retriever_cache_key,
                                query=query,
                                requested_pool_size=retrieval_pool_size,
                                retriever_config=retriever_config,
                            )

                        candidate_pool = retrieval_pool_cache[record_id]
                        candidate_documents = candidate_pool.documents[:scoring_top_n]
                        atoms = _build_variant_atoms(
                            record,
                            scoring_variant.estimator_config.estimator_type,
                        )
                        entity_scores = score_candidate_documents(
                            atoms=atoms,
                            formula=record["problog_formula"],
                            candidate_documents=candidate_documents,
                            estimator=estimator,
                            logic_backend=logic_backend,
                            include_retrieved_text=scoring_variant.estimator_config.include_retrieved_text,
                            record_id=record_id,
                        )

                        for variant in scoring_variants:
                            variant_candidate_documents = candidate_pool.documents[
                                : variant.top_n
                            ]
                            candidate_entities = [
                                document.title
                                for document in variant_candidate_documents
                            ]
                            relevant = None
                            if record_id in ground_truth_map:
                                relevant = set(ground_truth_map[record_id])
                            record_result, _ = build_record_result(
                                query=query,
                                atoms=atoms,
                                candidate_entities=candidate_entities,
                                entity_scores=entity_scores,
                                top_k=variant.top_k,
                                relevant=relevant,
                            )
                            variant_state = variant_states[variant.name]
                            variant_state["results"][record_id] = record_result
                            _write_result(
                                variant_state["handle"],
                                record_id,
                                record_result,
                            )

        for variant_name, state in variant_states.items():
            state["handle"].close()
            variant = state["variant"]
            metrics: dict[str, float] = {}
            if ground_truth is not None:
                metrics = compute_metrics(
                    state["results"],
                    ground_truth_map,
                    variant.top_k,
                )
                state["metrics_path"].write_text(
                    json.dumps(metrics, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            manifest[variant_name]["metrics"] = metrics

    finally:
        for state in variant_states.values():
            if not state["handle"].closed:
                state["handle"].close()

    (suite_results_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Experiment suite finished. Results folder: %s", suite_results_dir)


if __name__ == "__main__":
    main()
