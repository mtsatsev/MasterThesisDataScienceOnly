#!/usr/bin/env python3
"""Run a grid of retrieval-and-estimation experiments from one suite config.

The suite runner persists retrieval candidate pools per retriever and query,
loads each retriever index once, and reuses a single loaded LLM across
compatible estimator variants.
"""

import argparse
import json
import logging
from pathlib import Path
from llm_bayesian_reasoning.pipeline.config import ExperimentSuiteConfig
from llm_bayesian_reasoning.pipeline.suite_execution import run_experiment_suite

logger = logging.getLogger("run_experiment_suite")


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
    run_experiment_suite(
        suite_config=suite_config,
        config_payload=config_payload,
    )


if __name__ == "__main__":
    main()
