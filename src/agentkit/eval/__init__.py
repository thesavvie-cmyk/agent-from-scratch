"""Evaluation framework — dataset, traces, rubrics, judge, runner (block 24)."""
from agentkit.eval.dataset import EvalCase, EvalDataset
from agentkit.eval.judge import Verdict, judge_pairwise, judge_single
from agentkit.eval.rubrics import ADVERSARIAL_RUBRICS, ALL_RUBRICS, Rubric
from agentkit.eval.runner import EvalReport, EvalResult, EvalRunner
from agentkit.eval.traces import TraceFeatures, extract_features

__all__ = [
    "ADVERSARIAL_RUBRICS",
    "ALL_RUBRICS",
    "EvalCase",
    "EvalDataset",
    "EvalReport",
    "EvalResult",
    "EvalRunner",
    "Rubric",
    "TraceFeatures",
    "Verdict",
    "extract_features",
    "judge_pairwise",
    "judge_single",
]
