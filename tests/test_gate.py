"""Unit tests for the eval quality gate (block 25)."""
from __future__ import annotations

from agentkit.eval.gate import GateConfig, check_gate
from agentkit.eval.runner import EvalReport, EvalResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_report(verdicts: dict[str, str], rubric: str = "answer_relevance") -> EvalReport:
    """Build a minimal EvalReport from {case_id: verdict} dict."""
    results = [
        EvalResult(
            case_id=cid,
            category="core",
            output="test",
            rubric_name=rubric,
            verdict=v,
            reasoning="",
        )
        for cid, v in verdicts.items()
    ]
    return EvalReport(
        run_id="test",
        timestamp="2026-01-01T00:00:00Z",
        total_cases=len(verdicts),
        total_rubrics=1,
        results=results,
    )


def _mixed_report(case_ids: list[str], pass_ids: set[str], rubric: str = "answer_relevance") -> EvalReport:
    verdicts = {cid: ("PASS" if cid in pass_ids else "FAIL") for cid in case_ids}
    return _make_report(verdicts, rubric)


# ── No baseline: absolute thresholds ─────────────────────────────────────────


class TestNoBaseline:
    def test_above_threshold_passes(self):
        report = _make_report({f"c{i}": "PASS" for i in range(8)})
        report.results += [
            EvalResult("c8", "core", "", "answer_relevance", "FAIL", "")
            for _ in range(2)
        ]
        config = GateConfig(thresholds={"answer_relevance": 0.70})
        result = check_gate(report, None, config)
        assert result.passed

    def test_below_threshold_blocks(self):
        verdicts = {f"c{i}": ("PASS" if i < 4 else "FAIL") for i in range(10)}
        report = _make_report(verdicts)
        config = GateConfig(thresholds={"answer_relevance": 0.60})
        result = check_gate(report, None, config)
        assert not result.passed
        assert any("answer_relevance" in b for b in result.blockers)

    def test_no_threshold_configured_passes(self):
        verdicts = {f"c{i}": "FAIL" for i in range(10)}
        report = _make_report(verdicts)
        config = GateConfig(thresholds={})  # no threshold for this rubric
        result = check_gate(report, None, config)
        assert result.passed  # no threshold = no block

    def test_empty_report_passes(self):
        report = EvalReport("r", "t", 0, 0, results=[])
        result = check_gate(report, None)
        assert result.passed


# ── With baseline: tolerance ──────────────────────────────────────────────────


class TestWithBaseline:
    def _make_pair(self, base_pass: int, cur_pass: int, n: int = 10):
        base = _mixed_report([f"c{i}" for i in range(n)], {f"c{i}" for i in range(base_pass)})
        cur = _mixed_report([f"c{i}" for i in range(n)], {f"c{i}" for i in range(cur_pass)})
        return base, cur

    def test_no_drop_passes(self):
        base, cur = self._make_pair(8, 8)
        result = check_gate(cur, base, GateConfig(tolerance=0.10))
        assert result.passed
        assert not result.blockers

    def test_improvement_passes(self):
        base, cur = self._make_pair(6, 8)
        result = check_gate(cur, base, GateConfig(tolerance=0.10))
        assert result.passed
        assert not result.warnings

    def test_small_drop_warns_not_blocks(self):
        # Drop of 1/20 = 5% < tolerance 10% → warn, not block
        n = 20
        base = _mixed_report([f"c{i}" for i in range(n)], {f"c{i}" for i in range(16)})  # 80%
        cur = _mixed_report([f"c{i}" for i in range(n)], {f"c{i}" for i in range(15)})   # 75% (drop=5%)
        config = GateConfig(tolerance=0.10)
        result = check_gate(cur, base, config)
        assert result.passed
        assert result.warnings

    def test_large_drop_blocks(self):
        # Drop of 3/10 = 30% > 10% tolerance
        base, cur = self._make_pair(9, 6)
        config = GateConfig(tolerance=0.10)
        result = check_gate(cur, base, config)
        assert not result.passed
        assert result.blockers

    def test_multi_rubric_one_blocks(self):
        """One rubric dropping too much should block even if another is fine."""
        rel_base = _mixed_report([f"c{i}" for i in range(10)], {f"c{i}" for i in range(9)}, "answer_relevance")
        rel_cur = _mixed_report([f"c{i}" for i in range(10)], {f"c{i}" for i in range(5)}, "answer_relevance")
        fact_base = _mixed_report([f"c{i}" for i in range(10)], {f"c{i}" for i in range(8)}, "factual_accuracy")
        fact_cur = _mixed_report([f"c{i}" for i in range(10)], {f"c{i}" for i in range(8)}, "factual_accuracy")

        # Combine results into single reports
        from agentkit.eval.runner import EvalReport
        base = EvalReport("b", "t", 10, 2, results=rel_base.results + fact_base.results)
        cur = EvalReport("c", "t", 10, 2, results=rel_cur.results + fact_cur.results)

        result = check_gate(cur, base, GateConfig(tolerance=0.10))
        assert not result.passed
        assert any("answer_relevance" in b for b in result.blockers)
        assert not any("factual_accuracy" in b for b in result.blockers)


# ── must_pass_ids ─────────────────────────────────────────────────────────────


class TestMustPass:
    def test_must_pass_all_pass(self):
        verdicts = {"adv-001": "PASS", "adv-002": "PASS", "c1": "FAIL"}
        report = _make_report(verdicts, "adversarial_resistance")
        config = GateConfig(must_pass_ids=["adv-001", "adv-002"], thresholds={})
        result = check_gate(report, None, config)
        assert result.passed
        assert not result.must_pass_violations

    def test_must_pass_one_fails_blocks(self):
        verdicts = {"adv-001": "PASS", "adv-002": "FAIL", "c1": "PASS"}
        report = _make_report(verdicts, "adversarial_resistance")
        config = GateConfig(must_pass_ids=["adv-001", "adv-002"], thresholds={})
        result = check_gate(report, None, config)
        assert not result.passed
        assert "adv-002" in result.must_pass_violations

    def test_must_pass_not_in_report_is_ignored(self):
        """A must_pass ID not present in the report does not cause a block."""
        verdicts = {"adv-001": "PASS"}
        report = _make_report(verdicts, "adversarial_resistance")
        config = GateConfig(must_pass_ids=["adv-001", "adv-999"], thresholds={})
        result = check_gate(report, None, config)
        assert result.passed  # adv-999 not in report, cannot block

    def test_must_pass_blocks_regardless_of_tolerance(self):
        """must_pass overrides tolerance — no forgiveness."""
        base_verdicts = {"adv-001": "PASS"}
        cur_verdicts = {"adv-001": "FAIL"}
        base = _make_report(base_verdicts, "adversarial_resistance")
        cur = _make_report(cur_verdicts, "adversarial_resistance")
        config = GateConfig(
            must_pass_ids=["adv-001"],
            tolerance=1.0,  # maximum tolerance — still blocked
            thresholds={},
        )
        result = check_gate(cur, base, config)
        assert not result.passed


# ── GateResult output ─────────────────────────────────────────────────────────


class TestGateResultOutput:
    def test_summary_text_pass(self):
        report = _make_report({"c1": "PASS", "c2": "PASS"})
        result = check_gate(report, None, GateConfig(thresholds={}))
        text = result.summary_text()
        assert "PASS" in text

    def test_summary_text_fail(self):
        report = _make_report({"c1": "FAIL", "c2": "FAIL"})
        result = check_gate(report, None, GateConfig(thresholds={"answer_relevance": 0.80}))
        text = result.summary_text()
        assert "FAIL" in text
        assert "BLOCK" in text

    def test_summary_markdown_contains_table(self):
        report = _make_report({"c1": "PASS", "c2": "FAIL"})
        result = check_gate(report, None, GateConfig(thresholds={}))
        md = result.summary_markdown()
        assert "|" in md   # markdown table

    def test_to_dict_round_trip(self):
        import json
        report = _make_report({"c1": "PASS"})
        result = check_gate(report, None)
        d = result.to_dict()
        json_str = json.dumps(d)  # should not raise
        data = json.loads(json_str)
        assert "passed" in data
        assert "blockers" in data


# ── Gate blocks on deliberate degradation ─────────────────────────────────────


class TestDegradationDetection:
    """Verify that deliberately worsening quality is detected.

    This is the key scenario the user requested: 'гейт корректно блокирует
    при намеренно ухудшенном промпте'. We simulate it by building a good
    baseline (80%) and a degraded current report (40%).
    """

    def test_deliberate_degradation_blocked(self):
        n = 20
        baseline = _mixed_report(
            [f"task-{i}" for i in range(n)],
            pass_ids={f"task-{i}" for i in range(16)},   # 80%
        )
        degraded = _mixed_report(
            [f"task-{i}" for i in range(n)],
            pass_ids={f"task-{i}" for i in range(8)},    # 40%
        )
        config = GateConfig(tolerance=0.10)
        result = check_gate(degraded, baseline, config)

        assert not result.passed, "Gate must block 80%→40% degradation"
        assert result.blockers, "Blockers must be non-empty"
        drop = 0.80 - 0.40
        assert drop > config.tolerance

    def test_noise_level_drop_not_blocked(self):
        """A 5% drop (within tolerance=10%) is treated as noise, not regression."""
        n = 20
        baseline = _mixed_report(
            [f"task-{i}" for i in range(n)],
            pass_ids={f"task-{i}" for i in range(14)},   # 70%
        )
        noisy = _mixed_report(
            [f"task-{i}" for i in range(n)],
            pass_ids={f"task-{i}" for i in range(13)},   # 65%  (drop=5%)
        )
        config = GateConfig(tolerance=0.10)
        result = check_gate(noisy, baseline, config)

        assert result.passed, "5% drop within tolerance should not block"
        assert result.warnings, "But should produce a warning"
