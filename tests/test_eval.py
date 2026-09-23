"""Block 24 tests: eval framework — dataset, traces, rubrics, judge, runner."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentkit.eval.dataset import (
    EvalCase,
    EvalDataset,
    gaia_to_eval_cases,
    make_custom_dataset,
)
from agentkit.eval.judge import (
    Verdict,
    _parse_verdict,
    _parse_winner,
    _split_reasoning,
    judge_pairwise,
    judge_single,
    verdict_cache_key,
    verdicts_to_dict,
)
from agentkit.eval.rubrics import (
    ALL_RUBRICS,
    ANSWER_RELEVANCE,
    DATA_PROVENANCE,
    DEFAULT_RUBRICS,
    FORMAT_COMPLIANCE,
    TRAJECTORY_SOUNDNESS,
)
from agentkit.eval.runner import (
    EvalReport,
    EvalResult,
    EvalRunner,
    VerdictCache,
    _compute_metrics,
)
from agentkit.eval.traces import (
    TraceFeatures,
    extract_features,
    features_from_legacy,
    features_from_otel_spans,
    load_legacy_results,
    load_otel_spans,
)

# ── Dataset ────────────────────────────────────────────────────────────────────


class TestEvalCase:
    def test_roundtrip(self):
        case = EvalCase(
            id="test-001",
            input="What is 2+2?",
            expected="4",
            category="core",
            tags=["math"],
            metadata={"source": "test"},
        )
        d = case.to_dict()
        restored = EvalCase.from_dict(d)
        assert restored.id == case.id
        assert restored.input == case.input
        assert restored.expected == case.expected
        assert restored.category == case.category
        assert restored.tags == case.tags

    def test_from_dict_defaults(self):
        case = EvalCase.from_dict({"id": "x", "input": "hello"})
        assert case.expected is None
        assert case.category == "core"
        assert case.tags == []


class TestEvalDataset:
    def test_add_and_len(self):
        ds = EvalDataset()
        ds.add(EvalCase(id="1", input="a"))
        ds.add(EvalCase(id="2", input="b"))
        assert len(ds) == 2

    def test_iter(self):
        cases = [EvalCase(id=str(i), input=f"q{i}") for i in range(3)]
        ds = EvalDataset(cases)
        assert list(ds) == cases

    def test_save_load_roundtrip(self, tmp_path):
        ds = EvalDataset([
            EvalCase(id="c1", input="q1", expected="a1", category="core"),
            EvalCase(id="c2", input="q2", category="edge"),
        ])
        path = tmp_path / "dataset.jsonl"
        ds.save(path)
        loaded = EvalDataset.load(path)
        assert len(loaded) == 2
        assert loaded.cases[0].id == "c1"
        assert loaded.cases[1].category == "edge"

    def test_filter_by_category(self):
        ds = EvalDataset([
            EvalCase(id="1", input="a", category="core"),
            EvalCase(id="2", input="b", category="edge"),
            EvalCase(id="3", input="c", category="core"),
        ])
        filtered = ds.filter(category="core")
        assert len(filtered) == 2
        assert all(c.category == "core" for c in filtered)

    def test_filter_by_tag(self):
        ds = EvalDataset([
            EvalCase(id="1", input="a", tags=["math"]),
            EvalCase(id="2", input="b", tags=["search"]),
            EvalCase(id="3", input="c", tags=["math", "search"]),
        ])
        filtered = ds.filter(tags=["math"])
        assert len(filtered) == 2

    def test_split(self):
        ds = EvalDataset([EvalCase(id=str(i), input=str(i)) for i in range(5)])
        first, rest = ds.split(3)
        assert len(first) == 3
        assert len(rest) == 2

    def test_categories(self):
        ds = EvalDataset([
            EvalCase(id="1", input="a", category="core"),
            EvalCase(id="2", input="b", category="edge"),
        ])
        assert ds.categories() == ["core", "edge"]

    def test_tags_list(self):
        ds = EvalDataset([
            EvalCase(id="1", input="a", tags=["x", "y"]),
            EvalCase(id="2", input="b", tags=["y", "z"]),
        ])
        assert ds.tags() == ["x", "y", "z"]


class TestMakeCustomDataset:
    def test_has_cases(self):
        ds = make_custom_dataset()
        assert len(ds) >= 19

    def test_has_all_categories(self):
        ds = make_custom_dataset()
        cats = set(ds.categories())
        assert "core" in cats
        assert "edge" in cats
        assert "adversarial" in cats

    def test_adversarial_count(self):
        ds = make_custom_dataset().filter(category="adversarial")
        assert len(ds) >= 4

    def test_all_have_ids(self):
        for case in make_custom_dataset():
            assert case.id


class TestGaiaToEvalCases:
    def test_basic_conversion(self):
        tasks = [
            {
                "task_id": "gaia-001",
                "Question": "What year was Python created?",
                "Final answer": "1991",
                "Level": "1",
                "Annotator Metadata": {"Tools": "web_search", "Number of steps": "2", "Steps": "..."},
            }
        ]
        cases = gaia_to_eval_cases(tasks)
        assert len(cases) == 1
        c = cases[0]
        assert c.id == "gaia-001"
        assert c.input == "What year was Python created?"
        assert c.expected == "1991"
        assert "level_1" in c.tags
        assert c.metadata["source"] == "gaia"

    def test_missing_answer(self):
        tasks = [{"task_id": "x", "Question": "Q?", "Level": "2"}]
        cases = gaia_to_eval_cases(tasks)
        assert cases[0].expected is None


# ── Traces ─────────────────────────────────────────────────────────────────────


def _make_event(author: str, tool_name: str | None = None, status: str = "success"):
    content = []
    if tool_name:
        content.append({"type": "tool_call", "name": tool_name})
        content.append({"type": "tool_result", "status": status, "content": [{"text": "ok"}]})
    else:
        content.append({"type": "message", "content": "some text"})
    return {"author": author, "content": content}


class TestTraceFeatures:
    def test_total_tool_calls(self):
        tf = TraceFeatures(tool_calls={"search_web": 2, "execute_python": 1})
        assert tf.total_tool_calls == 3

    def test_used_search(self):
        tf = TraceFeatures(search_calls=1)
        assert tf.used_search is True

    def test_not_used_search(self):
        tf = TraceFeatures(search_calls=0)
        assert tf.used_search is False

    def test_summary_text(self):
        tf = TraceFeatures(steps=3, tool_calls={"search_web": 2}, search_calls=2)
        s = tf.summary_text()
        assert "Steps: 3" in s
        assert "search_web" in s


class TestExtractFeatures:
    def test_no_events(self):
        tf = extract_features([])
        assert tf.total_tool_calls == 0
        assert tf.steps == 0

    def test_counts_tool_calls(self):
        events = [
            _make_event("agent", "search_web"),
            _make_event("agent", "search_web"),
            _make_event("agent", "execute_python"),
        ]
        tf = extract_features(events)
        assert tf.tool_calls["search_web"] == 2
        assert tf.tool_calls["execute_python"] == 1
        assert tf.search_calls == 2
        assert tf.used_code is True

    def test_counts_errors(self):
        events = [_make_event("agent", "search_web", status="error")]
        tf = extract_features(events)
        assert tf.tool_errors == 1

    def test_task_id(self):
        tf = extract_features([], task_id="abc")
        assert tf.task_id == "abc"


class TestFeaturesFromLegacy:
    def test_basic(self):
        result = {
            "task_id": "t1",
            "events": [_make_event("agent", "search_web")],
            "hit_max": False,
            "input_tokens": 100,
            "output_tokens": 50,
        }
        tf = features_from_legacy(result)
        assert tf.task_id == "t1"
        assert tf.search_calls == 1
        assert tf.input_tokens == 100


class TestLoadLegacyResults:
    def test_load_list(self, tmp_path):
        data = [{"task_id": "a", "question": "q", "prediction": "p"}]
        p = tmp_path / "r.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_legacy_results(p)
        assert len(loaded) == 1
        assert loaded[0]["task_id"] == "a"

    def test_load_dict(self, tmp_path):
        data = {"a": {"task_id": "a"}, "b": {"task_id": "b"}}
        p = tmp_path / "r.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_legacy_results(p)
        assert len(loaded) == 2


class TestOtelSpans:
    def test_features_from_spans(self):
        spans = [
            {
                "trace_id": "t1",
                "name": "tool.execute",
                "attributes": {"tool.name": "search_web", "tool.status": "ok"},
            },
            {
                "trace_id": "t1",
                "name": "agent.step",
                "attributes": {},
            },
            {
                "trace_id": "t1",
                "name": "gen_ai.chat",
                "attributes": {
                    "gen_ai.usage.input_tokens": 200,
                    "gen_ai.usage.output_tokens": 100,
                },
            },
        ]
        tf = features_from_otel_spans(spans, "t1")
        assert tf.steps == 1
        assert tf.tool_calls.get("search_web") == 1
        assert tf.input_tokens == 200

    def test_ignores_other_trace_ids(self):
        spans = [
            {"trace_id": "other", "name": "tool.execute", "attributes": {"tool.name": "x"}},
        ]
        tf = features_from_otel_spans(spans, "t1")
        assert tf.total_tool_calls == 0

    def test_load_otel_spans(self, tmp_path):
        p = tmp_path / "spans.jsonl"
        p.write_text(
            json.dumps({"trace_id": "t1", "name": "agent.step", "attributes": {}}) + "\n",
            encoding="utf-8",
        )
        spans = load_otel_spans(p)
        assert len(spans) == 1


# ── Rubrics ────────────────────────────────────────────────────────────────────


class TestRubrics:
    def test_all_rubrics_present(self):
        assert "answer_relevance" in ALL_RUBRICS
        assert "source_credibility" in ALL_RUBRICS
        assert "format_compliance" in ALL_RUBRICS
        assert "trajectory_soundness" in ALL_RUBRICS
        assert "data_provenance" in ALL_RUBRICS

    def test_default_rubrics(self):
        assert len(DEFAULT_RUBRICS) == 5

    def test_trajectory_requires_trace(self):
        assert TRAJECTORY_SOUNDNESS.requires_trace is True
        assert DATA_PROVENANCE.requires_trace is True

    def test_standard_not_require_trace(self):
        assert ANSWER_RELEVANCE.requires_trace is False
        assert FORMAT_COMPLIANCE.requires_trace is False

    def test_rubric_has_examples(self):
        for rubric in DEFAULT_RUBRICS:
            assert rubric.examples, f"{rubric.name} has no examples"

    def test_rubric_fields(self):
        r = ANSWER_RELEVANCE
        assert r.name
        assert r.description
        assert r.pass_criterion
        assert r.fail_criterion


# ── Judge ──────────────────────────────────────────────────────────────────────


class TestParseVerdict:
    def test_pass(self):
        assert _parse_verdict("Some reasoning.\nVERDICT: PASS") == "PASS"

    def test_fail(self):
        assert _parse_verdict("Some reasoning.\nVERDICT: FAIL") == "FAIL"

    def test_unclear_no_verdict(self):
        assert _parse_verdict("Some reasoning without verdict.") == "UNCLEAR"

    def test_case_insensitive(self):
        assert _parse_verdict("VERDICT: pass") == "PASS"

    def test_last_line_wins(self):
        assert _parse_verdict("VERDICT: FAIL\nVERDICT: PASS") == "PASS"

    def test_extra_whitespace(self):
        assert _parse_verdict("  VERDICT:   PASS  ") == "PASS"


class TestParseWinner:
    def test_first_wins(self):
        result = _parse_winner("WINNER: A", "A", "B")
        assert result == "A"

    def test_second_wins(self):
        result = _parse_winner("WINNER: B", "A", "B")
        assert result == "B"

    def test_tie(self):
        result = _parse_winner("WINNER: TIE", "A", "B")
        assert result == "TIE"

    def test_unclear(self):
        result = _parse_winner("No winner here", "A", "B")
        assert result == "UNCLEAR"

    def test_reversed_labels(self):
        # label_first="B", label_second="A" means output_b was shown first as "B"
        # WINNER: B → output_b won → original assignment B wins
        result = _parse_winner("WINNER: B", "B", "A")
        assert result == "B"

    def test_a_was_shown_first(self):
        result = _parse_winner("WINNER: A", "A", "B")
        assert result == "A"


class TestSplitReasoning:
    def test_extracts_before_verdict(self):
        text = "First line.\nSecond line.\nVERDICT: PASS"
        r = _split_reasoning(text)
        assert r == "First line.\nSecond line."

    def test_extracts_before_winner(self):
        text = "Some reasoning.\nWINNER: A"
        r = _split_reasoning(text)
        assert r == "Some reasoning."

    def test_empty_if_no_reasoning(self):
        r = _split_reasoning("VERDICT: PASS")
        assert r == ""


class TestVerdictCacheKey:
    def test_stable(self):
        key1 = verdict_cache_key("case-1", "output text", "rubric")
        key2 = verdict_cache_key("case-1", "output text", "rubric")
        assert key1 == key2

    def test_different_inputs_different_keys(self):
        key1 = verdict_cache_key("case-1", "output A", "rubric")
        key2 = verdict_cache_key("case-1", "output B", "rubric")
        assert key1 != key2

    def test_length(self):
        key = verdict_cache_key("x", "y", "z")
        assert len(key) == 16


class TestVerdictsToDict:
    def test_basic(self):
        verdicts = [
            Verdict(rubric_name="r1", verdict="PASS", reasoning="ok", raw=""),
            Verdict(rubric_name="r2", verdict="FAIL", reasoning="bad", raw=""),
        ]
        d = verdicts_to_dict(verdicts)
        assert d["r1"]["verdict"] == "PASS"
        assert d["r2"]["reasoning"] == "bad"


def _mock_llm_with_response(text: str) -> MagicMock:
    from agentkit.llm import LlmResponse
    from agentkit.types import Message

    m = MagicMock()
    m.generate = AsyncMock(
        return_value=LlmResponse(content=[Message(role="assistant", content=text)])
    )
    return m


@pytest.mark.asyncio
async def test_judge_single_pass():
    llm = _mock_llm_with_response("Good answer.\nVERDICT: PASS")
    case = EvalCase(id="c1", input="What is 2+2?", expected="4")
    verdict = await judge_single(llm, case, "4", None, ANSWER_RELEVANCE)
    assert verdict.verdict == "PASS"
    assert verdict.rubric_name == "answer_relevance"
    assert "Good answer" in verdict.reasoning


@pytest.mark.asyncio
async def test_judge_single_fail():
    llm = _mock_llm_with_response("Wrong answer.\nVERDICT: FAIL")
    case = EvalCase(id="c1", input="What is 2+2?", expected="4")
    verdict = await judge_single(llm, case, "potato", None, ANSWER_RELEVANCE)
    assert verdict.verdict == "FAIL"


@pytest.mark.asyncio
async def test_judge_single_unclear():
    llm = _mock_llm_with_response("I cannot decide.")
    case = EvalCase(id="c1", input="Q", expected=None)
    verdict = await judge_single(llm, case, "A", None, ANSWER_RELEVANCE)
    assert verdict.verdict == "UNCLEAR"


@pytest.mark.asyncio
async def test_judge_single_includes_trace():
    llm = _mock_llm_with_response("Trace was fine.\nVERDICT: PASS")
    case = EvalCase(id="c1", input="Q")
    tf = TraceFeatures(steps=2, search_calls=1)
    verdict = await judge_single(llm, case, "A", tf, TRAJECTORY_SOUNDNESS)
    # Verify that the trace summary was included in the prompt
    call_args = llm.generate.call_args
    prompt_text = call_args[0][0].contents[0].content
    assert "search_calls" in prompt_text or "Breakdown" in prompt_text or "Steps:" in prompt_text
    assert verdict.verdict == "PASS"


@pytest.mark.asyncio
async def test_judge_pairwise_returns_verdict():
    llm = _mock_llm_with_response("Response A is better.\nWINNER: A")
    case = EvalCase(id="c1", input="Q")
    pv = await judge_pairwise(llm, case, "output A", "output B", ANSWER_RELEVANCE)
    assert pv.winner in ("A", "B", "TIE", "UNCLEAR")
    assert pv.first_shown in ("A", "B")
    assert pv.rubric_name == "answer_relevance"


@pytest.mark.asyncio
async def test_judge_pairwise_records_first_shown():
    # Run many times; first_shown should vary (randomized)
    results = set()
    for _ in range(20):
        llm = _mock_llm_with_response("WINNER: TIE")
        case = EvalCase(id="c1", input="Q")
        pv = await judge_pairwise(llm, case, "A output", "B output", ANSWER_RELEVANCE)
        results.add(pv.first_shown)
    # With 20 tries and p=0.5 both orderings should appear
    assert "A" in results
    assert "B" in results


# ── Runner ─────────────────────────────────────────────────────────────────────


class TestVerdictCache:
    def test_put_and_get(self, tmp_path):
        cache = VerdictCache(tmp_path / "cache.jsonl")
        cache.put("key1", "PASS", "Good.", "answer_relevance")
        result = cache.get("key1")
        assert result is not None
        assert result["verdict"] == "PASS"

    def test_miss(self, tmp_path):
        cache = VerdictCache(tmp_path / "cache.jsonl")
        assert cache.get("nonexistent") is None

    def test_persists_across_instances(self, tmp_path):
        p = tmp_path / "cache.jsonl"
        cache1 = VerdictCache(p)
        cache1.put("k", "FAIL", "Bad.", "rubric")
        cache2 = VerdictCache(p)
        assert cache2.get("k") is not None
        assert cache2.get("k")["verdict"] == "FAIL"


class TestComputeMetrics:
    def _make_report(self, results: list[tuple[str, str, str]]) -> EvalReport:
        report = EvalReport(
            run_id="test",
            timestamp="2024",
            total_cases=len(results),
            total_rubrics=1,
        )
        for case_id, verdict, category in results:
            report.results.append(EvalResult(
                case_id=case_id,
                category=category,
                output="out",
                rubric_name="answer_relevance",
                verdict=verdict,
                reasoning="",
            ))
        _compute_metrics(report)
        return report

    def test_overall_pass_rate(self):
        report = self._make_report([
            ("c1", "PASS", "core"),
            ("c2", "FAIL", "core"),
            ("c3", "PASS", "core"),
        ])
        assert report.pass_rate_overall == pytest.approx(2 / 3)

    def test_unclear_excluded_from_denominator(self):
        report = self._make_report([
            ("c1", "PASS", "core"),
            ("c2", "UNCLEAR", "core"),
        ])
        assert report.pass_rate_overall == 1.0
        assert report.unclear_count == 1

    def test_by_category(self):
        report = self._make_report([
            ("c1", "PASS", "core"),
            ("c2", "FAIL", "core"),
            ("c3", "PASS", "edge"),
        ])
        assert report.pass_rate_by_category["core"] == pytest.approx(0.5)
        assert report.pass_rate_by_category["edge"] == pytest.approx(1.0)

    def test_cached_count(self):
        report = EvalReport(run_id="t", timestamp="t", total_cases=2, total_rubrics=1)
        report.results.append(EvalResult(
            case_id="c1", category="core", output="",
            rubric_name="r", verdict="PASS", reasoning="", from_cache=True,
        ))
        report.results.append(EvalResult(
            case_id="c2", category="core", output="",
            rubric_name="r", verdict="FAIL", reasoning="", from_cache=False,
        ))
        _compute_metrics(report)
        assert report.cached_count == 1


@pytest.mark.asyncio
async def test_eval_runner_uses_cache(tmp_path):
    """Runner should not call judge again for a cached verdict."""
    llm = _mock_llm_with_response("Good.\nVERDICT: PASS")
    case = EvalCase(id="cached-case", input="Q", expected="A")
    ds = EvalDataset([case])
    cache_path = tmp_path / "cache.jsonl"

    # Pre-populate cache
    cache = VerdictCache(cache_path)
    key = verdict_cache_key("cached-case", "my output", "answer_relevance")
    cache.put(key, "FAIL", "Cached reasoning.", "answer_relevance")

    runner = EvalRunner(
        agent=None,
        judge_llm=llm,
        rubrics=[ANSWER_RELEVANCE],
        cache_path=cache_path,
    )
    report = await runner.run_dataset(ds, pre_outputs={"cached-case": "my output"})

    # Cache hit — judge LLM should NOT have been called
    llm.generate.assert_not_called()
    assert report.results[0].verdict == "FAIL"
    assert report.results[0].from_cache is True


@pytest.mark.asyncio
async def test_eval_runner_skips_trace_rubric_without_trace(tmp_path):
    llm = _mock_llm_with_response("VERDICT: PASS")
    case = EvalCase(id="c1", input="Q")
    ds = EvalDataset([case])

    runner = EvalRunner(
        agent=None,
        judge_llm=llm,
        rubrics=[TRAJECTORY_SOUNDNESS],
        cache_path=tmp_path / "cache.jsonl",
    )
    report = await runner.run_dataset(ds, pre_outputs={"c1": "output"})
    # No trace provided → UNCLEAR, judge not called
    assert report.results[0].verdict == "UNCLEAR"
    llm.generate.assert_not_called()


@pytest.mark.asyncio
async def test_eval_runner_full_flow(tmp_path):
    llm = _mock_llm_with_response("The answer is correct.\nVERDICT: PASS")
    case = EvalCase(id="c1", input="What is 2+2?", expected="4")
    ds = EvalDataset([case])

    runner = EvalRunner(
        agent=None,
        judge_llm=llm,
        rubrics=[ANSWER_RELEVANCE],
        cache_path=tmp_path / "cache.jsonl",
    )
    report = await runner.run_dataset(ds, pre_outputs={"c1": "4"})

    assert len(report.results) == 1
    assert report.results[0].verdict == "PASS"
    assert report.pass_rate_overall == 1.0
    assert "answer_relevance" in report.pass_rate_by_rubric


class TestEvalReport:
    def _make_report(self) -> EvalReport:
        report = EvalReport(
            run_id="test-run",
            timestamp="2024-01-01T00:00:00Z",
            total_cases=2,
            total_rubrics=1,
            pass_rate_overall=0.5,
            pass_rate_by_rubric={"answer_relevance": 0.5},
            pass_rate_by_category={"core": 0.5},
        )
        report.results.append(EvalResult(
            case_id="c1", category="core", output="good",
            rubric_name="answer_relevance", verdict="PASS", reasoning="ok",
        ))
        report.results.append(EvalResult(
            case_id="c2", category="core", output="bad",
            rubric_name="answer_relevance", verdict="FAIL", reasoning="nope",
        ))
        return report

    def test_save_json(self, tmp_path):
        report = self._make_report()
        path = tmp_path / "report.json"
        report.save_json(path)
        data = json.loads(path.read_text())
        assert data["run_id"] == "test-run"
        assert len(data["results"]) == 2

    def test_save_markdown(self, tmp_path):
        report = self._make_report()
        path = tmp_path / "report.md"
        report.save_markdown(path)
        md = path.read_text()
        assert "test-run" in md
        assert "answer_relevance" in md
        assert "PASS" in md
