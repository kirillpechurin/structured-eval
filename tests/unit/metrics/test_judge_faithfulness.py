"""JudgeFaithfulness — an LLM judge grading a whole subtree in one call.

Driven by a fake client so the tests are free and deterministic: it records the
prompts it was given and replies with whatever verdicts the case needs. What is
pinned is the metric's own behaviour, not the wording of the prompt:

- one call per attachment;
- the score it reports for the node it ran on;
- the verdicts it carries in `extra`, and the verdict→score policy;
- staying sane on a sloppy reply.

The judge is always attached **explicitly** (`root=` / `fields=`) rather than
through `config.metrics`: it applies to every node type, so cascading it would
buy one model call per node in the document.
"""

import json
from typing import Any

import pytest

from structured_eval import evaluate
from structured_eval.metrics import JudgeFaithfulness
from structured_eval.metrics.judge_faithfulness.criteria import relative_path
from structured_eval.models import (
    ArrayFieldConfig,
    EvalConfig,
    EvalReport,
    FieldConfig,
    ObjectFieldConfig,
)

pytestmark = pytest.mark.unit

SOURCE = "Invoice from Acme Corp, total amount 100.0 USD, issued in Berlin."
ACTUAL: dict[str, Any] = {"vendor": "Acme Corp", "total": 100.0}


class FakeJudge:
    """A stand-in LLM: replies with canned verdicts, remembers what it was asked.

    Shaped like `LlmClient.generate` so `resolve_client` adapts it as a
    plain callable — the injection path a user without an API key takes.
    """

    def __init__(self, verdicts: list[dict[str, str]]) -> None:
        """Freeze `verdicts` into the JSON body every call replies with."""
        self.reply = json.dumps({"verdicts": verdicts})
        self.prompts: list[str] = []

    def __call__(self, prompt: str, *, system: str | None = None) -> str:
        """The frozen reply; `prompt` is recorded so a test can read it back."""
        self.prompts.append(prompt)
        return self.reply


def judge(verdicts: list[dict[str, str]], **kwargs: Any) -> JudgeFaithfulness:
    """A `JudgeFaithfulness` wired to a fake client replying with these verdicts."""
    return JudgeFaithfulness(client=FakeJudge(verdicts), **kwargs)


def run(
    metric: JudgeFaithfulness,
    actual: dict[str, Any] | None = None,
    source: str | None = SOURCE,
    fields: dict[str, Any] | None = None,
) -> EvalReport:
    """Evaluate with the judge hung on the document root.

    Per-field configuration goes inside the root config: `config.root` and
    `config.fields` are alternatives, and the former wins.

    Returns:
        The report of an expected-free run — the judge on the root node,
        `presence` on every scalar leaf.
    """
    return evaluate(
        actual if actual is not None else ACTUAL,
        None,
        EvalConfig(
            root=ObjectFieldConfig(metrics=[metric], fields=fields or {}),
            default_scalar_metrics=["presence"],
        ),
        source=source,
    )


def verdicts(report: EvalReport, path: str = "$") -> dict[str, dict[str, str]]:
    """The judge's per-field verdicts at `path`, keyed by the field they name."""
    summary = report.field_scores[path].metrics["judge_faithfulness"].extra
    return {v["path"]: v for v in summary["verdict"]["verdicts"]}


# ── the judged node's score ──────────────────────────────────────────────────


def test_the_node_scores_the_mean_of_its_verdicts() -> None:
    report = run(
        judge(
            [
                {"path": "vendor", "verdict": "supported"},
                {"path": "total", "verdict": "contradicted"},
            ]
        )
    )
    assert report.field_scores["$"].metrics["judge_faithfulness"] == pytest.approx(0.5)


def test_the_mean_is_weighted_by_field_weight() -> None:
    # total counts triple: (1*1 + 0*3) / 4
    report = run(
        judge(
            [
                {"path": "vendor", "verdict": "supported"},
                {"path": "total", "verdict": "contradicted"},
            ]
        ),
        fields={"total": FieldConfig(weight=3.0)},
    )
    assert report.field_scores["$"].metrics["judge_faithfulness"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("verdict", "score"),
    [("supported", 1.0), ("not_stated", 0.0), ("contradicted", 0.0)],
    ids=["supported", "not_stated", "contradicted"],
)
def test_default_verdict_scores(verdict: str, score: float) -> None:
    report = run(judge([{"path": "vendor", "verdict": verdict}]))
    assert report.field_scores["$"].metrics["judge_faithfulness"] == score


def test_verdict_scores_are_configurable() -> None:
    # deepeval counts its "idk" as faithful; the policy is a parameter, so that
    # reading is one argument away rather than a fork of the metric.
    metric = judge(
        [{"path": "vendor", "verdict": "not_stated"}],
        verdict_scores={"not_stated": 1.0},
    )
    assert run(metric).field_scores["$"].metrics["judge_faithfulness"] == 1.0


def test_an_unreachable_verdict_in_the_policy_is_refused() -> None:
    # The reply schema only ever produces the three known words, so a score
    # written for a fourth would sit dead forever rather than fail loudly.
    with pytest.raises(ValueError, match="unknown verdicts"):
        JudgeFaithfulness(verdict_scores={"idk": 1.0})


# ── the verdicts it carries ──────────────────────────────────────────────────


def test_each_judged_field_gets_a_verdict_by_its_absolute_path() -> None:
    report = run(
        judge(
            [
                {"path": "vendor", "verdict": "supported"},
                {"path": "total", "verdict": "not_stated", "reason": "invented"},
            ]
        )
    )
    assert verdicts(report)["vendor"]["verdict"] == "supported"
    assert verdicts(report)["total"] == {
        "path": "total",
        "verdict": "not_stated",
        "reason": "invented",
    }


def test_a_verdict_is_not_the_score_of_the_field_it_names() -> None:
    # The judge reports for the node it was hung on; the field it ruled on keeps
    # its own metrics. To score a field by the judge, hang the judge on it.
    report = run(
        judge([{"path": "vendor", "verdict": "contradicted", "reason": "says Globex"}])
    )
    assert "judge_faithfulness" not in report.field_scores["vendor"].metrics


def test_nested_leaves_are_judged_by_the_object_above_them() -> None:
    report = run(
        judge(
            [
                {"path": "vendor.name", "verdict": "supported"},
                {"path": "vendor.city", "verdict": "not_stated"},
            ]
        ),
        actual={"vendor": {"name": "Acme Corp", "city": "Paris"}},
    )
    assert set(verdicts(report)) == {"vendor.name", "vendor.city"}
    assert report.field_scores["$"].metrics["judge_faithfulness"] == pytest.approx(0.5)


# ── one call per attachment, whatever its width ──────────────────────────────


def test_a_single_call_covers_every_field() -> None:
    client = FakeJudge(
        [
            {"path": "vendor", "verdict": "supported"},
            {"path": "total", "verdict": "supported"},
        ]
    )
    run(JudgeFaithfulness(client=client))
    assert len(client.prompts) == 1


def test_one_call_covers_a_whole_array() -> None:
    client = FakeJudge(
        [
            {"path": "[0].sku", "verdict": "supported"},
            {"path": "[1].sku", "verdict": "not_stated", "reason": "no X-3"},
        ]
    )
    report = evaluate(
        {"lines": [{"sku": "A-1"}, {"sku": "X-3"}]},
        None,
        EvalConfig(
            fields={
                "lines": ArrayFieldConfig(
                    metrics=[
                        JudgeFaithfulness({"[*].sku": "must appear"}, client=client)
                    ]
                )
            }
        ),
        source=SOURCE,
    )
    assert len(client.prompts) == 1
    assert report.field_scores["lines"].metrics["judge_faithfulness"] == pytest.approx(
        0.5
    )
    assert set(verdicts(report, "lines")) == {"lines[0].sku", "lines[1].sku"}


def test_an_array_of_scalars_is_judged_element_by_element() -> None:
    # No field to name inside an element: `[*]` *is* the value, and the element
    # addresses itself as `[0]`.
    client = FakeJudge(
        [
            {"path": "[0]", "verdict": "supported"},
            {"path": "[1]", "verdict": "not_stated", "reason": "never mentioned"},
        ]
    )
    report = evaluate(
        {"tags": ["invoice", "training"]},
        None,
        EvalConfig(
            fields={
                "tags": ArrayFieldConfig(
                    metrics=[
                        JudgeFaithfulness({"[*]": "must be mentioned"}, client=client)
                    ]
                )
            }
        ),
        source=SOURCE,
    )
    assert set(verdicts(report, "tags")) == {"tags[0]", "tags[1]"}
    assert report.field_scores["tags"].metrics["judge_faithfulness"] == pytest.approx(
        0.5
    )


def test_a_document_that_is_an_array_is_judged_at_the_root() -> None:
    # The judged node *is* the root, so relative and absolute paths coincide.
    client = FakeJudge(
        [
            {"path": "[0].sku", "verdict": "supported"},
            {"path": "[1].sku", "verdict": "not_stated", "reason": "no X-3"},
        ]
    )
    report = evaluate(
        [{"sku": "A-1"}, {"sku": "X-3"}],
        None,
        EvalConfig(
            root=ArrayFieldConfig(
                metrics=[JudgeFaithfulness({"[*].sku": "must appear"}, client=client)]
            )
        ),
        source=SOURCE,
    )
    assert set(verdicts(report)) == {"[0].sku", "[1].sku"}
    assert report.field_scores["$"].metrics["judge_faithfulness"] == pytest.approx(0.5)


def test_alternating_nesting_is_covered_by_one_call() -> None:
    # array → object → array: the judge covers its whole subtree however the
    # nesting alternates, and the criteria stack the same way.
    client = FakeJudge(
        [
            {"path": "[0].id", "verdict": "supported"},
            {"path": "[0].items[0].sku", "verdict": "supported"},
            {"path": "[1].id", "verdict": "not_stated", "reason": "invented"},
            {"path": "[1].items[0].sku", "verdict": "not_stated", "reason": "invented"},
        ]
    )
    report = evaluate(
        {
            "orders": [
                {"id": "INV-1", "items": [{"sku": "A-1"}]},
                {"id": "INV-9", "items": [{"sku": "X-3"}]},
            ]
        },
        None,
        EvalConfig(
            fields={
                "orders": ArrayFieldConfig(
                    metrics=[
                        JudgeFaithfulness(
                            {
                                "[*].id": "must appear",
                                "[*].items[*].sku": "must appear",
                            },
                            client=client,
                        )
                    ]
                )
            }
        ),
        source=SOURCE,
    )
    assert len(client.prompts) == 1
    assert set(verdicts(report, "orders")) == {
        "orders[0].id",
        "orders[0].items[0].sku",
        "orders[1].id",
        "orders[1].items[0].sku",
    }


def test_an_array_is_covered_by_the_judge_on_the_object_above() -> None:
    # An array is just part of the subtree, so a judge on the parent reaches its
    # elements — and the array node itself is not judged.
    report = run(
        judge(
            [
                {"path": "vendor", "verdict": "supported"},
                {"path": "line_items[0].sku", "verdict": "supported"},
                {"path": "line_items[1].sku", "verdict": "not_stated"},
            ]
        ),
        actual={"vendor": "Acme Corp", "line_items": [{"sku": "A-1"}, {"sku": "X-3"}]},
    )
    assert set(verdicts(report)) == {
        "vendor",
        "line_items[0].sku",
        "line_items[1].sku",
    }
    assert "judge_faithfulness" not in report.field_scores["line_items"].metrics


# ── what it costs ────────────────────────────────────────────────────────────


def test_cascading_buys_one_call_per_node() -> None:
    # The judge fits every node type, so `config.metrics` spreads it over the
    # whole document. Nothing forbids it; this pins the price so it stays known.
    client = FakeJudge([{"path": "month", "verdict": "supported"}])
    evaluate(
        {"vendor": "Acme Corp", "period": {"month": "January"}},
        None,
        EvalConfig(metrics=[JudgeFaithfulness(client=client)]),
        source=SOURCE,
    )
    # root, vendor, period, period.month
    assert len(client.prompts) == 4


def test_one_judge_per_array_element_costs_one_call_each() -> None:
    # `item=` applies the config to every element, so this is one call per row —
    # the honest way to say "judge this field of every row", and the expensive
    # one. Each call is about a single leaf, which names itself.
    client = FakeJudge([{"path": "severity", "verdict": "supported"}])
    report = evaluate(
        {"claims": [{"severity": "minor"}, {"severity": "total loss"}]},
        None,
        EvalConfig(
            fields={
                "claims": ArrayFieldConfig(
                    item=ObjectFieldConfig(
                        fields={
                            "severity": FieldConfig(
                                metrics=[JudgeFaithfulness("must match", client=client)]
                            )
                        }
                    )
                )
            }
        ),
        source=SOURCE,
    )
    assert len(client.prompts) == 2
    assert (
        report.field_scores["claims[0].severity"].metrics["judge_faithfulness"] == 1.0
    )
    assert (
        report.field_scores["claims[1].severity"].metrics["judge_faithfulness"] == 1.0
    )


def test_leaf_by_leaf_costs_one_call_per_leaf() -> None:
    # Three leaves judged separately: three calls, three prompts, and none of
    # them sees the others — versus one call for the same three fields.
    per_leaf = [FakeJudge([{"path": name, "verdict": "supported"}]) for name in ACTUAL]
    evaluate(
        ACTUAL,
        None,
        EvalConfig(
            fields={
                name: FieldConfig(metrics=[JudgeFaithfulness("stated", client=client)])
                for name, client in zip(ACTUAL, per_leaf, strict=True)
            }
        ),
        source=SOURCE,
    )
    assert sum(len(c.prompts) for c in per_leaf) == len(ACTUAL)

    whole = FakeJudge([{"path": name, "verdict": "supported"} for name in ACTUAL])
    run(JudgeFaithfulness(client=whole))
    assert len(whole.prompts) == 1


# ── the judge as the node's representative score ─────────────────────────────


def test_the_judge_can_be_the_nodes_key_metric() -> None:
    # `key_metric="judge_faithfulness"` makes the verdict the node's headline
    # score rather than one metric averaged among others.
    client = FakeJudge(
        [
            {"path": "name", "verdict": "supported"},
            {"path": "city", "verdict": "not_stated"},
        ]
    )
    report = evaluate(
        {"vendor": {"name": "Acme Corp", "city": "Paris"}},
        None,
        EvalConfig(
            fields={
                "vendor": ObjectFieldConfig(
                    metrics=[JudgeFaithfulness(client=client)],
                    key_metric="judge_faithfulness",
                )
            }
        ),
        source=SOURCE,
    )
    assert report.field_scores["vendor"].score == pytest.approx(0.5)


# ── criteria ─────────────────────────────────────────────────────────────────


def test_criteria_are_relative_to_the_judged_node() -> None:
    client = FakeJudge([{"path": "name", "verdict": "supported"}])
    evaluate(
        {"vendor": {"name": "Acme Corp"}},
        None,
        EvalConfig(
            fields={
                "vendor": ObjectFieldConfig(
                    metrics=[
                        JudgeFaithfulness({"name": "must be the issuer"}, client=client)
                    ]
                )
            }
        ),
        source=SOURCE,
    )
    assert "criterion: must be the issuer" in client.prompts[0]


def test_a_wildcard_criterion_reaches_every_element() -> None:
    client = FakeJudge([])
    run(
        JudgeFaithfulness({"tags[*]": "must be mentioned"}, client=client),
        actual={"tags": ["a", "b", "c"]},
    )
    assert client.prompts[0].count("criterion: must be mentioned") == 3


def test_a_literal_index_criterion_wins_over_the_wildcard() -> None:
    # For the rare case where one position genuinely differs from its
    # neighbours — whatever order the two were written in.
    client = FakeJudge([])
    run(
        JudgeFaithfulness(
            {"tags[*]": "any tag", "tags[0]": "the primary tag"}, client=client
        ),
        actual={"tags": ["a", "b"]},
    )
    assert (
        'path: tags[0]\n  value: "a"\n  criterion: the primary tag'
        in (client.prompts[0])
    )
    assert 'path: tags[1]\n  value: "b"\n  criterion: any tag' in client.prompts[0]


def test_fields_reach_the_prompt_in_document_order() -> None:
    # The tree sorts an object's children by name, which is right for a report
    # and wrong for a prompt: the judge is asked to read the fields together,
    # and "together" means as the document wrote them — an amount next to its
    # currency, not next to whatever shares its first letter.
    client = FakeJudge([])
    run(
        JudgeFaithfulness(client=client),
        actual={"total": 100.0, "currency": "USD", "authored_by": "Ann"},
    )
    listed = [
        line.removeprefix("- path: ")
        for line in client.prompts[0].splitlines()
        if line.startswith("- path: ")
    ]
    assert listed == ["total", "currency", "authored_by"]


def test_a_judged_leaf_is_addressed_by_its_own_name() -> None:
    # A relative path to oneself would be empty, and an empty label tells the
    # model nothing about what it is grading.
    client = FakeJudge([{"path": "city", "verdict": "supported"}])
    report = evaluate(
        {"vendor": {"city": "Berlin"}},
        None,
        EvalConfig(
            fields={
                "vendor": ObjectFieldConfig(
                    fields={
                        "city": FieldConfig(
                            metrics=[JudgeFaithfulness("must be stated", client=client)]
                        )
                    }
                )
            }
        ),
        source=SOURCE,
    )
    assert "path: city" in client.prompts[0]
    assert report.field_scores["vendor.city"].metrics["judge_faithfulness"] == 1.0


def test_fields_without_a_criterion_get_the_default() -> None:
    client = FakeJudge([{"path": "vendor", "verdict": "supported"}])
    run(JudgeFaithfulness({"total": "must match the invoice total"}, client=client))
    assert "must match the invoice total" in client.prompts[0]
    assert "follow directly from, the source" in client.prompts[0]


# ── nulls are claims, not gaps ───────────────────────────────────────────────


def test_a_null_field_is_judged_like_any_other() -> None:
    # null claims "the source states nothing here": true → supported,
    # false → contradicted. It is judged, never skipped.
    report = run(
        judge(
            [
                {"path": "vendor", "verdict": "supported"},
                {
                    "path": "total",
                    "verdict": "contradicted",
                    "reason": "source says 100",
                },
            ]
        ),
        actual={"vendor": None, "total": None},
    )
    assert set(verdicts(report)) == {"vendor", "total"}
    assert report.field_scores["$"].metrics["judge_faithfulness"] == pytest.approx(0.5)


def test_a_null_value_reaches_the_prompt_as_null() -> None:
    client = FakeJudge([{"path": "vendor", "verdict": "supported"}])
    run(JudgeFaithfulness(client=client), actual={"vendor": None})
    assert "value: null" in client.prompts[0]


def test_a_field_the_output_never_produced_is_not_judged() -> None:
    # The claims come from the output, as in the reference faithfulness metrics
    # — `zip` is a node only because `expected` has the key, and a field nobody
    # emitted asserts nothing about the source. That gap is recall (`Presence`
    # / `CoverageLeafScore`), and conflating the two would make one number mean
    # both. An explicit null is the opposite case: it *is* a claim, so it stays.
    client = FakeJudge([{"path": "vendor", "verdict": "supported"}])
    report = evaluate(
        {"vendor": "Acme Corp", "city": None},
        {"vendor": "Acme Corp", "city": "Berlin", "zip": "10115"},
        EvalConfig(root=ObjectFieldConfig(metrics=[JudgeFaithfulness(client=client)])),
        source=SOURCE,
    )
    listed = [
        line.removeprefix("- path: ")
        for line in client.prompts[0].splitlines()
        if line.startswith("- path: ")
    ]
    assert listed == ["vendor", "city"]
    assert "zip" not in client.prompts[0]
    assert report.field_scores["zip"].metrics.get("judge_faithfulness") is None


def test_a_node_whose_fields_are_all_absent_is_not_worth_a_call() -> None:
    # Nothing was emitted beneath it, so there is nothing to rule on — the same
    # opt-out an empty object takes.
    client = FakeJudge([])
    report = evaluate(
        {"vendor": {}},
        {"vendor": {"name": "Acme Corp"}},
        EvalConfig(
            fields={
                "vendor": ObjectFieldConfig(metrics=[JudgeFaithfulness(client=client)])
            }
        ),
        source=SOURCE,
    )
    assert client.prompts == []
    assert "judge_faithfulness" not in report.field_scores["vendor"].metrics


# ── where the detail lands ───────────────────────────────────────────────────


def test_the_detail_is_one_whole_verdict_per_judged_node() -> None:
    # `extra_values("verdict")` gathers one JudgeVerdict per node a judge ran
    # on, wherever the judges were hung — each keeping its own score together
    # with the fields backing it.
    report = evaluate(
        {"vendor": {"city": "Paris"}, "period": {"month": "March"}},
        None,
        EvalConfig(
            fields={
                "vendor": ObjectFieldConfig(
                    metrics=[
                        JudgeFaithfulness(
                            client=FakeJudge(
                                [{"path": "city", "verdict": "not_stated"}]
                            )
                        )
                    ]
                ),
                "period": ObjectFieldConfig(
                    metrics=[
                        JudgeFaithfulness(
                            client=FakeJudge(
                                [{"path": "month", "verdict": "contradicted"}]
                            )
                        )
                    ]
                ),
            }
        ),
        source=SOURCE,
    )
    gathered = report.metrics["judge_faithfulness"].extra_values("verdict")
    assert len(gathered) == 2  # one per judged node, not one per verdict
    assert [
        (v["path"], v["verdict"]) for summary in gathered for v in summary["verdicts"]
    ] == [
        ("period.month", "contradicted"),
        ("vendor.city", "not_stated"),
    ]
    assert [summary["score"] for summary in gathered] == [0.0, 0.0]


def test_the_detail_carries_the_judged_nodes_own_score() -> None:
    # The score is the MetricResult *and* travels inside the summary, so a
    # gathered verdict is readable on its own — it does not need the node it
    # came from to say what it decided.
    report = run(
        judge(
            [
                {"path": "vendor", "verdict": "supported"},
                {"path": "total", "verdict": "contradicted"},
            ]
        )
    )
    result = report.field_scores["$"].metrics["judge_faithfulness"]
    assert result.extra["verdict"]["score"] == pytest.approx(float(result))


# ── how a path is spelled on the wire ────────────────────────────────────────


@pytest.mark.parametrize(
    ("root", "path", "relative"),
    [
        ("$", "vendor.name", "vendor.name"),
        ("$", "[0].sku", "[0].sku"),
        ("vendor", "vendor.name", "name"),
        ("line_items", "line_items[0].sku", "[0].sku"),
        ("vendor.city", "vendor.city", "city"),
        ("tags", "tags", "tags"),
        ("vendor", "invoice_id", "invoice_id"),
    ],
    ids=[
        "root-object",
        "root-array",
        "nested-object",
        "array-element",
        "the-judged-leaf-names-itself",
        "a-judged-leaf-at-the-top",
        "not-under-this-root",
    ],
)
def test_relative_path_addresses_a_node_from_the_judged_one(
    root: str, path: str, relative: str
) -> None:
    # One convention, so criteria and verdicts speak the same language wherever
    # the judge is hung. A judged node names *itself* rather than pointing at
    # itself: a relative path to oneself would be empty, and an empty label
    # tells the model nothing about what it is grading. A path that is not under
    # `root` at all cannot be shortened, so it travels whole — the judge only
    # ever asks about its own subtree, so this is a guard, not a case.
    assert relative_path(root, path) == relative


# ── a sloppy reply must not invent numbers ───────────────────────────────────


def test_a_node_with_no_leaves_is_not_worth_a_call() -> None:
    # An empty object claims nothing about the source, so there is nothing to
    # rule on — and no reason to pay for a call that would ask about no fields.
    client = FakeJudge([])
    report = evaluate(
        {"vendor": {}},
        None,
        EvalConfig(
            fields={
                "vendor": ObjectFieldConfig(metrics=[JudgeFaithfulness(client=client)])
            }
        ),
        source=SOURCE,
    )
    assert client.prompts == []
    assert "judge_faithfulness" not in report.field_scores["vendor"].metrics


def test_a_field_the_judge_ignored_is_left_out_of_the_mean() -> None:
    # Silence is missing evidence, not a failed check: `total` is absent from
    # the verdicts rather than counted as a zero.
    report = run(judge([{"path": "vendor", "verdict": "supported"}]))
    assert set(verdicts(report)) == {"vendor"}
    assert report.field_scores["$"].metrics["judge_faithfulness"] == 1.0


def test_a_path_the_judge_invented_is_dropped() -> None:
    report = run(
        judge(
            [
                {"path": "vendor", "verdict": "supported"},
                {"path": "ghost", "verdict": "contradicted"},
            ]
        )
    )
    assert set(verdicts(report)) == {"vendor"}


def test_a_repeated_path_keeps_the_first_verdict() -> None:
    report = run(
        judge(
            [
                {"path": "vendor", "verdict": "supported"},
                {"path": "vendor", "verdict": "contradicted"},
            ]
        )
    )
    assert verdicts(report)["vendor"]["verdict"] == "supported"
    assert report.field_scores["$"].metrics["judge_faithfulness"] == 1.0


def test_no_verdicts_at_all_means_no_metric() -> None:
    report = run(judge([]))
    assert "judge_faithfulness" not in report.metrics


# ── configuration errors ─────────────────────────────────────────────────────


def test_requires_a_source() -> None:
    with pytest.raises(ValueError, match="source"):
        run(judge([{"path": "vendor", "verdict": "supported"}]), source=None)


def test_path_keyed_criteria_on_a_leaf_are_refused() -> None:
    # A leaf has nothing beneath it for the keys to address. The constructor
    # cannot know where it will be hung, so this is caught when it runs.
    with pytest.raises(ValueError, match="nothing beneath it"):
        evaluate(
            {"vendor": "Acme Corp"},
            None,
            EvalConfig(
                fields={
                    "vendor": FieldConfig(
                        metrics=[judge([], criteria={"name": "must be the issuer"})]
                    )
                }
            ),
            source=SOURCE,
        )


def test_building_the_metric_needs_no_credentials(monkeypatch: Any) -> None:
    """A config must be constructible without an API key; the client waits."""
    monkeypatch.delenv("STRUCTURED_EVAL_LLM_MODEL", raising=False)
    metric = JudgeFaithfulness({"total": "..."})

    with pytest.raises(ValueError, match="STRUCTURED_EVAL_LLM_MODEL"):
        _ = metric.client
