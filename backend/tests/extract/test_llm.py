import inspect
import json
from types import SimpleNamespace

from anthropic.resources.messages.messages import Messages
from anthropic.types.json_output_format_param import JSONOutputFormatParam
from anthropic.types.output_config_param import OutputConfigParam

from biolit.domain.enums import Source, TextType
from biolit.domain.paper import Paper
from biolit.extract.llm import _SYSTEM, LlmExtractor

# sentence_spans(TEXT) == [(0, 20), (21, 39), (40, 53)]
TEXT = "Metformin was given. Acidosis followed. Insulin fell."


def _paper(*, allowed: bool = True, abstract: str | None = TEXT) -> Paper:
    return Paper(
        id="p1",
        source=Source.pubmed,
        title="t",
        abstract=abstract,
        text_type=TextType.abstract_only,
        extraction_allowed=allowed,
    )


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _thinking_block() -> SimpleNamespace:
    """A block with no `.text` attribute at all -- reading `.text` off it raises."""
    return SimpleNamespace(type="thinking", thinking="...")


class _StubClient:
    """Records the request and returns a canned response. Tests never touch the API."""

    def __init__(
        self,
        *,
        content: list[SimpleNamespace] | None = None,
        stop_reason: str = "end_turn",
        error: BaseException | None = None,
    ) -> None:
        if content is None:
            content = [_text_block('{"finding_sentences": [1]}')]
        self._content, self._stop_reason, self._error = content, stop_reason, error
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        # `error` stands in for a transport failure -- a 429, a timeout, a dropped
        # connection. The real SDK raises those from this call.
        if self._error is not None:
            raise self._error
        return SimpleNamespace(stop_reason=self._stop_reason, content=self._content)


def test_the_prompt_numbers_sentences_and_indices_become_located_findings():
    client = _StubClient()
    findings = LlmExtractor(client).findings(_paper())
    assert [f.sentence_index for f in findings] == [1]
    assert findings[0].text == "Acidosis followed."
    assert TEXT[findings[0].start : findings[0].end] == findings[0].text
    # The model must see numbered sentences -- that is the whole interface.
    sent = json.dumps(client.calls[0]["messages"])
    assert "[0] Metformin was given." in sent
    assert "[1] Acidosis followed." in sent


def test_every_request_parameter_name_matches_the_installed_sdk():
    """THE STUB IS A BLIND SPOT: `_create(**kwargs)` accepts any name, so a typo'd
    parameter would pass every other test in this file and fail only at the real run,
    after money is spent. Check the names -- top level and nested -- against the
    installed anthropic SDK's own signature and TypedDicts instead.
    """
    client = _StubClient()
    LlmExtractor(client).findings(_paper())
    call = client.calls[0]
    accepted = set(inspect.signature(Messages.create).parameters) - {"self"}
    assert set(call) <= accepted
    assert set(call["output_config"]) <= set(OutputConfigParam.__annotations__)
    assert set(call["output_config"]["format"]) <= set(JSONOutputFormatParam.__annotations__)


def test_the_request_pins_the_model_the_effort_and_a_budget_above_the_thinking_floor():
    # max_tokens caps thinking AND response text together, and claude-opus-5 runs adaptive
    # thinking by default -- a small budget is consumed by thinking and comes back as
    # stop_reason "max_tokens" with empty content. 1024 is not a safe budget here.
    client = _StubClient()
    LlmExtractor(client).findings(_paper())
    call = client.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["output_config"]["effort"] == "medium"
    assert call["max_tokens"] >= 16000
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["format"]["schema"]["required"] == ["finding_sentences"]
    # No cache_control marker: the system prompt is far below the 512-token cache minimum,
    # so a marker here would be inert and would read as caching that does not happen.
    assert call["system"] == _SYSTEM


def test_a_refusal_yields_no_findings_and_is_counted_without_reading_content():
    # stop_reason must be checked BEFORE content: a pre-output refusal's content is an
    # EMPTY list, so an implementation that parsed first would raise JSONDecodeError.
    client = _StubClient(content=[], stop_reason="refusal")
    extractor = LlmExtractor(client)
    assert extractor.findings(_paper()) == []
    assert extractor.refusals == 1
    # A refusal is NOT lumped in with the truncation bucket -- it is the failure mode the
    # spec names and is reported on its own.
    assert extractor.unusable_stops == {}


def test_a_truncated_response_is_counted_by_its_stop_reason_and_never_parsed():
    # `refusal` is not the only non-end_turn stop reason. The SDK's StopReason literal also
    # holds max_tokens, stop_sequence, tool_use, pause_turn and
    # model_context_window_exceeded; on every one of them the payload is absent or partial.
    # Handling only `refusal` lets the rest reach json.loads and abort the run mid-corpus.
    # A truncated response is a COUNTED DIAGNOSTIC, not an exception: an eval arm that
    # raises loses the whole run, while a counter keeps the arm scoreable and the failure
    # visible in the log.
    client = _StubClient(content=[], stop_reason="max_tokens")
    extractor = LlmExtractor(client)
    assert extractor.findings(_paper()) == []
    assert extractor.unusable_stops == {"max_tokens": 1}
    # ... and it is not miscounted as a refusal, which would overstate the safety story.
    assert extractor.refusals == 0


def test_the_payload_is_read_from_the_first_text_block_not_the_first_block():
    # Adaptive thinking is ON BY DEFAULT on claude-opus-5, so a thinking block precedes the
    # answer on every live call. `_thinking_block` has no `.text` attribute at all, so
    # dropping the `type == "text"` filter raises AttributeError rather than quietly
    # returning garbage.
    client = _StubClient(content=[_thinking_block(), _text_block('{"finding_sentences": [0, 2]}')])
    extractor = LlmExtractor(client)
    assert [f.sentence_index for f in extractor.findings(_paper())] == [0, 2]


def test_an_out_of_range_index_is_dropped_and_counted_once_however_often_repeated():
    # `out_of_range` counts INDICES DROPPED, not values emitted, because
    # findings_from_sentence_indices dedupes (`sorted(set(indices))`): [1, 99, 99] drops
    # exactly ONE index. Summing over the raw list would say 2. The count is derived from
    # what came back rather than by restating the `0 <= i < len(spans)` predicate that
    # already lives at base.py:30, so the two cannot drift apart.
    client = _StubClient(content=[_text_block('{"finding_sentences": [1, 99, 99]}')])
    extractor = LlmExtractor(client)
    assert [f.sentence_index for f in extractor.findings(_paper())] == [1]
    assert extractor.out_of_range == 1


def test_a_repeated_in_range_index_yields_one_finding_and_no_out_of_range():
    # The dedupe cuts both ways, and this is the fixture that distinguishes the two
    # candidate formulas: with NOTHING dropped, `len(indices) - len(out)` reports 1 while
    # `len(set(indices)) - len(out)` reports 0. Without a repeated IN-range index in the
    # suite the weaker formula survives.
    client = _StubClient(content=[_text_block('{"finding_sentences": [1, 1]}')])
    extractor = LlmExtractor(client)
    assert [f.sentence_index for f in extractor.findings(_paper())] == [1]
    assert extractor.out_of_range == 0


def test_a_transport_failure_is_counted_by_exception_type_and_costs_only_that_paper():
    # The eval makes 500 sequential calls and runs at least twice. One transient 429 must not
    # lose the whole run, so a raise from `messages.create` is a COUNTED DIAGNOSTIC and no
    # findings -- the identical policy this class already applies to an unusable stop reason,
    # and for the identical reason: an arm that raises loses the run, while a counter keeps
    # the arm scoreable and the failure visible in the log.
    # Counted BY TYPE, not as a bare total: a systematic harness bug shows up as 500 identical
    # entries where a flaky network shows a handful of mixed ones, and a scalar hides that.
    # The SDK already retries 429/5xx twice by default, so this is the residual backstop.
    client = _StubClient(error=RuntimeError("429 rate limited"))
    extractor = LlmExtractor(client)
    assert extractor.findings(_paper()) == []
    assert extractor.errors == {"RuntimeError": 1}
    # ... and it is NOT miscounted as a refusal or as a stop reason. A refusal is the failure
    # mode the spec names and an operational finding in its own right; folding a network
    # error into it would manufacture a safety story out of a dropped connection.
    assert extractor.refusals == 0
    assert extractor.unusable_stops == {}


def test_a_malformed_payload_on_a_clean_stop_is_counted_rather_than_aborting_the_run():
    # The second failure the eval must survive, and the one the stop-reason guard cannot see:
    # `stop_reason` is `end_turn`, so the truncation check passes, and the payload is still
    # not the JSON the schema promised. Structured outputs make this unlikely, not impossible,
    # and mid-corpus one JSONDecodeError costs every paper after it. The catch is NARROW --
    # around the parse only -- so a bug in `findings_from_sentence_indices` still raises
    # instead of being silently counted as a model failure.
    client = _StubClient(content=[_text_block("not json at all")])
    extractor = LlmExtractor(client)
    assert extractor.findings(_paper()) == []
    assert extractor.errors == {"JSONDecodeError": 1}
    assert extractor.refusals == 0


def test_a_licence_forbidden_paper_is_never_sent_to_the_api():
    # The strongest form of this assertion: not merely "no findings", but NO CALL AT ALL.
    client = _StubClient()
    extractor = LlmExtractor(client)
    assert extractor.findings(_paper(allowed=False)) == []
    assert client.calls == []
    assert extractor.licence_skipped == 1


def test_a_missing_or_whitespace_only_abstract_is_never_sent_to_the_api():
    # sentence_spans NEVER returns an empty list -- windowing.py:35 falls back to
    # [(0, len(text))] -- so an emptiness guard written against `spans` can never fire and
    # is not what stops this. The TEXT is, and both an absent abstract and a whitespace-only
    # one must stop before the call: sending "[0]    " costs money and can only produce a
    # fabricated index. Both sides of the `paper.abstract or ""` fallback are exercised.
    for abstract in (None, "   \n\t "):
        client = _StubClient()
        extractor = LlmExtractor(client)
        assert extractor.findings(_paper(abstract=abstract)) == []
        assert client.calls == []
        # Not a licence decision and not a model failure -- no counter moves.
        assert extractor.licence_skipped == 0
        assert extractor.refusals == 0
