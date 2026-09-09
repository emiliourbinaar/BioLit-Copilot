import io
import json

from biolit.pipeline.__main__ import emit_run
from biolit.state.pipeline import PipelineState


class _Cp1252Stream(io.StringIO):
    """A Windows console. `≥` is not in cp1252, and writing it raises exactly as the real one
    does -- which is how five of eleven completed pipeline runs were lost."""

    encoding = "cp1252"

    def write(self, text: str) -> int:
        text.encode(self.encoding)
        return super().write(text)


def test_a_run_is_persisted_even_when_the_console_cannot_display_it(tmp_path):
    """⚠️ ORDER IS THE FIX, and it is the whole fix. `--json-out` used to be written AFTER the
    answer was printed, so a console that could not encode one character in the answer threw
    away a completed run: retrieval, NER, the licence gate, extraction and clustering all done,
    exit code 1, no artifact, and a ledger on screen that looked like success.

    Measured, not hypothetical: on 2026-09-07 this discarded 5 of 11 screen runs, and the
    failures looked like NCBI rate limiting until one was rerun with stderr visible.

    Two assertions because the bug has two halves. The artifact must exist -- that is the
    ordering. And the call must not raise -- a CLI that saves the run and then dies on a
    traceback has still failed, and an unencodable character in the ANSWER is not a reason to
    fail a run that is already complete and already on disk.
    """
    out = _Cp1252Stream()
    state = PipelineState(question="statins and rhabdomyolysis")
    state.answer = "Creatine kinase ≥ 10x the upper limit of normal in 3 of 4 cohorts."
    path = tmp_path / "run.json"

    emit_run(state, query=state.question, clusters=[], json_out=str(path), out=out)

    assert json.loads(path.read_text(encoding="utf-8"))["answer"] == state.answer
    assert "10x the upper limit" in out.getvalue(), "the answer is still shown, lossily"
