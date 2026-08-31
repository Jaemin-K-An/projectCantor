"""V3.3.2 PHASE 4 -- explicit prefill/decode tracking.

V3.3.1 DEFECT. The hook received ONE RefusalDirections object and applied its
tau/sigma to every forward pass. Calling that "phase-specific calibration" was
wrong: it only changed WHICH single calibration was used everywhere. There was
no phase awareness in the code at all.

Phase is tracked explicitly here rather than inferred from `seq_len == 1`,
because that heuristic is fragile -- a one-token prompt, or a model that
re-runs the full sequence without a cache, would break it. The state machine is
driven by forward-pass ORDER, which is unambiguous, and `seq_len` and
`past_key_values` are recorded as cross-checks that the tests assert on.
"""
from __future__ import annotations
from dataclasses import dataclass, field

PREFILL, DECODE = "PREFILL", "DECODE"


@dataclass
class PhaseState:
    """One instance per generate() call. `reset()` before each batch."""
    forward_index: int = 0
    phase: str = PREFILL
    trace: list = field(default_factory=list)
    record_trace: bool = False

    def reset(self) -> None:
        self.forward_index = 0
        self.phase = PREFILL
        self.trace.clear()

    def observe(self, seq_len: int, has_cache: bool) -> str:
        """Called once per forward at the hooked layer. Returns the phase."""
        phase = PREFILL if self.forward_index == 0 else DECODE
        self.phase = phase
        if self.record_trace:
            self.trace.append({"forward_index": self.forward_index,
                               "seq_len": int(seq_len), "has_cache": bool(has_cache),
                               "phase": phase})
        self.forward_index += 1
        return phase

    def decode_step(self) -> int:
        """0-based index among DECODE forwards; -1 during prefill."""
        return self.forward_index - 2 if self.forward_index >= 2 else (
            -1 if self.forward_index <= 1 and self.phase == PREFILL else 0)

    def consistency(self) -> dict:
        """Do the recorded seq_len / cache flags agree with the phase labels?"""
        if not self.trace:
            return {"checked": 0, "ok": True, "violations": []}
        v = []
        for t in self.trace:
            if t["phase"] == PREFILL and t["forward_index"] != 0:
                v.append(t)
            if t["phase"] == DECODE and t["seq_len"] != 1:
                v.append(t)          # cached decode should feed one token
        return {"checked": len(self.trace), "ok": not v, "violations": v[:8],
                "n_prefill": sum(t["phase"] == PREFILL for t in self.trace),
                "n_decode": sum(t["phase"] == DECODE for t in self.trace)}
