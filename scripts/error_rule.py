"""The single definition of "this step's command failed", shared by every error analysis.

Three rules were in use before this module existed, and two of them were live at once: a broad
regex (70.4% of steps) behind the event-response contrast and the text baseline, and a strict
regex (51.9%) behind the anticipation and recovery analyses. Two analyses in one paper labelling
the same events differently is not defensible, so they are unified here.

WHY THE RETURN CODE. The scaffold reports the command's exit status in the observation as
``<returncode>N``, present on 96.6% of the 12,464 steps. That is the environment's own signal
that a command failed, not a guess from its output text, and it cannot be triggered by the
CONTENT of a file the agent happens to read. Measured against it:

    broad regex   70.4% of steps, agrees with the return code on 99.1%
    strict regex  51.9% of steps, agrees on 79.0%

The broad regex was already a 99%-accurate proxy for the return code, so adopting the return code
changes the headline result very little. The strict regex is the one that was wrong: it misses
19.3% of steps where a command exited non-zero without raising a Python exception (a failing test
run, a build error, a bad path), and it adds 1.7% where an "Error:" string appears in the output
of a command that SUCCEEDED -- file contents and test logs matching a pattern meant for failures.

STEPS WITH NO RETURN CODE (3.4%) are not errors. 91% of them are the final step of their run with
an empty observation: the trajectory ended, so no command ran and none could have failed. They
also carry no forward difference, so the event-response analysis drops them regardless.

WHAT THIS DOES NOT CAPTURE. A command that exits zero while doing the wrong thing is not marked.
These are operational labels for "the command failed", not a judgement about whether the agent is
making progress, and the paper should say so wherever it uses the word "error".

Reproduce: python scripts/error_rule.py
"""

from __future__ import annotations

import re

# <returncode>0</returncode> on success; any non-zero digit sequence is a failure.
RETURNCODE_RE = re.compile(r"<returncode>(\d+)")


def is_error(observation: str | None) -> bool:
    """True when the step's command reported a non-zero exit status."""
    m = RETURNCODE_RE.search(observation or "")
    return m is not None and m.group(1) != "0"


def has_returncode(observation: str | None) -> bool:
    """True when the observation carries an exit status at all, error or not."""
    return RETURNCODE_RE.search(observation or "") is not None
