"""
Matched validation-subject rule for LOSOCV (opt-in, NEUMA_VAL_RULE=matched).

Under the legacy rule the proposed model (evaluation/losocv.py) and the
baselines (baselines/run_baselines.py) each draw the validation subject from
their own random stream, so the two protocols hold out a different validation
subject in the same outer fold.  ``pick_validation_subject`` is deterministic
from (seed, test_subject) alone -- independent of the order in which the
candidates are listed and of any RNG state advanced by earlier folds -- so
every runner that calls it with the same seed selects the same validation
subject for the same test subject.

Subjects that carry a single class under the current label (``eligible`` is
given) are never used for validation: early stopping, temperature scaling and
the decision threshold are all fitted on the validation subject and are
undefined on a single-class set.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import numpy as np


def _subject_hash(subject: str) -> int:
    """Stable 32-bit integer for a subject id (not Python's salted hash())."""
    return int(hashlib.sha256(str(subject).encode("utf-8")).hexdigest()[:8], 16)


def pick_validation_subject(
    test_subject: str,
    candidates: Iterable[str],
    eligible: Optional[set] = None,
    seed: int = 42,
) -> str:
    """Return the validation subject for the fold that tests ``test_subject``.

    Parameters
    ----------
    test_subject : held-out subject of this fold (never returned).
    candidates   : the subjects available for training in this fold; the
                   choice does not depend on their order.
    eligible     : subjects that carry both classes.  When given, the draw is
                   restricted to ``candidates & eligible``; when that
                   intersection is empty the full candidate list is used.
    seed         : run seed (RANDOM_SEED / --seed); the same seed must be used
                   by every runner that should share the validation subject.
    """
    pool = sorted({str(c) for c in candidates if str(c) != str(test_subject)})
    if not pool:
        raise ValueError(f"no validation candidate for test subject {test_subject!r}")
    if eligible:
        elig = [c for c in pool if c in {str(e) for e in eligible}]
        if elig:
            pool = elig
    rng = np.random.default_rng([int(seed), _subject_hash(test_subject)])
    return pool[int(rng.integers(len(pool)))]
