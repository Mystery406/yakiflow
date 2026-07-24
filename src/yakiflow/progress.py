from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import monotonic
from typing import Callable


@dataclass(frozen=True, slots=True)
class StageRange:
    start: float
    span: float


class ProgressPlan:
    """Map conditional pipeline stages onto one monotonic 0..1 timeline."""

    def __init__(self, weights: dict[str, float]):
        positive = [(name, weight) for name, weight in weights.items() if weight > 0]
        total = sum(weight for _, weight in positive)
        if total <= 0:
            raise ValueError("progress plan needs at least one positive stage")
        cursor = 0.0
        self.ranges: dict[str, StageRange] = {}
        for name, weight in positive:
            span = weight / total
            self.ranges[name] = StageRange(cursor, span)
            cursor += span

    def value(self, stage: str, fraction: float) -> float:
        selected = self.ranges[stage]
        fraction = max(0.0, min(1.0, fraction))
        return min(1.0, selected.start + selected.span * fraction)


class StageTimeEstimator:
    """Estimate remaining time without mixing the speeds of unlike stages."""

    def __init__(
        self,
        plan: ProgressPlan,
        *,
        clock: Callable[[], float] = monotonic,
        minimum_observation: float = 1.0,
    ) -> None:
        self.plan = plan
        self.clock = clock
        self.minimum_observation = minimum_observation
        self._stage: str | None = None
        self._stage_started = 0.0
        self._stage_fraction = 0.0
        self._recorded: set[str] = set()
        self._completed_seconds = 0.0
        self._completed_span = 0.0

    def _baseline_rate(self) -> float | None:
        if self._completed_span <= 0:
            return None
        return self._completed_seconds / self._completed_span

    def update(
        self,
        stage: str,
        fraction: float,
        *,
        now: float | None = None,
    ) -> int | None:
        """Return estimated whole seconds remaining, or ``None`` while unknown."""
        selected = self.plan.ranges.get(stage)
        if selected is None:
            return None

        timestamp = self.clock() if now is None else now
        fraction = max(0.0, min(1.0, fraction))
        if stage != self._stage:
            self._stage = stage
            self._stage_started = timestamp
            self._stage_fraction = fraction
        else:
            fraction = max(self._stage_fraction, fraction)
            self._stage_fraction = fraction

        elapsed = max(0.0, timestamp - self._stage_started)
        observed_rate: float | None = None
        if elapsed >= self.minimum_observation and fraction > 0:
            # Seconds per normalized plan span, using the entire current stage
            # rather than a short rolling window.
            observed_rate = elapsed / (selected.span * fraction)

        if (
            fraction >= 1
            and stage not in self._recorded
            and elapsed >= self.minimum_observation
        ):
            self._recorded.add(stage)
            self._completed_seconds += elapsed
            self._completed_span += selected.span

        baseline_rate = self._baseline_rate()
        future_span = max(0.0, 1.0 - selected.start - selected.span)

        if fraction >= 1:
            if future_span <= 1e-9:
                return 0
            return None if baseline_rate is None else ceil(future_span * baseline_rate)

        if observed_rate is not None:
            current_remaining = selected.span * (1.0 - fraction) * observed_rate
            if baseline_rate is None:
                future_rate = observed_rate
            else:
                # As evidence accumulates in the active stage, prefer its rate
                # over calibration inherited from already completed stages.
                confidence = min(1.0, fraction / 0.25)
                future_rate = (
                    baseline_rate * (1.0 - confidence) + observed_rate * confidence
                )
        elif baseline_rate is not None:
            current_remaining = selected.span * (1.0 - fraction) * baseline_rate
            future_rate = baseline_rate
        else:
            return None

        return ceil(max(1.0, current_remaining + future_span * future_rate))


def make_progress_plan(
    *,
    needs_model: bool,
    is_url: bool,
    streaming: bool,
    needs_acquire: bool = True,
    needs_transcription: bool = True,
    needs_translation: bool = True,
    needs_post_alignment_translation: bool = False,
    transcription_scale: float = 1.0,
) -> ProgressPlan:
    """Build weights from expected wall-clock cost, not equal stage counts."""
    effective_stream = streaming and is_url
    transcription_scale = max(0.0, min(1.0, transcription_scale))
    weights = {
        # Model downloads are large but happen only on the first run.
        "model": 14 if needs_model else 0,
        # URL acquisition is slower than local extraction. Streaming also runs
        # preview ASR and draft agents while acquisition remains open.
        "acquire": (28 if effective_stream else (16 if is_url else 6)) if needs_acquire else 0,
        # Full-audio Whisper remains the dominant non-streaming operation.
        "transcribe": (
            (28 if effective_stream else 38) * transcription_scale
            if needs_transcription
            else 0
        ),
        "translate": (14 if effective_stream else 20) if needs_translation else 0,
        # Whisper VAD timing is always part of the pipeline.
        "align": 8,
        # WhisperX can split parent cues and require translation-only batches.
        "post-align-translate": 10 if needs_post_alignment_translation else 0,
        "publish": 2,
    }
    return ProgressPlan(weights)
