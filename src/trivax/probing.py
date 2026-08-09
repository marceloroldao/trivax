from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeState:
    requested: bool
    applied: bool
    offset: float
    low_confidence_steps: int
    steps_since_probe: int
    probe_count: int


class IdentificationProbePolicy:
    """Bounded active-identification policy for low causal confidence.

    The policy requests a small alternating action offset only after causal
    confidence has remained weak for several steps. Probes are rate-limited and
    bounded so identification effort remains explicit and measurable.
    """

    def __init__(
        self,
        amplitude: float = 0.005,
        trigger_steps: int = 24,
        min_interval: int = 32,
        max_probes: int = 32,
    ) -> None:
        if amplitude <= 0.0:
            raise ValueError("amplitude must be positive")
        if trigger_steps <= 0:
            raise ValueError("trigger_steps must be positive")
        if min_interval <= 0:
            raise ValueError("min_interval must be positive")
        if max_probes < 0:
            raise ValueError("max_probes must be non-negative")

        self.amplitude = float(amplitude)
        self.trigger_steps = int(trigger_steps)
        self.min_interval = int(min_interval)
        self.max_probes = int(max_probes)

        self.low_confidence_steps = 0
        self.steps_since_probe = self.min_interval
        self.probe_count = 0
        self._sign = 1

    def step(
        self,
        *,
        confidence_ok: bool,
        update_blocked: bool,
    ) -> ProbeState:
        if confidence_ok:
            self.low_confidence_steps = 0
        elif not update_blocked:
            self.low_confidence_steps += 1

        requested = (
            not confidence_ok
            and not update_blocked
            and self.low_confidence_steps >= self.trigger_steps
            and self.steps_since_probe >= self.min_interval
            and self.probe_count < self.max_probes
        )

        offset = 0.0
        applied = False
        if requested:
            offset = self._sign * self.amplitude
            self._sign *= -1
            self.probe_count += 1
            self.steps_since_probe = 0
            self.low_confidence_steps = 0
            applied = True
        else:
            self.steps_since_probe += 1

        return ProbeState(
            requested=requested,
            applied=applied,
            offset=offset,
            low_confidence_steps=self.low_confidence_steps,
            steps_since_probe=self.steps_since_probe,
            probe_count=self.probe_count,
        )
