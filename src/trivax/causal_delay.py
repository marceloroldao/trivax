from __future__ import annotations

from dataclasses import dataclass

from .delay_estimator import DelayEstimate, DelayEstimator


@dataclass(frozen=True)
class CausalDelayState:
    raw_estimate: DelayEstimate
    accepted_delay: int | None
    candidate_delay: int | None
    candidate_count: int
    excitation: float
    update_blocked: bool
    changed: bool


class CausalDelayConfidence:
    """Conservative confidence layer around a lag-correlation estimator.

    Closed-loop correlation alone can confuse plant dynamics with causal delay.
    This layer therefore requires repeated agreement, sufficient action
    excitation, and hysteresis before changing an already accepted lag.

    Callers may block updates during known outliers or abrupt transition events.
    """

    def __init__(
        self,
        estimator: DelayEstimator | None = None,
        confirmation: int = 4,
        change_confirmation: int = 8,
        min_excitation: float = 1e-4,
    ) -> None:
        if confirmation <= 0:
            raise ValueError("confirmation must be positive")
        if change_confirmation < confirmation:
            raise ValueError("change_confirmation must be >= confirmation")
        if min_excitation < 0.0:
            raise ValueError("min_excitation must be non-negative")

        self.estimator = estimator or DelayEstimator()
        self.confirmation = int(confirmation)
        self.change_confirmation = int(change_confirmation)
        self.min_excitation = float(min_excitation)

        self.accepted_delay: int | None = None
        self.candidate_delay: int | None = None
        self.candidate_count = 0
        self.previous_action: float | None = None
        self.excitation = 0.0

    def update(
        self,
        action: float,
        observation: float,
        *,
        block_update: bool = False,
    ) -> CausalDelayState:
        action = float(action)
        if self.previous_action is None:
            delta_action = 0.0
        else:
            delta_action = abs(action - self.previous_action)
        self.previous_action = action

        self.excitation = 0.9 * self.excitation + 0.1 * delta_action
        raw = self.estimator.update(action, float(observation))

        changed = False
        update_blocked = block_update or self.excitation < self.min_excitation
        if update_blocked or not raw.stable or raw.delay is None:
            self.candidate_delay = None
            self.candidate_count = 0
        else:
            delay = int(raw.delay)
            if delay == self.candidate_delay:
                self.candidate_count += 1
            else:
                self.candidate_delay = delay
                self.candidate_count = 1

            required = (
                self.confirmation
                if self.accepted_delay is None or delay == self.accepted_delay
                else self.change_confirmation
            )
            if self.candidate_count >= required and delay != self.accepted_delay:
                self.accepted_delay = delay
                changed = True

        return CausalDelayState(
            raw_estimate=raw,
            accepted_delay=self.accepted_delay,
            candidate_delay=self.candidate_delay,
            candidate_count=self.candidate_count,
            excitation=self.excitation,
            update_blocked=update_blocked,
            changed=changed,
        )
