from __future__ import annotations

from dataclasses import dataclass

from .causal_delay import CausalDelayConfidence, CausalDelayState
from .delay_estimator import DelayEstimate, DelayEstimator
from .observation_router import ObservationRoute, ObservationRouter
from .temporal_credit import HistoricalCreditController, TemporalCreditState
from .value_of_information import ValueOfInformationProbePolicy, ValueOfInformationState


@dataclass(frozen=True)
class CoreRCState:
    raw_observation: float
    routed_observation: float
    observation_route: ObservationRoute
    estimated_delay: int | None
    delay_score: float
    delay_stable: bool
    delay_applied: int
    validator_ran: bool
    validator_state: CausalDelayState | None
    validator_conflict: bool
    probe_state: ValueOfInformationState | None
    action: float
    credit_state: TemporalCreditState


class TrivaxCoreRC:
    """Release-candidate-oriented TRIVAX runtime.

    The hot path deliberately follows Runtime V2: robust routing, online delay
    estimation, and historical temporal credit. Causal validation is sampled
    only every ``validator_interval`` steps and never replaces the fast
    estimator directly. Optional VOI probing is disabled by default and is
    considered only when the sampled validator persistently disagrees with the
    fast delay estimate.

    This architecture is intentionally smaller and easier to audit/port than
    Runtime V5/V6 while preserving the mechanisms that dominated the ablation.
    """

    def __init__(
        self,
        controller: HistoricalCreditController | None = None,
        router: ObservationRouter | None = None,
        delay_estimator: DelayEstimator | None = None,
        validator: CausalDelayConfidence | None = None,
        probe_policy: ValueOfInformationProbePolicy | None = None,
        *,
        delay_confirmation: int = 3,
        validator_interval: int = 16,
        conflict_confirmation: int = 3,
        enable_probes: bool = False,
        min_action: float = 0.0,
        max_action: float = 1.0,
    ) -> None:
        if delay_confirmation <= 0:
            raise ValueError("delay_confirmation must be positive")
        if validator_interval <= 0:
            raise ValueError("validator_interval must be positive")
        if conflict_confirmation <= 0:
            raise ValueError("conflict_confirmation must be positive")

        self.controller = controller or HistoricalCreditController()
        self.router = router or ObservationRouter()
        self.delay_estimator = delay_estimator or DelayEstimator()
        self.validator = validator or CausalDelayConfidence()
        self.probe_policy = probe_policy or ValueOfInformationProbePolicy()

        self.delay_confirmation = int(delay_confirmation)
        self.validator_interval = int(validator_interval)
        self.conflict_confirmation = int(conflict_confirmation)
        self.enable_probes = bool(enable_probes)
        self.min_action = float(min_action)
        self.max_action = float(max_action)

        self.action = float(self.controller.action)
        self._step_index = 0
        self._candidate_delay: int | None = None
        self._candidate_count = 0
        self.estimated_delay: int | None = None
        self._conflict_count = 0

    def _clip(self, value: float) -> float:
        return min(self.max_action, max(self.min_action, value))

    def _accept_fast_delay(self, estimate: DelayEstimate) -> None:
        if not estimate.stable or estimate.delay is None:
            self._candidate_delay = None
            self._candidate_count = 0
            return

        delay = int(estimate.delay)
        if delay == self._candidate_delay:
            self._candidate_count += 1
        else:
            self._candidate_delay = delay
            self._candidate_count = 1

        if self._candidate_count >= self.delay_confirmation:
            self.estimated_delay = delay
            self.controller.set_delay(delay)

    @staticmethod
    def _validator_confidence(state: CausalDelayState | None) -> float:
        if state is None:
            return 1.0
        raw = state.raw_estimate
        if not raw.stable or raw.delay is None:
            return 0.0
        return max(0.0, min(1.0, float(raw.score)))

    def step(self, observation: float) -> tuple[float, CoreRCState]:
        raw = float(observation)
        routed = self.router.process(raw)

        fast = self.delay_estimator.update(self.action, raw)
        self._accept_fast_delay(fast)

        base_action, credit = self.controller.step(routed.output)

        validator_ran = (self._step_index % self.validator_interval) == 0
        validator_state: CausalDelayState | None = None
        validator_conflict = False
        probe_state: ValueOfInformationState | None = None

        if validator_ran:
            validator_state = self.validator.update(
                self.action,
                raw,
                block_update=bool(routed.is_outlier),
            )
            accepted = validator_state.accepted_delay
            validator_conflict = (
                accepted is not None
                and self.estimated_delay is not None
                and int(accepted) != int(self.estimated_delay)
            )
            if validator_conflict:
                self._conflict_count += 1
            else:
                self._conflict_count = max(0, self._conflict_count - 1)

            if self.enable_probes and self._conflict_count >= self.conflict_confirmation:
                probe_state = self.probe_policy.step(
                    confidence=self._validator_confidence(validator_state),
                    excitation=float(validator_state.excitation),
                    update_blocked=bool(validator_state.update_blocked or routed.is_outlier),
                )

        offset = 0.0 if probe_state is None else float(probe_state.offset)
        self.action = self._clip(float(base_action) + offset)
        self._step_index += 1

        state = CoreRCState(
            raw_observation=raw,
            routed_observation=float(routed.output),
            observation_route=routed.route,
            estimated_delay=self.estimated_delay,
            delay_score=float(fast.score),
            delay_stable=bool(fast.stable),
            delay_applied=int(self.controller.delay),
            validator_ran=validator_ran,
            validator_state=validator_state,
            validator_conflict=validator_conflict,
            probe_state=probe_state,
            action=float(self.action),
            credit_state=credit,
        )
        return self.action, state
