from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .causal_delay import CausalDelayConfidence, CausalDelayState
from .delay_estimator import DelayEstimate, DelayEstimator
from .observation_router import ObservationRoute, ObservationRouter
from .temporal_credit import HistoricalCreditController, TemporalCreditState
from .value_of_information import ValueOfInformationProbePolicy, ValueOfInformationState


class RuntimeMode(str, Enum):
    FAST = "fast"
    VERIFY = "verify"


@dataclass(frozen=True)
class RuntimeV6State:
    raw_observation: float
    routed_observation: float
    observation_route: ObservationRoute
    mode: RuntimeMode
    estimated_delay: int | None
    delay_applied: int
    fast_estimate: DelayEstimate
    causal_state: CausalDelayState
    credit_state: TemporalCreditState
    voi_state: ValueOfInformationState
    action: float
    conflict_steps: int
    stable_steps: int


class TrivaxRuntimeV6:
    """Selective hybrid runtime.

    V6 keeps the low-latency V2-style delay path during consistent operation.
    It enters VERIFY mode only after persistent disagreement or instability. In
    VERIFY mode, delay changes require conservative causal confirmation and VOI
    probes are allowed. After sustained agreement, V6 returns to FAST mode.
    """

    def __init__(
        self,
        controller: HistoricalCreditController | None = None,
        router: ObservationRouter | None = None,
        delay_estimator: DelayEstimator | None = None,
        causal_delay: CausalDelayConfidence | None = None,
        probe_policy: ValueOfInformationProbePolicy | None = None,
        fast_confirmation: int = 3,
        conflict_trigger: int = 6,
        recover_confirmation: int = 12,
        min_action: float = 0.0,
        max_action: float = 1.0,
    ) -> None:
        if fast_confirmation <= 0 or conflict_trigger <= 0 or recover_confirmation <= 0:
            raise ValueError("confirmation counts must be positive")
        self.controller = controller or HistoricalCreditController()
        self.router = router or ObservationRouter()
        self.delay_estimator = delay_estimator or DelayEstimator()
        self.causal_delay = causal_delay or CausalDelayConfidence()
        self.probe_policy = probe_policy or ValueOfInformationProbePolicy()
        self.fast_confirmation = int(fast_confirmation)
        self.conflict_trigger = int(conflict_trigger)
        self.recover_confirmation = int(recover_confirmation)
        self.min_action = float(min_action)
        self.max_action = float(max_action)

        self.mode = RuntimeMode.FAST
        self.action = float(self.controller.action)
        self.estimated_delay: int | None = None
        self._candidate_delay: int | None = None
        self._candidate_count = 0
        self._conflict_steps = 0
        self._stable_steps = 0

    def _clip(self, value: float) -> float:
        return min(self.max_action, max(self.min_action, value))

    def _fast_accept(self, estimate: DelayEstimate) -> None:
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
        if self._candidate_count >= self.fast_confirmation:
            self.estimated_delay = delay
            self.controller.set_delay(delay)

    @staticmethod
    def _confidence(causal: CausalDelayState) -> float:
        raw = causal.raw_estimate
        if not raw.stable or raw.delay is None:
            return 0.0
        score = max(0.0, min(1.0, float(raw.score)))
        if causal.accepted_delay is None:
            return 0.5 * score
        return score if raw.delay == causal.accepted_delay else 0.25 * score

    def step(self, observation: float) -> tuple[float, RuntimeV6State]:
        raw = float(observation)
        routed = self.router.process(raw)

        fast = self.delay_estimator.update(self.action, raw)
        causal = self.causal_delay.update(
            self.action,
            raw,
            block_update=bool(routed.is_outlier),
        )

        fast_agrees = (
            fast.stable
            and fast.delay is not None
            and self.estimated_delay is not None
            and int(fast.delay) == int(self.estimated_delay)
        )
        fast_conflicts = (
            self.estimated_delay is not None
            and (not fast.stable or fast.delay is None or int(fast.delay) != int(self.estimated_delay))
        )

        if self.mode is RuntimeMode.FAST:
            self._fast_accept(fast)
            if fast_conflicts and not routed.is_outlier:
                self._conflict_steps += 1
            else:
                self._conflict_steps = 0
            if self._conflict_steps >= self.conflict_trigger:
                self.mode = RuntimeMode.VERIFY
                self._stable_steps = 0
        else:
            if causal.accepted_delay is not None:
                self.estimated_delay = int(causal.accepted_delay)
                self.controller.set_delay(self.estimated_delay)

            causal_agrees = (
                causal.accepted_delay is not None
                and fast.stable
                and fast.delay is not None
                and int(fast.delay) == int(causal.accepted_delay)
            )
            if causal_agrees:
                self._stable_steps += 1
            else:
                self._stable_steps = 0
            if self._stable_steps >= self.recover_confirmation:
                self.mode = RuntimeMode.FAST
                self._conflict_steps = 0
                self._candidate_delay = self.estimated_delay
                self._candidate_count = self.fast_confirmation

        base_action, credit = self.controller.step(routed.output)

        if self.mode is RuntimeMode.VERIFY:
            voi = self.probe_policy.step(
                confidence=self._confidence(causal),
                excitation=float(causal.excitation),
                update_blocked=bool(causal.update_blocked or routed.is_outlier),
            )
        else:
            voi = self.probe_policy.step(
                confidence=1.0,
                excitation=max(float(causal.excitation), self.probe_policy.excitation_target),
                update_blocked=True,
            )

        self.action = self._clip(float(base_action) + voi.offset)
        state = RuntimeV6State(
            raw_observation=raw,
            routed_observation=float(routed.output),
            observation_route=routed.route,
            mode=self.mode,
            estimated_delay=self.estimated_delay,
            delay_applied=int(self.controller.delay),
            fast_estimate=fast,
            causal_state=causal,
            credit_state=credit,
            voi_state=voi,
            action=float(self.action),
            conflict_steps=self._conflict_steps,
            stable_steps=self._stable_steps,
        )
        return self.action, state
