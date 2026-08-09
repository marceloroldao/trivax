from __future__ import annotations

from dataclasses import dataclass

from .causal_delay import CausalDelayConfidence, CausalDelayState
from .observation_router import ObservationRoute, ObservationRouter
from .temporal_credit import HistoricalCreditController, TemporalCreditState
from .value_of_information import (
    ValueOfInformationProbePolicy,
    ValueOfInformationState,
)


@dataclass(frozen=True)
class RuntimeV5State:
    raw_observation: float
    routed_observation: float
    observation_route: ObservationRoute
    accepted_delay: int | None
    delay_applied: int
    base_action: float
    action: float
    causal_delay_state: CausalDelayState
    credit_state: TemporalCreditState
    voi_state: ValueOfInformationState


class TrivaxRuntimeV5:
    """TRIVAX runtime with causal confidence, temporal credit, and VOI probes."""

    def __init__(
        self,
        controller: HistoricalCreditController | None = None,
        router: ObservationRouter | None = None,
        causal_delay: CausalDelayConfidence | None = None,
        probe_policy: ValueOfInformationProbePolicy | None = None,
        min_action: float = 0.0,
        max_action: float = 1.0,
    ) -> None:
        self.controller = controller or HistoricalCreditController()
        self.router = router or ObservationRouter()
        self.causal_delay = causal_delay or CausalDelayConfidence()
        self.probe_policy = probe_policy or ValueOfInformationProbePolicy()
        self.min_action = float(min_action)
        self.max_action = float(max_action)
        self.action = float(self.controller.action)

    def _clip(self, value: float) -> float:
        return min(self.max_action, max(self.min_action, value))

    @staticmethod
    def _causal_confidence(causal: CausalDelayState) -> float:
        raw = causal.raw_estimate
        if not raw.stable or raw.delay is None:
            return 0.0
        score = max(0.0, min(1.0, float(raw.score)))
        if causal.accepted_delay is None:
            return 0.5 * score
        if raw.delay == causal.accepted_delay:
            return score
        return 0.25 * score

    def step(self, observation: float) -> tuple[float, RuntimeV5State]:
        raw = float(observation)
        routed = self.router.process(raw)

        causal = self.causal_delay.update(
            self.action,
            raw,
            block_update=bool(routed.is_outlier),
        )
        if causal.accepted_delay is not None:
            self.controller.set_delay(int(causal.accepted_delay))

        base_action, credit = self.controller.step(routed.output)

        confidence = self._causal_confidence(causal)
        voi = self.probe_policy.step(
            confidence=confidence,
            excitation=float(causal.excitation),
            update_blocked=bool(causal.update_blocked or routed.is_outlier),
        )

        self.action = self._clip(float(base_action) + voi.offset)
        state = RuntimeV5State(
            raw_observation=raw,
            routed_observation=float(routed.output),
            observation_route=routed.route,
            accepted_delay=causal.accepted_delay,
            delay_applied=int(self.controller.delay),
            base_action=float(base_action),
            action=float(self.action),
            causal_delay_state=causal,
            credit_state=credit,
            voi_state=voi,
        )
        return self.action, state
