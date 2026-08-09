from __future__ import annotations

from dataclasses import dataclass

from .causal_delay import CausalDelayConfidence, CausalDelayState
from .observation_router import ObservationRoute, ObservationRouter
from .temporal_credit import HistoricalCreditController, TemporalCreditState


@dataclass(frozen=True)
class RuntimeV3State:
    raw_observation: float
    routed_observation: float
    observation_route: ObservationRoute
    accepted_delay: int | None
    delay_applied: int
    action: float
    causal_delay_state: CausalDelayState
    credit_state: TemporalCreditState


class TrivaxRuntimeV3:
    """Experimental TRIVAX runtime with conservative causal-delay confidence."""

    def __init__(
        self,
        controller: HistoricalCreditController | None = None,
        router: ObservationRouter | None = None,
        causal_delay: CausalDelayConfidence | None = None,
    ) -> None:
        self.controller = controller or HistoricalCreditController()
        self.router = router or ObservationRouter()
        self.causal_delay = causal_delay or CausalDelayConfidence()
        self.action = float(self.controller.action)

    def step(self, observation: float) -> tuple[float, RuntimeV3State]:
        raw = float(observation)
        routed = self.router.process(raw)

        causal = self.causal_delay.update(
            self.action,
            raw,
            block_update=bool(routed.is_outlier),
        )

        if causal.accepted_delay is not None:
            self.controller.set_delay(int(causal.accepted_delay))

        self.action, credit = self.controller.step(routed.output)
        state = RuntimeV3State(
            raw_observation=raw,
            routed_observation=float(routed.output),
            observation_route=routed.route,
            accepted_delay=causal.accepted_delay,
            delay_applied=int(self.controller.delay),
            action=float(self.action),
            causal_delay_state=causal,
            credit_state=credit,
        )
        return self.action, state
