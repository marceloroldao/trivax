from __future__ import annotations

from dataclasses import dataclass

from .causal_delay import CausalDelayConfidence, CausalDelayState
from .observation_router import ObservationRoute, ObservationRouter
from .probing import IdentificationProbePolicy, ProbeState
from .temporal_credit import HistoricalCreditController, TemporalCreditState


@dataclass(frozen=True)
class RuntimeV4State:
    raw_observation: float
    routed_observation: float
    observation_route: ObservationRoute
    accepted_delay: int | None
    delay_applied: int
    base_action: float
    action: float
    causal_delay_state: CausalDelayState
    credit_state: TemporalCreditState
    probe_state: ProbeState


class TrivaxRuntimeV4:
    """TRIVAX runtime with conservative causal confidence and active probes."""

    def __init__(
        self,
        controller: HistoricalCreditController | None = None,
        router: ObservationRouter | None = None,
        causal_delay: CausalDelayConfidence | None = None,
        probe_policy: IdentificationProbePolicy | None = None,
        min_action: float = 0.0,
        max_action: float = 1.0,
    ) -> None:
        self.controller = controller or HistoricalCreditController()
        self.router = router or ObservationRouter()
        self.causal_delay = causal_delay or CausalDelayConfidence()
        self.probe_policy = probe_policy or IdentificationProbePolicy()
        self.min_action = float(min_action)
        self.max_action = float(max_action)
        self.action = float(self.controller.action)

    def _clip(self, value: float) -> float:
        return min(self.max_action, max(self.min_action, value))

    def step(self, observation: float) -> tuple[float, RuntimeV4State]:
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

        raw_estimate = causal.raw_estimate
        confidence_ok = (
            causal.accepted_delay is not None
            and raw_estimate.stable
            and raw_estimate.delay == causal.accepted_delay
        )
        probe = self.probe_policy.step(
            confidence_ok=bool(confidence_ok),
            update_blocked=bool(causal.update_blocked or routed.is_outlier),
        )

        self.action = self._clip(float(base_action) + probe.offset)
        state = RuntimeV4State(
            raw_observation=raw,
            routed_observation=float(routed.output),
            observation_route=routed.route,
            accepted_delay=causal.accepted_delay,
            delay_applied=int(self.controller.delay),
            base_action=float(base_action),
            action=float(self.action),
            causal_delay_state=causal,
            credit_state=credit,
            probe_state=probe,
        )
        return self.action, state
