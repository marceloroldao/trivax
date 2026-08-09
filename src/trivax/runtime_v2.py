from __future__ import annotations

from dataclasses import dataclass

from .delay_estimator import DelayEstimate, DelayEstimator
from .observation_router import ObservationRoute, ObservationRouter
from .temporal_credit import HistoricalCreditController, TemporalCreditState


@dataclass(frozen=True)
class RuntimeV2State:
    raw_observation: float
    routed_observation: float
    observation_route: ObservationRoute
    estimated_delay: int | None
    delay_score: float
    delay_stable: bool
    delay_applied: int
    action: float
    credit_state: TemporalCreditState


class TrivaxRuntimeV2:
    """Experimental TRIVAX runtime with online delay estimation and historical credit.

    Unlike the first integrated runtime, v2 does not hold the actuator for
    `delay + 1` samples. Once a lag estimate is confirmed, the historical-credit
    controller updates its attribution delay while continuing to emit actions
    every cycle.
    """

    def __init__(
        self,
        controller: HistoricalCreditController | None = None,
        router: ObservationRouter | None = None,
        delay_estimator: DelayEstimator | None = None,
        delay_confirmation: int = 3,
    ) -> None:
        if delay_confirmation <= 0:
            raise ValueError("delay_confirmation must be positive")

        self.controller = controller or HistoricalCreditController()
        self.router = router or ObservationRouter()
        self.delay_estimator = delay_estimator or DelayEstimator()
        self.delay_confirmation = int(delay_confirmation)

        self.action = float(self.controller.action)
        self._candidate_delay: int | None = None
        self._candidate_count = 0
        self.estimated_delay: int | None = None

    def _accept_delay(self, estimate: DelayEstimate) -> None:
        if not estimate.stable or estimate.delay is None:
            self._candidate_delay = None
            self._candidate_count = 0
            return

        if estimate.delay == self._candidate_delay:
            self._candidate_count += 1
        else:
            self._candidate_delay = int(estimate.delay)
            self._candidate_count = 1

        if self._candidate_count >= self.delay_confirmation:
            self.estimated_delay = int(estimate.delay)
            self.controller.set_delay(self.estimated_delay)

    def step(self, observation: float) -> tuple[float, RuntimeV2State]:
        raw = float(observation)
        routed = self.router.process(raw)

        # Timing identification uses raw feedback; robust routing protects the
        # controller from impulses without distorting lag evidence.
        estimate = self.delay_estimator.update(self.action, raw)
        self._accept_delay(estimate)

        self.action, credit_state = self.controller.step(routed.output)
        state = RuntimeV2State(
            raw_observation=raw,
            routed_observation=float(routed.output),
            observation_route=routed.route,
            estimated_delay=self.estimated_delay,
            delay_score=float(estimate.score),
            delay_stable=bool(estimate.stable),
            delay_applied=int(self.controller.delay),
            action=float(self.action),
            credit_state=credit_state,
        )
        return self.action, state
