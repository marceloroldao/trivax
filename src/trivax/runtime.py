from __future__ import annotations

from dataclasses import dataclass

from .delay_estimator import DelayEstimate, DelayEstimator
from .observation_router import ObservationRoute, ObservationRouter
from .probabilistic import ProbabilisticRegimeController, ProbabilisticState


@dataclass(frozen=True)
class RuntimeState:
    raw_observation: float
    routed_observation: float
    observation_route: ObservationRoute
    estimated_delay: int | None
    delay_score: float
    delay_stable: bool
    hold_period: int
    update_applied: bool
    action: float
    controller_state: ProbabilisticState | None


class TrivaxRuntime:
    """Experimental integrated TRIVAX runtime.

    Pipeline:
        observation -> robust routing -> delay estimation -> temporal credit
        assignment -> probabilistic regime controller -> action

    The runtime begins without assuming a sensor delay. Once the lag estimator
    has stable evidence, the control update period is changed to delay + 1.
    This is intentionally conservative: an unstable estimate never changes the
    temporal policy.
    """

    def __init__(
        self,
        controller: ProbabilisticRegimeController | None = None,
        router: ObservationRouter | None = None,
        delay_estimator: DelayEstimator | None = None,
        delay_confirmation: int = 3,
    ) -> None:
        if delay_confirmation <= 0:
            raise ValueError("delay_confirmation must be positive")

        self.controller = controller or ProbabilisticRegimeController()
        self.router = router or ObservationRouter()
        self.delay_estimator = delay_estimator or DelayEstimator()
        self.delay_confirmation = int(delay_confirmation)

        self.action = float(self.controller.action)
        self.hold_period = 1
        self.counter = 0
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
            self._candidate_delay = estimate.delay
            self._candidate_count = 1

        if self._candidate_count >= self.delay_confirmation:
            self.estimated_delay = int(estimate.delay)
            self.hold_period = max(1, self.estimated_delay + 1)

    def step(self, observation: float) -> tuple[float, RuntimeState]:
        raw = float(observation)
        routed = self.router.process(raw)

        # Use raw plant feedback for lag identification. Robust routing is for
        # control protection; filtering it here could distort timing evidence.
        estimate = self.delay_estimator.update(self.action, raw)
        self._accept_delay(estimate)

        update_applied = self.counter % self.hold_period == 0
        controller_state: ProbabilisticState | None = None
        if update_applied:
            result = self.controller.step(routed.output)
            self.action = float(result[0])
            controller_state = result[1]

        state = RuntimeState(
            raw_observation=raw,
            routed_observation=float(routed.output),
            observation_route=routed.route,
            estimated_delay=self.estimated_delay,
            delay_score=float(estimate.score),
            delay_stable=bool(estimate.stable),
            hold_period=self.hold_period,
            update_applied=update_applied,
            action=self.action,
            controller_state=controller_state,
        )
        self.counter += 1
        return self.action, state
