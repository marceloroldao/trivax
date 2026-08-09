from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResolutiveState:
    """Compact inspectable state used by the initial TRIVAX controller."""

    observation: float
    delta: float
    coherence: float
    direction: int
    step_size: float


class TrivaxController:
    """Minimal adaptive scalar controller for the v0.1 reference problem.

    This implementation is deliberately small and deterministic. It is not a
    validated optimal-control algorithm; it is the first executable scaffold
    for testing TRIVAX closed-loop concepts against conventional baselines.
    """

    def __init__(
        self,
        initial_action: float = 0.5,
        step_size: float = 0.05,
        min_action: float = 0.0,
        max_action: float = 1.0,
        coherence_alpha: float = 0.2,
    ) -> None:
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if min_action >= max_action:
            raise ValueError("min_action must be smaller than max_action")
        if not min_action <= initial_action <= max_action:
            raise ValueError("initial_action must be within action bounds")
        if not 0.0 < coherence_alpha <= 1.0:
            raise ValueError("coherence_alpha must be in (0, 1]")

        self.action = float(initial_action)
        self.step_size = float(step_size)
        self.min_action = float(min_action)
        self.max_action = float(max_action)
        self.coherence_alpha = float(coherence_alpha)

        self.direction = 1
        self.previous_observation: float | None = None
        self.coherence = 0.5

    def _clip(self, value: float) -> float:
        return min(self.max_action, max(self.min_action, value))

    def observe(self, observation: float) -> ResolutiveState:
        observation = float(observation)

        if self.previous_observation is None:
            delta = 0.0
        else:
            delta = observation - self.previous_observation

            # Improvement supports the current direction; degradation reverses it.
            if delta < 0.0:
                self.direction *= -1

            # Coherence estimates how consistently recent feedback supports the
            # currently selected search direction. This is intentionally simple
            # in v0.1 so its behavior is inspectable and easy to ablate.
            agreement = 1.0 if delta >= 0.0 else 0.0
            a = self.coherence_alpha
            self.coherence = (1.0 - a) * self.coherence + a * agreement

        state = ResolutiveState(
            observation=observation,
            delta=delta,
            coherence=self.coherence,
            direction=self.direction,
            step_size=self.step_size,
        )
        self.previous_observation = observation
        return state

    def decide(self, state: ResolutiveState | None = None) -> float:
        """Return the next bounded control action.

        Low coherence slightly increases exploration while high coherence keeps
        the nominal local step. Bounds prevent invalid actuator commands.
        """

        coherence = self.coherence if state is None else state.coherence
        exploration_gain = 1.0 + 0.5 * (1.0 - coherence)
        candidate = self.action + self.direction * self.step_size * exploration_gain
        self.action = self._clip(candidate)
        return self.action

    def step(self, observation: float) -> tuple[float, ResolutiveState]:
        state = self.observe(observation)
        action = self.decide(state)
        return action, state
