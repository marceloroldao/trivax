from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValueOfInformationState:
    requested: bool
    applied: bool
    offset: float
    confidence: float
    uncertainty: float
    excitation: float
    expected_information_gain: float
    perturbation_cost: float
    net_value: float
    steps_since_probe: int
    probe_count: int


class ValueOfInformationProbePolicy:
    """Bounded probe policy driven by an explicit information-value balance.

    A probe is considered only when causal confidence is weak, natural action
    excitation is insufficient, the probe interval has elapsed, and the
    estimated information value exceeds its perturbation cost.

    The model is deliberately lightweight and inspectable. It is not a learned
    information-theoretic model; it is an experimental heuristic that exposes
    every term used in the decision.
    """

    def __init__(
        self,
        amplitude: float = 0.005,
        excitation_target: float = 0.01,
        information_weight: float = 1.0,
        cost_weight: float = 8.0,
        min_net_value: float = 0.02,
        min_interval: int = 32,
        max_probes: int = 32,
    ) -> None:
        if amplitude <= 0.0:
            raise ValueError("amplitude must be positive")
        if excitation_target <= 0.0:
            raise ValueError("excitation_target must be positive")
        if information_weight < 0.0 or cost_weight < 0.0:
            raise ValueError("weights must be non-negative")
        if min_net_value < 0.0:
            raise ValueError("min_net_value must be non-negative")
        if min_interval <= 0:
            raise ValueError("min_interval must be positive")
        if max_probes < 0:
            raise ValueError("max_probes must be non-negative")

        self.amplitude = float(amplitude)
        self.excitation_target = float(excitation_target)
        self.information_weight = float(information_weight)
        self.cost_weight = float(cost_weight)
        self.min_net_value = float(min_net_value)
        self.min_interval = int(min_interval)
        self.max_probes = int(max_probes)

        self.steps_since_probe = self.min_interval
        self.probe_count = 0
        self._sign = 1

    def step(
        self,
        *,
        confidence: float,
        excitation: float,
        update_blocked: bool,
    ) -> ValueOfInformationState:
        confidence = max(0.0, min(1.0, float(confidence)))
        excitation = max(0.0, float(excitation))
        uncertainty = 1.0 - confidence

        excitation_deficit = max(
            0.0,
            1.0 - min(1.0, excitation / self.excitation_target),
        )
        expected_information_gain = (
            self.information_weight * uncertainty * excitation_deficit
        )
        perturbation_cost = self.cost_weight * (self.amplitude ** 2)
        net_value = expected_information_gain - perturbation_cost

        requested = (
            not update_blocked
            and self.steps_since_probe >= self.min_interval
            and self.probe_count < self.max_probes
            and net_value >= self.min_net_value
        )

        offset = 0.0
        applied = False
        if requested:
            offset = self._sign * self.amplitude
            self._sign *= -1
            self.probe_count += 1
            self.steps_since_probe = 0
            applied = True
        else:
            self.steps_since_probe += 1

        return ValueOfInformationState(
            requested=requested,
            applied=applied,
            offset=offset,
            confidence=confidence,
            uncertainty=uncertainty,
            excitation=excitation,
            expected_information_gain=expected_information_gain,
            perturbation_cost=perturbation_cost,
            net_value=net_value,
            steps_since_probe=self.steps_since_probe,
            probe_count=self.probe_count,
        )
