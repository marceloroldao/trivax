from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PIDState:
    error: float
    integral: float
    derivative: float
    action: float


class PIDController:
    """Lightweight bounded PID baseline with anti-windup.

    Important: PID is a tracking-controller reference. It requires an explicit
    measurement setpoint/error signal and is therefore not information-matched
    to black-box extremum optimizers that observe only an objective value.
    """

    def __init__(self, *, kp: float = 0.8, ki: float = 0.02, kd: float = 0.08,
                 action0: float = 0.5, lower: float = 0.0, upper: float = 1.0,
                 integral_limit: float = 5.0) -> None:
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.action = float(action0)
        self.lower = float(lower)
        self.upper = float(upper)
        self.integral_limit = float(integral_limit)
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, measurement: float, setpoint: float = 1.0) -> tuple[float, PIDState]:
        error = float(setpoint) - float(measurement)
        candidate_integral = self._integral + error
        candidate_integral = max(-self.integral_limit, min(self.integral_limit, candidate_integral))
        derivative = error - self._prev_error
        raw = self.action + self.kp * error + self.ki * candidate_integral + self.kd * derivative
        bounded = max(self.lower, min(self.upper, raw))
        if bounded == raw or (bounded == self.upper and error < 0) or (bounded == self.lower and error > 0):
            self._integral = candidate_integral
        self._prev_error = error
        self.action = bounded
        return self.action, PIDState(error, self._integral, derivative, self.action)


@dataclass(frozen=True)
class ESState:
    phase: float
    estimate: float
    action: float


class ExtremumSeekingController:
    """Compact dependency-free extremum-seeking reference.

    ES is information-matched to TRIVAX's black-box objective setting: it uses
    only the scalar objective observation and its internally generated dither.
    """

    def __init__(self, *, action0: float = 0.5, amplitude: float = 0.015,
                 omega: float = 0.37, gain: float = 0.018,
                 lower: float = 0.0, upper: float = 1.0) -> None:
        self.action = float(action0)
        self.amplitude = float(amplitude)
        self.omega = float(omega)
        self.gain = float(gain)
        self.lower = float(lower)
        self.upper = float(upper)
        self._phase = 0.0
        self._baseline = 0.0
        self._initialized = False

    def step(self, observation: float) -> tuple[float, ESState]:
        y = float(observation)
        if not self._initialized:
            self._baseline = y
            self._initialized = True
        self._baseline = 0.96 * self._baseline + 0.04 * y
        centered = y - self._baseline
        demod = centered * math.sin(self._phase)
        self.action += self.gain * demod
        self._phase += self.omega
        dither = self.amplitude * math.sin(self._phase)
        commanded = max(self.lower, min(self.upper, self.action + dither))
        self.action = commanded
        return commanded, ESState(self._phase, demod, commanded)


@dataclass(frozen=True)
class SimpleMPCState:
    chosen_action: float
    predicted_value: float


class SimpleModelPredictiveController:
    """One-step local-model optimizer used as a model-aware reference.

    This class is intentionally transparent and lightweight; it is not a full
    industrial MPC implementation. Results from it must be labelled
    "simple/model-aware reference", not state-of-the-art MPC evidence.
    """

    def __init__(self, *, action0: float = 0.5, step: float = 0.02,
                 lower: float = 0.0, upper: float = 1.0) -> None:
        self.action = float(action0)
        self.step_size = float(step)
        self.lower = float(lower)
        self.upper = float(upper)
        self._last_y: float | None = None
        self._last_action: float | None = None
        self._slope = 0.0

    def step(self, observation: float) -> tuple[float, SimpleMPCState]:
        y = float(observation)
        if self._last_y is not None and self._last_action is not None:
            da = self.action - self._last_action
            if abs(da) > 1e-9:
                local = (y - self._last_y) / da
                self._slope = 0.85 * self._slope + 0.15 * local
        direction = 1.0 if self._slope >= 0 else -1.0
        candidates = (
            self.action,
            max(self.lower, min(self.upper, self.action + direction * self.step_size)),
            max(self.lower, min(self.upper, self.action - direction * self.step_size)),
        )
        predicted = [y + self._slope * (a - self.action) for a in candidates]
        best_i = max(range(len(candidates)), key=lambda i: predicted[i])
        self._last_action = self.action
        self._last_y = y
        self.action = float(candidates[best_i])
        return self.action, SimpleMPCState(self.action, float(predicted[best_i]))
