
# baldr_rtc/rtc/controllers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


class Controller(Protocol):
    def reset(self) -> None: ...
    def process(self, e: np.ndarray) -> np.ndarray: ...


def _as_vec(x, n: int, default: float = 0.0) -> np.ndarray:
    if x is None:
        return np.full(n, default, dtype=np.float64)
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size == 1:
        return np.full(n, float(a[0]), dtype=np.float64)
    if a.size != n:
        raise ValueError(f"expected size {n}, got {a.size}")
    return a


@dataclass
class PIDController:
    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray
    dt: float
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None
    setpoint: Optional[np.ndarray] = None

    _i: np.ndarray = None  # integral state
    _e_prev: np.ndarray = None

    def __post_init__(self) -> None:
        self.kp = np.asarray(self.kp, dtype=np.float64).reshape(-1)
        self.ki = np.asarray(self.ki, dtype=np.float64).reshape(-1)
        self.kd = np.asarray(self.kd, dtype=np.float64).reshape(-1)
        n = self.kp.size
        if self.ki.size != n or self.kd.size != n:
            raise ValueError("kp/ki/kd size mismatch")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")

        self._i = np.zeros(n, dtype=np.float64)
        self._e_prev = np.zeros(n, dtype=np.float64)

        if self.u_min is not None:
            self.u_min = np.asarray(self.u_min, dtype=np.float64).reshape(-1)
            if self.u_min.size == 1:
                self.u_min = np.full(n, float(self.u_min[0]), dtype=np.float64)
            elif self.u_min.size != n:
                raise ValueError("u_min size mismatch")

        if self.u_max is not None:
            self.u_max = np.asarray(self.u_max, dtype=np.float64).reshape(-1)
            if self.u_max.size == 1:
                self.u_max = np.full(n, float(self.u_max[0]), dtype=np.float64)
            elif self.u_max.size != n:
                raise ValueError("u_max size mismatch")

        if self.setpoint is not None:
            self.setpoint = np.asarray(self.setpoint, dtype=np.float64).reshape(-1)
            if self.setpoint.size == 1:
                self.setpoint = np.full(n, float(self.setpoint[0]), dtype=np.float64)
            elif self.setpoint.size != n:
                raise ValueError("setpoint size mismatch")

    def reset(self) -> None:
        self._i.fill(0.0)
        self._e_prev.fill(0.0)

    def process(self, e: np.ndarray) -> np.ndarray:
        e = np.asarray(e, dtype=np.float64).reshape(-1)
        if self.setpoint is not None:
            e = e - self.setpoint

        # i[k] = i[k-1] + e*dt
        self._i += e * self.dt

        # d = (e - e_prev)/dt
        d = (e - self._e_prev) * (1.0 / self.dt)
        self._e_prev[:] = e

        u = self.kp * e + self.ki * self._i + self.kd * d

        if self.u_min is not None or self.u_max is not None:
            if self.u_min is None:
                np.minimum(u, self.u_max, out=u)
            elif self.u_max is None:
                np.maximum(u, self.u_min, out=u)
            else:
                np.clip(u, self.u_min, self.u_max, out=u)

        return u


@dataclass
class LeakyIntegrator:
    rho: np.ndarray          # leak factor in [0,1] typically
    ki: np.ndarray           # integral gain (per element)
    kp: Optional[np.ndarray] = None  # optional proportional term
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None
    setpoint: Optional[np.ndarray] = None

    _state: np.ndarray = None

    def __post_init__(self) -> None:
        self.rho = np.asarray(self.rho, dtype=np.float64).reshape(-1)
        self.ki = np.asarray(self.ki, dtype=np.float64).reshape(-1)
        n = self.rho.size
        if self.ki.size != n:
            raise ValueError("rho/ki size mismatch")

        if self.kp is not None:
            self.kp = np.asarray(self.kp, dtype=np.float64).reshape(-1)
            if self.kp.size == 1:
                self.kp = np.full(n, float(self.kp[0]), dtype=np.float64)
            elif self.kp.size != n:
                raise ValueError("kp size mismatch")

        self._state = np.zeros(n, dtype=np.float64)

        if self.u_min is not None:
            self.u_min = np.asarray(self.u_min, dtype=np.float64).reshape(-1)
            if self.u_min.size == 1:
                self.u_min = np.full(n, float(self.u_min[0]), dtype=np.float64)
            elif self.u_min.size != n:
                raise ValueError("u_min size mismatch")

        if self.u_max is not None:
            self.u_max = np.asarray(self.u_max, dtype=np.float64).reshape(-1)
            if self.u_max.size == 1:
                self.u_max = np.full(n, float(self.u_max[0]), dtype=np.float64)
            elif self.u_max.size != n:
                raise ValueError("u_max size mismatch")

        if self.setpoint is not None:
            self.setpoint = np.asarray(self.setpoint, dtype=np.float64).reshape(-1)
            if self.setpoint.size == 1:
                self.setpoint = np.full(n, float(self.setpoint[0]), dtype=np.float64)
            elif self.setpoint.size != n:
                raise ValueError("setpoint size mismatch")

    def reset(self) -> None:
        self._state.fill(0.0)

    def process(self, e: np.ndarray) -> np.ndarray:
        e = np.asarray(e, dtype=np.float64).reshape(-1)
        if self.setpoint is not None:
            e = e - self.setpoint

        # state = rho*state + ki*e
        self._state *= self.rho
        self._state += self.ki * e

        u = self._state if self.kp is None else (self._state + self.kp * e)

        if self.u_min is not None or self.u_max is not None:
            if self.u_min is None:
                np.minimum(u, self.u_max, out=u)
            elif self.u_max is None:
                np.maximum(u, self.u_min, out=u)
            else:
                np.clip(u, self.u_min, self.u_max, out=u)

        return u


def build_controller(
    controller_type: str,
    n: int,
    *,
    dt: float = 1.0,
    kp=None,
    ki=None,
    kd=None,
    rho=None,
    u_min=None,
    u_max=None,
    setpoint=None,
) -> Controller:
    t = controller_type.strip().lower()
    if t == "pid":
        return PIDController(
            kp=_as_vec(kp, n, 0.0),
            ki=_as_vec(ki, n, 0.0),
            kd=_as_vec(kd, n, 0.0),
            dt=float(dt),
            u_min=_as_vec(u_min, n) if u_min is not None else None,
            u_max=_as_vec(u_max, n) if u_max is not None else None,
            setpoint=_as_vec(setpoint, n) if setpoint is not None else None,
        )
    if t == "leaky":
        return LeakyIntegrator(
            rho=_as_vec(rho, n, 1.0),
            ki=_as_vec(ki, n, 0.0),
            kp=_as_vec(kp, n, 0.0) if kp is not None else None,
            u_min=_as_vec(u_min, n) if u_min is not None else None,
            u_max=_as_vec(u_max, n) if u_max is not None else None,
            setpoint=_as_vec(setpoint, n) if setpoint is not None else None,
        )
    raise UserWarning(f"controller_type '{controller_type}' not implemented")