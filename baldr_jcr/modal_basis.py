from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from aotools import zernike  # type: ignore
import math


class ModalBasis(ABC):
    @abstractmethod
    def sample(self, i: int, x: float, y: float) -> float:
        """Sample the ith basis function at coordinates."""
        pass

    def modes(self, xx: np.ndarray, yy: np.ndarray, nmodes: int) -> np.ndarray:
        """evaluate the "sample" function at the coordinates specified, and
        compile a matrix of the function response for modes up to "nmodes".
        xx and yy are vectors of the same length. (xx[i], yy[i]) is the ith
        coordinate. The resulting matrix of this function will have:
          shape == (xx.shape[0], nmodes)
        """
        out = np.zeros((xx.shape[0], nmodes), dtype=float)
        for i in range(nmodes):
            out[:, i] = np.r_[[self.sample(i, x, y) for (x, y) in zip(xx, yy)]]
        return out

    def modes_on_unit_disk(
        self, nsamplex: int, nmodes: int, norm: bool = True
    ) -> np.ndarray:
        """Defines a square grid ensquaring the unit circle, and produces a
        modal matrix on this grid.
        norm==True implies that the modes should be divided by their individual
        standard deviation (across the whole square grid, not just the circle)
        """
        xx, yy = np.meshgrid(
            np.linspace(-1, 1, nsamplex), np.linspace(-1, 1, nsamplex), indexing="xy"
        )
        modes = self.modes(xx.flatten(), yy.flatten(), nmodes=nmodes)
        # modes /= modes.std(axis=0)[None, :]
        return modes


class Zernike(ModalBasis):
    def sample(self, i: int, x: float, y: float) -> float:
        n, m = zernike.zernIndex(i + 2)
        r = (x**2 + y**2) ** 0.5
        # The following snippet is taken from the aotools library source code:
        # https://github.com/AOtools/aotools/blob/main/aotools/functions/zernike.py#L59
        theta = np.arctan2(y, x)
        if m == 0:
            z = np.sqrt(n + 1) * zernike.zernikeRadialFunc(n, 0, r)
        else:
            if m > 0:  # j is even
                z = (
                    np.sqrt(2 * (n + 1))
                    * zernike.zernikeRadialFunc(n, m, r)
                    * np.cos((m * theta))
                )
            else:  # i is odd
                m = abs(m)
                z = (
                    np.sqrt(2 * (n + 1))
                    * zernike.zernikeRadialFunc(n, m, r)
                    * np.sin((m * theta))
                )
        return z


class Fourier(ModalBasis):
    @staticmethod
    def spiral_coords(n: int) -> Tuple[int, int]:
        k = math.ceil((n**0.5 - 1) / 2)
        t = 2 * k + 1
        m = t**2
        t = t - 1
        if n >= m - t:
            return (k - (m - n), -k)
        else:
            m = m - t
        if n >= m - t:
            return (-k, -k + (m - n))
        else:
            m = m - t
        if n >= m - t:
            return (-k + (m - n), k)
        else:
            return (k, k - (m - n - t))

    def sample(self, i: int, x: float, y: float) -> float:
        n = math.floor(i / 2) + 2
        p, q = self.spiral_coords(n)
        if x == -1 and y == -1:
            print(i, n, p, q)
        freq_x: float = 1.0 * np.pi * p / 2
        freq_y: float = 1.0 * np.pi * q / 2
        remainder = i % 2
        if remainder == 0:
            return np.cos(freq_x * x + freq_y * y)
        else:
            return np.sin(freq_x * x + freq_y * y)


class Zonal(ModalBasis):
    NACTX: int = 12

    def sample(self, i: int, x: float, y: float) -> float:
        a = (i % self.NACTX) / (self.NACTX - 1) * 2.0 - 1.0
        b = math.floor(i / self.NACTX) / (self.NACTX - 1) * 2.0 - 1.0
        # no gaussian shenanigans, just straight up actutator-by-actuator modes
        if (a - x) ** 2 + (b - y) ** 2 < (1 / (self.NACTX - 1)) ** 2:
            return 1.0
        else:
            return 0.0


if __name__ == "__main__":
    mb = Zernike()
    import time

    t1 = time.perf_counter()
    print(mb.modes_on_unit_disk(nsamplex=12, nmodes=100))
    t2 = time.perf_counter()
    print(f"time: {t2-t1:0.3e}")
