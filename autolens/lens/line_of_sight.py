from __future__ import annotations

from typing import Dict, Tuple

import autofit as af
import numpy as np


class LineOfSightBase(af.ModelObject):
    """
    Base object describing line-of-sight (LOS) tidal corrections as defined in Fleury et al. (2021, arXiv:2104.08883).

    Concrete implementations must provide the :pyattr:`parameters` property returning the full set of LOS coefficients.
    """

    def __init__(self):
        super().__init__()

    @property
    def parameters(self) -> Dict[str, float]:
        """
        Returns all LOS parameters expressed in the full Fleury et al. convention.
        """

        raise NotImplementedError

    @staticmethod
    def distortion_matrix(
        kappa: float = 0.0,
        gamma1: float = 0.0,
        gamma2: float = 0.0,
        omega: float = 0.0,
        xp=np,
    ):
        """
        Returns the 2x2 amplification matrix used throughout the LOS formalism.
        """

        return xp.array(
            [
                [1 - kappa - gamma1, -gamma2 + omega],
                [-gamma2 - omega, 1 - kappa + gamma1],
            ]
        )

    @classmethod
    def distort_vector(
        cls,
        x,
        y,
        kappa: float = 0.0,
        gamma1: float = 0.0,
        gamma2: float = 0.0,
        omega: float = 0.0,
        xp=np,
    ) -> Tuple:
        """
        Applies the distortion matrix to vectors expressed in the (x, y) ordering used by lenstronomy.
        """

        matrix = cls.distortion_matrix(
            kappa=kappa, gamma1=gamma1, gamma2=gamma2, omega=omega, xp=xp
        )

        vec = xp.stack([x, y], axis=-1)
        distorted = xp.einsum("ij,...j->...i", matrix, vec)

        return distorted[..., 0], distorted[..., 1]

    @classmethod
    def apply_matrix(
        cls, matrix, x, y, xp=np
    ) -> Tuple[
        np.ndarray, np.ndarray
    ]:  # pragma: no cover - simple wrapper exercised indirectly
        """
        Applies a pre-computed 2x2 matrix to vectors expressed in the (x, y) ordering.
        """

        vec = xp.stack([x, y], axis=-1)
        distorted = xp.einsum("ij,...j->...i", matrix, vec)
        return distorted[..., 0], distorted[..., 1]

    @classmethod
    def left_multiply(
        cls,
        f_xx,
        f_xy,
        f_yx,
        f_yy,
        kappa: float = 0.0,
        gamma1: float = 0.0,
        gamma2: float = 0.0,
        omega: float = 0.0,
        xp=np,
    ):
        matrix = cls.distortion_matrix(
            kappa=kappa, gamma1=gamma1, gamma2=gamma2, omega=omega, xp=xp
        )
        f__xx = matrix[0, 0] * f_xx + matrix[0, 1] * f_yx
        f__xy = matrix[0, 0] * f_xy + matrix[0, 1] * f_yy
        f__yx = matrix[1, 0] * f_xx + matrix[1, 1] * f_yx
        f__yy = matrix[1, 0] * f_xy + matrix[1, 1] * f_yy
        return f__xx, f__xy, f__yx, f__yy

    @classmethod
    def right_multiply(
        cls,
        f_xx,
        f_xy,
        f_yx,
        f_yy,
        kappa: float = 0.0,
        gamma1: float = 0.0,
        gamma2: float = 0.0,
        omega: float = 0.0,
        xp=np,
    ):
        f__xx = (1 - kappa - gamma1) * f_xx - (gamma2 + omega) * f_xy
        f__xy = (-gamma2 + omega) * f_xx + (1 - kappa + gamma1) * f_xy
        f__yx = (1 - kappa - gamma1) * f_yx - (gamma2 + omega) * f_yy
        f__yy = (-gamma2 + omega) * f_yx + (1 - kappa + gamma1) * f_yy
        return f__xx, f__xy, f__yx, f__yy


class LineOfSight(LineOfSightBase):
    """
    Full LOS parameterisation with the 12 convergence, shear and rotation coefficients from Fleury et al. (2021).
    """

    param_names = [
        "kappa_od",
        "kappa_os",
        "kappa_ds",
        "gamma1_od",
        "gamma2_od",
        "gamma1_os",
        "gamma2_os",
        "gamma1_ds",
        "gamma2_ds",
        "omega_od",
        "omega_os",
        "omega_ds",
    ]

    def __init__(
        self,
        kappa_od: float = 0.0,
        kappa_os: float = 0.0,
        kappa_ds: float = 0.0,
        gamma1_od: float = 0.0,
        gamma2_od: float = 0.0,
        gamma1_os: float = 0.0,
        gamma2_os: float = 0.0,
        gamma1_ds: float = 0.0,
        gamma2_ds: float = 0.0,
        omega_od: float = 0.0,
        omega_os: float = 0.0,
        omega_ds: float = 0.0,
    ):
        super().__init__()
        self.kappa_od = kappa_od
        self.kappa_os = kappa_os
        self.kappa_ds = kappa_ds
        self.gamma1_od = gamma1_od
        self.gamma2_od = gamma2_od
        self.gamma1_os = gamma1_os
        self.gamma2_os = gamma2_os
        self.gamma1_ds = gamma1_ds
        self.gamma2_ds = gamma2_ds
        self.omega_od = omega_od
        self.omega_os = omega_os
        self.omega_ds = omega_ds

    @property
    def parameters(self) -> Dict[str, float]:
        return {
            "kappa_od": self.kappa_od,
            "kappa_os": self.kappa_os,
            "kappa_ds": self.kappa_ds,
            "gamma1_od": self.gamma1_od,
            "gamma2_od": self.gamma2_od,
            "gamma1_os": self.gamma1_os,
            "gamma2_os": self.gamma2_os,
            "gamma1_ds": self.gamma1_ds,
            "gamma2_ds": self.gamma2_ds,
            "omega_od": self.omega_od,
            "omega_os": self.omega_os,
            "omega_ds": self.omega_ds,
        }


class LineOfSightMinimal(LineOfSightBase):
    """
    Minimal LOS parameterisation with 8 free parameters, following Appendix C of Fleury et al. (2021).
    """

    param_names = [
        "kappa_od",
        "gamma1_od",
        "gamma2_od",
        "omega_od",
        "kappa_los",
        "gamma1_los",
        "gamma2_los",
        "omega_los",
    ]

    def __init__(
        self,
        kappa_od: float = 0.0,
        gamma1_od: float = 0.0,
        gamma2_od: float = 0.0,
        omega_od: float = 0.0,
        kappa_los: float = 0.0,
        gamma1_los: float = 0.0,
        gamma2_los: float = 0.0,
        omega_los: float = 0.0,
    ):
        super().__init__()
        self.kappa_od = kappa_od
        self.gamma1_od = gamma1_od
        self.gamma2_od = gamma2_od
        self.omega_od = omega_od
        self.kappa_los = kappa_los
        self.gamma1_los = gamma1_los
        self.gamma2_los = gamma2_los
        self.omega_los = omega_los

    @property
    def parameters(self) -> Dict[str, float]:
        return {
            "kappa_od": self.kappa_od,
            "kappa_os": self.kappa_los,
            "kappa_ds": self.kappa_od,
            "gamma1_od": self.gamma1_od,
            "gamma2_od": self.gamma2_od,
            "gamma1_os": self.gamma1_los,
            "gamma2_os": self.gamma2_los,
            "gamma1_ds": self.gamma1_od,
            "gamma2_ds": self.gamma2_od,
            "omega_od": self.omega_od,
            "omega_os": self.omega_los,
            "omega_ds": self.omega_od,
        }
