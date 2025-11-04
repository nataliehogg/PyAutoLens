"""
Line-of-sight (LOS) shear models for adding tidal perturbations to single-plane lensing.

This module implements the line-of-sight shear formalism from Fleury et al. (2021, arXiv:2104.08883),
which treats one deflector as the main lens while adding tidal perturbations from the surrounding
large-scale structure. This is technically simpler than multi-plane lensing while allowing consistent
modeling of sub-critical perturbations.

The implementation includes:
- Full LOS model with 12 parameters (3 convergences, 6 shear components, 3 rotations)
- Minimal LOS model with 8 parameters
- Distortion matrix operations for modifying deflection angles and Hessians
"""

__all__ = ["LineOfSightShear", "LineOfSightShearMinimal"]


class LineOfSightShear:
    """
    Line-of-sight shear model with full parameterization.

    This class adds tidal line-of-sight effects (convergence, shear, and rotation) to single-plane
    lensing. It's not a mass profile itself, but modifies how deflection angles and Hessians are
    computed when present in a tracer.

    The 12 parameters represent perturbations across three planes:
    - Observer-Deflector (od): kappa_od, gamma1_od, gamma2_od, omega_od
    - Observer-Source (os): kappa_os, gamma1_os, gamma2_os, omega_os
    - Deflector-Source (ds): kappa_ds, gamma1_ds, gamma2_ds, omega_ds

    These parameters follow the convention in Fleury et al. (2021), arXiv:2104.08883.

    Parameters
    ----------
    kappa_od : float
        Convergence between observer and deflector
    kappa_os : float
        Convergence between observer and source
    kappa_ds : float
        Convergence between deflector and source
    gamma1_od : float
        First shear component between observer and deflector
    gamma2_od : float
        Second shear component between observer and deflector
    gamma1_os : float
        First shear component between observer and source
    gamma2_os : float
        Second shear component between observer and source
    gamma1_ds : float
        First shear component between deflector and source
    gamma2_ds : float
        Second shear component between deflector and source
    omega_od : float
        Rotation between observer and deflector
    omega_os : float
        Rotation between observer and source
    omega_ds : float
        Rotation between deflector and source
    """

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
        """
        Initialize the line-of-sight shear model.

        Parameters default to zero (no LOS effects).
        """
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

    @staticmethod
    def distort_vector(x, y, kappa=0.0, gamma1=0.0, gamma2=0.0, omega=0.0):
        """
        Apply a distortion matrix to a position vector (x, y).

        The distortion matrix represents the combined effect of convergence, shear, and rotation:

        .. math::
            \\begin{pmatrix}
            x' \\\\
            y'
            \\end{pmatrix}
            =
            \\begin{pmatrix}
            1 - \\kappa - \\gamma_1 & -\\gamma_2 + \\omega \\\\
            -\\gamma_2 - \\omega & 1 - \\kappa + \\gamma_1
            \\end{pmatrix}
            \\begin{pmatrix}
            x \\\\
            y
            \\end{pmatrix}

        Parameters
        ----------
        x : float or ndarray
            x-component of the vector
        y : float or ndarray
            y-component of the vector
        kappa : float
            Convergence
        gamma1 : float
            First shear component
        gamma2 : float
            Second shear component
        omega : float
            Rotation

        Returns
        -------
        x_, y_ : tuple
            The distorted vector components
        """
        x_ = (1 - kappa - gamma1) * x + (-gamma2 + omega) * y
        y_ = (1 - kappa + gamma1) * y - (gamma2 + omega) * x

        return x_, y_

    @staticmethod
    def left_multiply(f_xx, f_xy, f_yx, f_yy, kappa=0.0, gamma1=0.0, gamma2=0.0, omega=0.0):
        """
        Left-multiply a Hessian matrix by a distortion matrix.

        .. math::
            \\mathsf{H}'
            =
            \\begin{pmatrix}
            1 - \\kappa - \\gamma_1 & -\\gamma_2 + \\omega \\\\
            -\\gamma_2 - \\omega & 1 - \\kappa + \\gamma_1
            \\end{pmatrix}
            \\mathsf{H}

        Parameters
        ----------
        f_xx : float or ndarray
            H_{xx} element of the Hessian
        f_xy : float or ndarray
            H_{xy} element of the Hessian
        f_yx : float or ndarray
            H_{yx} element of the Hessian
        f_yy : float or ndarray
            H_{yy} element of the Hessian
        kappa : float
            Convergence
        gamma1 : float
            First shear component
        gamma2 : float
            Second shear component
        omega : float
            Rotation

        Returns
        -------
        f__xx, f__xy, f__yx, f__yy : tuple
            The transformed Hessian components
        """
        f__xx = (1 - kappa - gamma1) * f_xx + (-gamma2 + omega) * f_yx
        f__xy = (1 - kappa - gamma1) * f_xy + (-gamma2 + omega) * f_yy
        f__yx = -(gamma2 + omega) * f_xx + (1 - kappa + gamma1) * f_yx
        f__yy = -(gamma2 + omega) * f_xy + (1 - kappa + gamma1) * f_yy

        return f__xx, f__xy, f__yx, f__yy

    @staticmethod
    def right_multiply(f_xx, f_xy, f_yx, f_yy, kappa=0.0, gamma1=0.0, gamma2=0.0, omega=0.0):
        """
        Right-multiply a Hessian matrix by a distortion matrix.

        .. math::
            \\mathsf{H}'
            =
            \\mathsf{H}
            \\begin{pmatrix}
            1 - \\kappa - \\gamma_1 & -\\gamma_2 + \\omega \\\\
            -\\gamma_2 - \\omega & 1 - \\kappa + \\gamma_1
            \\end{pmatrix}

        Parameters
        ----------
        f_xx : float or ndarray
            H_{xx} element of the Hessian
        f_xy : float or ndarray
            H_{xy} element of the Hessian
        f_yx : float or ndarray
            H_{yx} element of the Hessian
        f_yy : float or ndarray
            H_{yy} element of the Hessian
        kappa : float
            Convergence
        gamma1 : float
            First shear component
        gamma2 : float
            Second shear component
        omega : float
            Rotation

        Returns
        -------
        f__xx, f__xy, f__yx, f__yy : tuple
            The transformed Hessian components
        """
        f__xx = (1 - kappa - gamma1) * f_xx - (gamma2 + omega) * f_xy
        f__xy = (-gamma2 + omega) * f_xx + (1 - kappa + gamma1) * f_xy
        f__yx = (1 - kappa - gamma1) * f_yx - (gamma2 + omega) * f_yy
        f__yy = (-gamma2 + omega) * f_yx + (1 - kappa + gamma1) * f_yy

        return f__xx, f__xy, f__yx, f__yy


class LineOfSightShearMinimal(LineOfSightShear):
    """
    Minimal line-of-sight shear model with reduced parameterization.

    This is a simplified version of the full LOS model with 8 instead of 12 parameters.
    It follows the "minimal model" defined in Fleury et al. (2021), arXiv:2104.08883.

    The reduction is achieved by setting:
    - kappa_os = kappa_los
    - gamma1_os = gamma1_los
    - gamma2_os = gamma2_los
    - omega_os = omega_los
    - kappa_ds = kappa_od
    - gamma1_ds = gamma1_od
    - gamma2_ds = gamma2_od
    - omega_ds = omega_od

    Parameters
    ----------
    kappa_od : float
        Convergence between observer and deflector (also used for deflector-source)
    gamma1_od : float
        First shear component between observer and deflector (also used for deflector-source)
    gamma2_od : float
        Second shear component between observer and deflector (also used for deflector-source)
    omega_od : float
        Rotation between observer and deflector (also used for deflector-source)
    kappa_los : float
        Line-of-sight convergence (used for observer-source)
    gamma1_los : float
        First line-of-sight shear component (used for observer-source)
    gamma2_los : float
        Second line-of-sight shear component (used for observer-source)
    omega_los : float
        Line-of-sight rotation (used for observer-source)
    """

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
        """
        Initialize the minimal line-of-sight shear model.

        The minimal model uses the od parameters for both od and ds planes,
        and the los parameters for the os plane.
        """
        super().__init__(
            kappa_od=kappa_od,
            kappa_os=kappa_los,
            kappa_ds=kappa_od,
            gamma1_od=gamma1_od,
            gamma2_od=gamma2_od,
            gamma1_os=gamma1_los,
            gamma2_os=gamma2_los,
            gamma1_ds=gamma1_od,
            gamma2_ds=gamma2_od,
            omega_od=omega_od,
            omega_os=omega_los,
            omega_ds=omega_od,
        )

        # Store the original parameters for reference
        self.kappa_los = kappa_los
        self.gamma1_los = gamma1_los
        self.gamma2_los = gamma2_los
        self.omega_los = omega_los
