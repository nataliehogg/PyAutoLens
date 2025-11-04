"""
Tracer with line-of-sight (LOS) shear corrections for single-plane lensing.

This module implements a Tracer variant that applies line-of-sight shear effects
based on Fleury et al. (2021, arXiv:2104.08883). It extends the standard Tracer
to modify deflection angles, Hessians, and other lensing quantities according to
the LOS shear formalism.
"""

from typing import List, Union, Optional
import numpy as np

import autoarray as aa
import autogalaxy as ag

from autolens.lens.tracer import Tracer
from autolens.lens.line_of_sight_shear import LineOfSightShear, LineOfSightShearMinimal

__all__ = ["TracerLOS"]


class TracerLOS(Tracer):
    """
    A Tracer that applies line-of-sight (LOS) shear corrections to single-plane lensing.

    This class extends the standard Tracer to include tidal line-of-sight effects from
    the surrounding large-scale structure. It modifies how deflection angles and Hessians
    are computed while maintaining compatibility with the PyAutoLens API.

    The LOS corrections are applied in the dominant-lens approximation, where one deflector
    is treated as the main lens and perturbations from other structures are treated as
    tidal effects characterized by convergence, shear, and rotation.

    Note: This is incompatible with multi-plane lensing. If you need multi-plane ray tracing
    with explicit line-of-sight galaxies, use the standard Tracer with line_of_sight_galaxies.

    Parameters
    ----------
    galaxies
        The list of galaxies (should typically be a single main lens and source)
    los_shear
        The LineOfSightShear or LineOfSightShearMinimal object containing LOS parameters
    cosmology
        The cosmology used for lensing calculations
    """

    def __init__(
        self,
        galaxies: Union[List[ag.Galaxy], ag.Galaxies],
        los_shear: Union[LineOfSightShear, LineOfSightShearMinimal],
        cosmology: ag.cosmo.LensingCosmology = None,
    ):
        """
        Initialize TracerLOS with galaxies and LOS shear parameters.

        Parameters
        ----------
        galaxies
            The main lens and source galaxies (should not include explicit LOS galaxies)
        los_shear
            The line-of-sight shear object with tidal perturbation parameters
        cosmology
            The cosmology for lensing calculations
        """
        super().__init__(galaxies=galaxies, cosmology=cosmology)

        self.los_shear = los_shear

        # Verify we're in single-plane mode for the lens
        # (we expect typically 2 planes: one lens, one source)
        if len(self.plane_redshifts) > 2:
            raise ValueError(
                "TracerLOS is designed for single-plane lensing systems. "
                f"Found {len(self.plane_redshifts)} planes with redshifts {self.plane_redshifts}. "
                "For multi-plane systems, use the standard Tracer with line_of_sight_galaxies."
            )

        # Separate lens and source galaxies
        self._lens_galaxies = []
        self._source_galaxies = []

        for galaxy in self.galaxies_ascending_redshift:
            if galaxy == self.galaxies_ascending_redshift[-1]:
                self._source_galaxies.append(galaxy)
            else:
                self._lens_galaxies.append(galaxy)

    @aa.grid_dec.to_grid
    def traced_grid_2d_list_from(
        self, grid: aa.type.Grid2DLike, plane_index_limit: Optional[int] = None
    ) -> List[aa.type.Grid2DLike]:
        """
        Returns ray-traced grids with LOS corrections applied.

        This overrides the standard tracer's ray-tracing to include LOS effects:
        1. Apply od distortion to image-plane positions
        2. Compute main lens deflections at distorted positions
        3. Apply ds distortion to deflections
        4. Add os distortion contribution
        5. Ray-trace to source plane

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates on which ray-tracing is performed
        plane_index_limit
            The integer index of the last plane to ray-trace to

        Returns
        -------
        traced_grid_list
            List of ray-traced grids for each plane with LOS corrections
        """
        # Start with the image-plane grid
        traced_grid_list = [grid]

        if plane_index_limit is None:
            plane_index_limit = self.total_planes

        if plane_index_limit == 1:
            return traced_grid_list

        # Extract coordinates as numpy arrays
        grid_y = np.array(grid[:, 0])
        grid_x = np.array(grid[:, 1])

        # Apply od distortion to get position at deflector plane
        x_d, y_d = self.los_shear.distort_vector(
            grid_x,
            grid_y,
            kappa=self.los_shear.kappa_od,
            gamma1=self.los_shear.gamma1_od,
            gamma2=self.los_shear.gamma2_od,
            omega=self.los_shear.omega_od,
        )

        # Create grid at deflector plane
        grid_d = grid.copy()
        grid_d[:, 0] = y_d
        grid_d[:, 1] = x_d

        # Compute deflections from main lens at deflector plane
        lens_plane_galaxies = ag.Galaxies(galaxies=self._lens_galaxies)
        deflections = lens_plane_galaxies.deflections_yx_2d_from(grid=grid_d)

        # Extract deflection components - handle both array and VectorYX2D types
        if hasattr(deflections, 'array'):
            deflections_y = np.array(deflections.array[:, 0])
            deflections_x = np.array(deflections.array[:, 1])
        else:
            deflections_y = np.array(deflections[:, 0])
            deflections_x = np.array(deflections[:, 1])

        # Apply ds distortion to deflections
        deflections_x_ds, deflections_y_ds = self.los_shear.distort_vector(
            deflections_x,
            deflections_y,
            kappa=self.los_shear.kappa_ds,
            gamma1=self.los_shear.gamma1_ds,
            gamma2=self.los_shear.gamma2_ds,
            omega=self.los_shear.omega_ds,
        )

        # Compute os distortion (perturbation in absence of main lens)
        x_os, y_os = self.los_shear.distort_vector(
            grid_x,
            grid_y,
            kappa=self.los_shear.kappa_os,
            gamma1=self.los_shear.gamma1_os,
            gamma2=self.los_shear.gamma2_os,
            omega=self.los_shear.omega_os,
        )

        # Complete source position: β = θ - (α_ds + (θ - θ_os))
        # Which simplifies to: β = θ_os - α_ds
        source_y = y_os - deflections_y_ds
        source_x = x_os - deflections_x_ds

        # Create source-plane grid
        source_grid = grid.copy()
        source_grid[:, 0] = source_y
        source_grid[:, 1] = source_x

        traced_grid_list.append(source_grid)

        if plane_index_limit is not None and plane_index_limit < len(traced_grid_list):
            return traced_grid_list[:plane_index_limit]

        return traced_grid_list

    @aa.grid_dec.to_vector_yx
    def deflections_yx_2d_from(
        self, grid: aa.type.Grid2DLike
    ) -> Union[aa.VectorYX2D, aa.VectorYX2DIrregular]:
        """
        Returns deflection angles with LOS corrections.

        This computes the effective deflection from image plane to source plane,
        accounting for LOS perturbations.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where deflections are evaluated

        Returns
        -------
        deflections
            The deflection angles with LOS corrections
        """
        traced_grids = self.traced_grid_2d_list_from(grid=grid)

        # Deflections are image_plane - source_plane
        return traced_grids[0] - traced_grids[-1]

    def convergence_2d_from_with_los(self, grid: aa.type.Grid2DLike, buffer: float = 0.01) -> aa.Array2D:
        """
        Returns convergence including LOS contribution.

        This is computed from the Hessian: kappa = (f_xx + f_yy) / 2

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where convergence is evaluated
        buffer
            The spacing used for finite difference calculation of the Hessian (default: 0.01 arcsec)

        Returns
        -------
        convergence
            The convergence including LOS effects
        """
        # Compute from Hessian to ensure consistency
        f_xx, f_xy, f_yx, f_yy = self.hessian_from_with_los(grid=grid, buffer=buffer)

        # kappa = (f_xx + f_yy) / 2
        # Extract to plain array to avoid ArrayIrregular issues
        try:
            convergence_vals = np.asarray([float(v) for v in (f_xx + f_yy)]) / 2.0
        except (TypeError, ValueError):
            convergence_vals = (f_xx + f_yy) / 2.0

        # Convert to Array2D or ArrayIrregular
        if isinstance(grid, aa.Grid2D):
            return aa.Array2D(values=convergence_vals, mask=grid.mask)
        else:
            return aa.ArrayIrregular(values=convergence_vals)


    def hessian_from_with_los(
        self, grid: aa.type.Grid2DLike, buffer: float = 0.01
    ) -> tuple:
        """
        Returns Hessian components with LOS corrections.

        The Hessian transformation follows:
        1. Apply od distortion to grid positions
        2. Compute main lens Hessian at distorted positions
        3. Left-multiply by (1 - Gamma_ds) distortion matrix
        4. Right-multiply by (1 - Gamma_od) distortion matrix
        5. Add LOS contribution from os terms

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where Hessian is evaluated
        buffer
            The spacing in the y and x directions around each grid coordinate where deflection
            angles are computed and used to estimate the derivative (default: 0.01 arcsec).
            Smaller values give more accurate Hessians but may be sensitive to numerical precision.

        Returns
        -------
        f_xx, f_xy, f_yx, f_yy
            The Hessian components with LOS corrections
        """
        # Apply od distortion to get positions at deflector
        x_d, y_d = self.los_shear.distort_vector(
            grid[:, 1],
            grid[:, 0],
            kappa=self.los_shear.kappa_od,
            gamma1=self.los_shear.gamma1_od,
            gamma2=self.los_shear.gamma2_od,
            omega=self.los_shear.omega_od,
        )

        grid_d = grid.copy()
        grid_d[:, 0] = y_d
        grid_d[:, 1] = x_d

        # Compute Hessian of main lens directly
        # IMPORTANT: autogalaxy returns (f_yy, f_xy, f_yx, f_xx) NOT (f_xx, f_xy, f_yx, f_yy)
        lens_plane_galaxies = ag.Galaxies(galaxies=self._lens_galaxies)
        f_yy, f_xy, f_yx, f_xx = lens_plane_galaxies.hessian_from(grid=grid_d, buffer=buffer)

        # Extract to arrays if needed
        if hasattr(f_xx, 'slim'):
            f_xx = f_xx.slim
            f_xy = f_xy.slim
            f_yx = f_yx.slim
            f_yy = f_yy.slim

        # Left-multiply by (1 - Gamma_ds)
        f_xx, f_xy, f_yx, f_yy = self.los_shear.left_multiply(
            f_xx, f_xy, f_yx, f_yy,
            kappa=self.los_shear.kappa_ds,
            gamma1=self.los_shear.gamma1_ds,
            gamma2=self.los_shear.gamma2_ds,
            omega=self.los_shear.omega_ds,
        )

        # Right-multiply by (1 - Gamma_od)
        f_xx, f_xy, f_yx, f_yy = self.los_shear.right_multiply(
            f_xx, f_xy, f_yx, f_yy,
            kappa=self.los_shear.kappa_od,
            gamma1=self.los_shear.gamma1_od,
            gamma2=self.los_shear.gamma2_od,
            omega=self.los_shear.omega_od,
        )

        # Add LOS contribution in absence of main lens
        f_xx += self.los_shear.kappa_os + self.los_shear.gamma1_os
        f_xy += self.los_shear.gamma2_os - self.los_shear.omega_os
        f_yx += self.los_shear.gamma2_os + self.los_shear.omega_os
        f_yy += self.los_shear.kappa_os - self.los_shear.gamma1_os

        # Return as ArrayIrregular for compatibility
        if isinstance(grid, aa.Grid2D):
            return (
                aa.Array2D(values=f_xx, mask=grid.mask),
                aa.Array2D(values=f_xy, mask=grid.mask),
                aa.Array2D(values=f_yx, mask=grid.mask),
                aa.Array2D(values=f_yy, mask=grid.mask),
            )
        else:
            return (
                aa.ArrayIrregular(values=np.asarray([float(v) for v in f_xx])),
                aa.ArrayIrregular(values=np.asarray([float(v) for v in f_xy])),
                aa.ArrayIrregular(values=np.asarray([float(v) for v in f_yx])),
                aa.ArrayIrregular(values=np.asarray([float(v) for v in f_yy])),
            )

    def shear_yx_2d_from_with_los(self, grid: aa.type.Grid2DLike, buffer: float = 0.01) -> aa.VectorYX2D:
        """
        Returns shear components with LOS corrections.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where shear is evaluated
        buffer
            The spacing used for finite difference calculation of the Hessian (default: 0.01 arcsec)

        Returns
        -------
        shear
            The shear (gamma1, gamma2) components with LOS corrections
        """
        f_xx, f_xy, f_yx, f_yy = self.hessian_from_with_los(grid=grid, buffer=buffer)

        # Convert Hessian to shear
        # gamma1 = (f_xx - f_yy) / 2
        # gamma2 = (f_xy + f_yx) / 2
        gamma1_raw = (f_xx - f_yy) / 2.0
        gamma2_raw = (f_xy + f_yx) / 2.0

        # Extract values to plain arrays by iterating
        try:
            gamma1 = np.asarray([float(g) for g in gamma1_raw])
            gamma2 = np.asarray([float(g) for g in gamma2_raw])
        except (TypeError, ValueError):
            # If iteration fails, try direct conversion
            gamma1 = np.array([float(gamma1_raw)])
            gamma2 = np.array([float(gamma2_raw)])

        # Convert to VectorYX2D - handle both Grid2D and Grid2DIrregular
        if isinstance(grid, aa.Grid2D):
            # For Grid2D, create proper VectorYX2D
            shear_values = np.zeros((grid.shape[0], 2))
            shear_values[:, 0] = gamma1
            shear_values[:, 1] = gamma2
            return aa.VectorYX2D(values=shear_values, mask=grid.mask)
        else:
            # For Grid2DIrregular, return VectorYX2DIrregular
            shear_values = np.zeros((len(grid), 2))
            shear_values[:, 0] = gamma1
            shear_values[:, 1] = gamma2
            return aa.VectorYX2DIrregular(values=shear_values, grid=grid)
