"""
Tests for line-of-sight shear models.

This test suite verifies the implementation of the LOS shear formalism from
Fleury et al. (2021, arXiv:2104.08883).
"""

import numpy as np
import pytest

import autolens as al


class TestLineOfSightShear:
    """Tests for the full LineOfSightShear class."""

    def test__init__default_parameters(self):
        """Test initialization with default parameters (all zero)."""
        los_shear = al.LineOfSightShear()

        assert los_shear.kappa_od == 0.0
        assert los_shear.kappa_os == 0.0
        assert los_shear.kappa_ds == 0.0
        assert los_shear.gamma1_od == 0.0
        assert los_shear.gamma2_od == 0.0
        assert los_shear.gamma1_os == 0.0
        assert los_shear.gamma2_os == 0.0
        assert los_shear.gamma1_ds == 0.0
        assert los_shear.gamma2_ds == 0.0
        assert los_shear.omega_od == 0.0
        assert los_shear.omega_os == 0.0
        assert los_shear.omega_ds == 0.0

    def test__init__custom_parameters(self):
        """Test initialization with custom parameters."""
        los_shear = al.LineOfSightShear(
            kappa_od=0.1,
            gamma1_od=0.05,
            gamma2_od=-0.03,
            omega_od=0.01,
            kappa_os=0.08,
            gamma1_os=0.04,
            gamma2_os=0.02,
            omega_os=-0.01,
            kappa_ds=0.06,
            gamma1_ds=-0.02,
            gamma2_ds=0.03,
            omega_ds=0.005,
        )

        assert los_shear.kappa_od == 0.1
        assert los_shear.gamma1_od == 0.05
        assert los_shear.gamma2_od == -0.03
        assert los_shear.omega_od == 0.01
        assert los_shear.kappa_os == 0.08
        assert los_shear.gamma1_os == 0.04
        assert los_shear.gamma2_os == 0.02
        assert los_shear.omega_os == -0.01
        assert los_shear.kappa_ds == 0.06
        assert los_shear.gamma1_ds == -0.02
        assert los_shear.gamma2_ds == 0.03
        assert los_shear.omega_ds == 0.005

    def test__distort_vector__identity(self):
        """Test distort_vector with no distortion (identity transformation)."""
        x, y = 1.0, 2.0

        x_distorted, y_distorted = al.LineOfSightShear.distort_vector(x, y)

        assert x_distorted == x
        assert y_distorted == y

    def test__distort_vector__convergence_only(self):
        """Test distort_vector with only convergence."""
        x, y = 1.0, 2.0
        kappa = 0.1

        x_distorted, y_distorted = al.LineOfSightShear.distort_vector(
            x, y, kappa=kappa
        )

        # With kappa only: x' = (1-kappa)*x, y' = (1-kappa)*y
        assert x_distorted == pytest.approx((1 - kappa) * x)
        assert y_distorted == pytest.approx((1 - kappa) * y)

    def test__distort_vector__shear_only(self):
        """Test distort_vector with only shear."""
        x, y = 1.0, 2.0
        gamma1, gamma2 = 0.05, 0.03

        x_distorted, y_distorted = al.LineOfSightShear.distort_vector(
            x, y, gamma1=gamma1, gamma2=gamma2
        )

        # With shear: x' = (1-gamma1)*x + (-gamma2)*y
        #             y' = (1+gamma1)*y - gamma2*x
        assert x_distorted == pytest.approx((1 - gamma1) * x - gamma2 * y)
        assert y_distorted == pytest.approx((1 + gamma1) * y - gamma2 * x)

    def test__distort_vector__rotation_only(self):
        """Test distort_vector with only rotation."""
        x, y = 1.0, 2.0
        omega = 0.02

        x_distorted, y_distorted = al.LineOfSightShear.distort_vector(
            x, y, omega=omega
        )

        # With rotation: x' = x + omega*y
        #                y' = y - omega*x
        assert x_distorted == pytest.approx(x + omega * y)
        assert y_distorted == pytest.approx(y - omega * x)

    def test__distort_vector__combined(self):
        """Test distort_vector with all components."""
        x, y = 1.0, 2.0
        kappa, gamma1, gamma2, omega = 0.1, 0.05, 0.03, 0.01

        x_distorted, y_distorted = al.LineOfSightShear.distort_vector(
            x, y, kappa=kappa, gamma1=gamma1, gamma2=gamma2, omega=omega
        )

        # Full transformation
        expected_x = (1 - kappa - gamma1) * x + (-gamma2 + omega) * y
        expected_y = (1 - kappa + gamma1) * y - (gamma2 + omega) * x

        assert x_distorted == pytest.approx(expected_x)
        assert y_distorted == pytest.approx(expected_y)

    def test__distort_vector__array_input(self):
        """Test distort_vector with numpy array inputs."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.5, 1.0, 1.5])
        kappa, gamma1 = 0.1, 0.05

        x_distorted, y_distorted = al.LineOfSightShear.distort_vector(
            x, y, kappa=kappa, gamma1=gamma1
        )

        expected_x = (1 - kappa - gamma1) * x
        expected_y = (1 - kappa + gamma1) * y

        assert np.allclose(x_distorted, expected_x)
        assert np.allclose(y_distorted, expected_y)

    def test__left_multiply__identity(self):
        """Test left_multiply with no distortion."""
        f_xx, f_xy, f_yx, f_yy = 1.0, 0.2, 0.2, 0.8

        result = al.LineOfSightShear.left_multiply(f_xx, f_xy, f_yx, f_yy)

        assert result[0] == pytest.approx(f_xx)
        assert result[1] == pytest.approx(f_xy)
        assert result[2] == pytest.approx(f_yx)
        assert result[3] == pytest.approx(f_yy)

    def test__left_multiply__convergence(self):
        """Test left_multiply with convergence."""
        f_xx, f_xy, f_yx, f_yy = 1.0, 0.2, 0.2, 0.8
        kappa = 0.1

        f_xx_new, f_xy_new, f_yx_new, f_yy_new = al.LineOfSightShear.left_multiply(
            f_xx, f_xy, f_yx, f_yy, kappa=kappa
        )

        # Left multiply by (1-kappa)*Identity
        assert f_xx_new == pytest.approx((1 - kappa) * f_xx)
        assert f_xy_new == pytest.approx((1 - kappa) * f_xy)
        assert f_yx_new == pytest.approx((1 - kappa) * f_yx)
        assert f_yy_new == pytest.approx((1 - kappa) * f_yy)

    def test__right_multiply__identity(self):
        """Test right_multiply with no distortion."""
        f_xx, f_xy, f_yx, f_yy = 1.0, 0.2, 0.2, 0.8

        result = al.LineOfSightShear.right_multiply(f_xx, f_xy, f_yx, f_yy)

        assert result[0] == pytest.approx(f_xx)
        assert result[1] == pytest.approx(f_xy)
        assert result[2] == pytest.approx(f_yx)
        assert result[3] == pytest.approx(f_yy)

    def test__right_multiply__convergence(self):
        """Test right_multiply with convergence."""
        f_xx, f_xy, f_yx, f_yy = 1.0, 0.2, 0.2, 0.8
        kappa = 0.1

        f_xx_new, f_xy_new, f_yx_new, f_yy_new = al.LineOfSightShear.right_multiply(
            f_xx, f_xy, f_yx, f_yy, kappa=kappa
        )

        # Right multiply by (1-kappa)*Identity
        assert f_xx_new == pytest.approx((1 - kappa) * f_xx)
        assert f_xy_new == pytest.approx((1 - kappa) * f_xy)
        assert f_yx_new == pytest.approx((1 - kappa) * f_yx)
        assert f_yy_new == pytest.approx((1 - kappa) * f_yy)


class TestLineOfSightShearMinimal:
    """Tests for the minimal LineOfSightShearMinimal class."""

    def test__init__parameter_mapping(self):
        """Test that minimal parameters map correctly to full parameters."""
        los_shear = al.LineOfSightShearMinimal(
            kappa_od=0.1,
            gamma1_od=0.05,
            gamma2_od=-0.03,
            omega_od=0.01,
            kappa_los=0.08,
            gamma1_los=0.04,
            gamma2_los=0.02,
            omega_los=-0.01,
        )

        # od parameters used for both od and ds
        assert los_shear.kappa_od == 0.1
        assert los_shear.kappa_ds == 0.1
        assert los_shear.gamma1_od == 0.05
        assert los_shear.gamma1_ds == 0.05
        assert los_shear.gamma2_od == -0.03
        assert los_shear.gamma2_ds == -0.03
        assert los_shear.omega_od == 0.01
        assert los_shear.omega_ds == 0.01

        # los parameters used for os
        assert los_shear.kappa_os == 0.08
        assert los_shear.gamma1_os == 0.04
        assert los_shear.gamma2_os == 0.02
        assert los_shear.omega_os == -0.01

        # Original parameters stored
        assert los_shear.kappa_los == 0.08
        assert los_shear.gamma1_los == 0.04
        assert los_shear.gamma2_los == 0.02
        assert los_shear.omega_los == -0.01

    def test__minimal_inherits_methods(self):
        """Test that minimal model inherits distortion methods."""
        los_shear = al.LineOfSightShearMinimal(
            kappa_od=0.1, gamma1_od=0.05, kappa_los=0.08, gamma1_los=0.04
        )

        x, y = 1.0, 2.0

        # Should use inherited distort_vector method
        x_distorted, y_distorted = los_shear.distort_vector(
            x, y, kappa=los_shear.kappa_od, gamma1=los_shear.gamma1_od
        )

        expected_x = (1 - 0.1 - 0.05) * x
        expected_y = (1 - 0.1 + 0.05) * y

        assert x_distorted == pytest.approx(expected_x)
        assert y_distorted == pytest.approx(expected_y)


class TestTracerLOS:
    """Tests for the TracerLOS class."""

    def test__init__single_plane(self):
        """Test TracerLOS initialization with single lens and source."""
        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.Isothermal(
                centre=(0.0, 0.0),
                einstein_radius=1.0,
            ),
        )

        source_galaxy = al.Galaxy(
            redshift=1.0,
            light=al.lp.Sersic(
                centre=(0.0, 0.0),
                intensity=1.0,
                effective_radius=0.5,
                sersic_index=1.0,
            ),
        )

        los_shear = al.LineOfSightShear(kappa_od=0.05, gamma1_od=0.02)

        tracer_los = al.TracerLOS(
            galaxies=[lens_galaxy, source_galaxy],
            los_shear=los_shear,
        )

        assert tracer_los.los_shear == los_shear
        assert len(tracer_los._lens_galaxies) == 1
        assert len(tracer_los._source_galaxies) == 1
        assert tracer_los.total_planes == 2

    def test__init__multi_plane_raises_error(self):
        """Test that TracerLOS raises error for multi-plane systems."""
        galaxies = [
            al.Galaxy(redshift=0.3, mass=al.mp.Isothermal(einstein_radius=0.5)),
            al.Galaxy(redshift=0.5, mass=al.mp.Isothermal(einstein_radius=1.0)),
            al.Galaxy(redshift=1.0, light=al.lp.Sersic(intensity=1.0)),
        ]

        los_shear = al.LineOfSightShear()

        with pytest.raises(ValueError, match="single-plane lensing"):
            al.TracerLOS(galaxies=galaxies, los_shear=los_shear)

    def test__traced_grid_with_zero_los__matches_standard_tracer(self):
        """Test that TracerLOS with zero LOS parameters matches standard Tracer."""
        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.Isothermal(
                centre=(0.0, 0.0),
                einstein_radius=1.0,
            ),
        )

        source_galaxy = al.Galaxy(redshift=1.0)

        # Create standard tracer
        tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

        # Create LOS tracer with zero perturbations
        los_shear = al.LineOfSightShear()  # All zeros
        tracer_los = al.TracerLOS(
            galaxies=[lens_galaxy, source_galaxy],
            los_shear=los_shear,
        )

        # Test on a grid
        grid = al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.1)

        traced_grids_standard = tracer.traced_grid_2d_list_from(grid=grid)
        traced_grids_los = tracer_los.traced_grid_2d_list_from(grid=grid)

        # Source plane grids should match when LOS effects are zero
        assert np.allclose(
            traced_grids_standard[-1].array, traced_grids_los[-1].array, atol=1e-8
        )

    def test__traced_grid_with_los__differs_from_standard(self):
        """Test that TracerLOS with LOS parameters differs from standard Tracer."""
        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.Isothermal(
                centre=(0.0, 0.0),
                einstein_radius=1.0,
            ),
        )

        source_galaxy = al.Galaxy(redshift=1.0)

        # Create standard tracer
        tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

        # Create LOS tracer with non-zero perturbations
        los_shear = al.LineOfSightShear(
            kappa_od=0.05, gamma1_od=0.03, kappa_os=0.04, gamma1_os=0.02
        )
        tracer_los = al.TracerLOS(
            galaxies=[lens_galaxy, source_galaxy],
            los_shear=los_shear,
        )

        # Test on a grid
        grid = al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.1)

        traced_grids_standard = tracer.traced_grid_2d_list_from(grid=grid)
        traced_grids_los = tracer_los.traced_grid_2d_list_from(grid=grid)

        # Source plane grids should differ when LOS effects are present
        assert not np.allclose(traced_grids_standard[-1].array, traced_grids_los[-1].array)

    def test__deflections_with_los(self):
        """Test that deflections are modified by LOS effects."""
        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.Isothermal(
                centre=(0.0, 0.0),
                einstein_radius=1.0,
            ),
        )

        source_galaxy = al.Galaxy(redshift=1.0)

        los_shear = al.LineOfSightShear(kappa_od=0.05, kappa_ds=0.03, kappa_os=0.04)

        tracer_los = al.TracerLOS(
            galaxies=[lens_galaxy, source_galaxy],
            los_shear=los_shear,
        )

        grid = al.Grid2D.uniform(shape_native=(5, 5), pixel_scales=0.2)

        deflections = tracer_los.deflections_yx_2d_from(grid=grid)

        # Deflections should be non-zero
        assert not np.allclose(deflections.array, 0.0)

        # Check shape
        assert deflections.shape == (25, 2)

    def test__minimal_parameterization(self):
        """Test TracerLOS with minimal LOS parameterization."""
        lens_galaxy = al.Galaxy(
            redshift=0.5,
            mass=al.mp.Isothermal(
                centre=(0.0, 0.0),
                einstein_radius=1.0,
            ),
        )

        source_galaxy = al.Galaxy(redshift=1.0)

        # Use minimal parameterization
        los_shear = al.LineOfSightShearMinimal(
            kappa_od=0.05,
            gamma1_od=0.03,
            kappa_los=0.04,
            gamma1_los=0.02,
        )

        tracer_los = al.TracerLOS(
            galaxies=[lens_galaxy, source_galaxy],
            los_shear=los_shear,
        )

        grid = al.Grid2D.uniform(shape_native=(5, 5), pixel_scales=0.2)

        # Should work with minimal parameterization
        traced_grids = tracer_los.traced_grid_2d_list_from(grid=grid)

        assert len(traced_grids) == 2
        assert traced_grids[0].shape == (25, 2)
        assert traced_grids[1].shape == (25, 2)
