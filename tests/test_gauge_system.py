"""
Unit tests for the analog gauge digitization system.

Covers every pure-math function that was previously untested.
Run from the project root:
    pytest src/tests/ -v

These tests have ZERO video/camera/OpenCV-window dependencies; they test
the math layer in isolation.
"""
import math
import sys
import os
import pytest

# Make src/ importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alert_logic import (
    classify_range, AlertStatus,
    get_interval, AlertStateMachine,
    LOW_ALERT_PERCENT, WARNING_PERCENT,
)
from angle_calculation import calculate_tip_angle, get_pivot
from gauge_mapping import angle_to_percent, GaugeScale, calibrate_from_angles
from config import GaugeConfig, CalibrationError


# ===========================================================================
# alert_logic
# ===========================================================================

class TestClassifyRange:
    def test_below_low_alert(self):
        assert classify_range(0.0)  == AlertStatus.LOW_ALERT
        assert classify_range(19.9) == AlertStatus.LOW_ALERT

    def test_at_low_alert_boundary(self):
        # boundary is < LOW_ALERT_PERCENT, so exactly at boundary is WARNING
        assert classify_range(LOW_ALERT_PERCENT) == AlertStatus.WARNING

    def test_warning_band(self):
        assert classify_range(LOW_ALERT_PERCENT + 0.1) == AlertStatus.WARNING
        assert classify_range(WARNING_PERCENT - 0.1)   == AlertStatus.WARNING

    def test_normal(self):
        assert classify_range(WARNING_PERCENT)     == AlertStatus.NORMAL
        assert classify_range(WARNING_PERCENT + 1) == AlertStatus.NORMAL
        assert classify_range(100.0)               == AlertStatus.NORMAL

    def test_thresholds_consistent_with_constants(self):
        """Thresholds in the function must match the exported constants.
        If this breaks, classify_range and LOW_ALERT_PERCENT/WARNING_PERCENT
        have drifted apart."""
        assert classify_range(LOW_ALERT_PERCENT - 0.01) == AlertStatus.LOW_ALERT
        assert classify_range(WARNING_PERCENT  - 0.01) == AlertStatus.WARNING


class TestGetInterval:
    def test_midrange(self):
        lower, upper = get_interval(55.0)
        assert lower == 54.0
        assert upper == 56.0

    def test_zero(self):
        lower, upper = get_interval(0.0)
        assert lower == 0.0
        assert upper == 2.0

    def test_near_100(self):
        lower, upper = get_interval(99.0)
        assert lower == 98.0
        assert upper == 100.0

    def test_exact_major_boundary(self):
        lower, upper = get_interval(40.0)
        assert lower == 40.0
        assert upper == 42.0


class TestAlertStateMachine:
    def test_raises_only_after_streak(self):
        sm = AlertStateMachine(streak_needed=3)
        assert sm.update(10.0) == AlertStatus.NORMAL   # streak 1 -- no change yet
        assert sm.update(10.0) == AlertStatus.NORMAL   # streak 2
        assert sm.update(10.0) == AlertStatus.LOW_ALERT  # streak 3 -- triggers

    def test_single_bad_frame_does_not_clear_alert(self):
        sm = AlertStateMachine(streak_needed=2)
        sm.update(10.0)
        sm.update(10.0)  # now LOW_ALERT
        assert sm.update(90.0) == AlertStatus.LOW_ALERT  # 1 normal frame -- not cleared yet
        assert sm.update(90.0) == AlertStatus.NORMAL     # 2nd -- now clears

    def test_reset(self):
        sm = AlertStateMachine(streak_needed=2)
        sm.update(10.0)
        sm.update(10.0)
        sm.reset()
        assert sm.update(10.0) == AlertStatus.NORMAL  # streak restarted

    def test_candidate_switch_resets_streak(self):
        sm = AlertStateMachine(streak_needed=5)
        sm.update(10.0)  # LOW candidate, streak 1
        sm.update(10.0)  # streak 2
        sm.update(90.0)  # NORMAL candidate, streak resets to 1
        sm.update(90.0)  # NORMAL streak 2
        # Should still be NORMAL (never reached 5 for LOW_ALERT)
        assert sm.update(90.0) == AlertStatus.NORMAL


# ===========================================================================
# angle_calculation
# ===========================================================================

class TestGetPivot:
    def test_returns_center_bottom(self):
        px, py = get_pivot((600, 800))
        assert px == 400
        assert py == 450   # 600 * 0.75


class TestCalculateTipAngle:
    """
    The function returns angle in 0-360 standard math convention.
    Pivot defaults to bottom-center of an 800x600 frame = (400, 450).
    """

    def _make_line(self, tx, ty, pivot=(400, 450)):
        """Build a line from pivot -> (tx, ty) in [[x1,y1,x2,y2]] format."""
        px, py = pivot
        return [[px, py, tx, ty]]

    def test_right_is_zero(self):
        # Tip directly to the right of pivot -> 0 degrees
        line = self._make_line(500, 450)
        angle = calculate_tip_angle(line, (600, 800), pivot=(400, 450))
        assert abs(angle - 0.0) < 1.0 or abs(angle - 360.0) < 1.0

    def test_up_is_90(self):
        # Tip directly above pivot -> 90 degrees (Y is inverted in image space)
        line = self._make_line(400, 350)
        angle = calculate_tip_angle(line, (600, 800), pivot=(400, 450))
        assert abs(angle - 90.0) < 1.0

    def test_left_is_180(self):
        # Tip to the left -> 180 degrees
        line = self._make_line(300, 450)
        angle = calculate_tip_angle(line, (600, 800), pivot=(400, 450))
        assert abs(angle - 180.0) < 1.0

    def test_tip_is_farther_point(self):
        # The function should pick whichever endpoint is farther from pivot
        # as the tip, regardless of order.
        px, py = 400, 450
        # Far point is (500, 450), close point is (410, 450)
        line_a = [[px, py, 500, 450]]   # far point second
        line_b = [[500, 450, px, py]]   # far point first
        pivot = (px, py)
        a1 = calculate_tip_angle(line_a, (600, 800), pivot=pivot)
        a2 = calculate_tip_angle(line_b, (600, 800), pivot=pivot)
        assert abs(a1 - a2) < 0.5

    def test_no_negative_angles(self):
        # Result should always be in [0, 360)
        for tx, ty in [(300, 550), (500, 550), (300, 350), (500, 350)]:
            line = self._make_line(tx, ty)
            a = calculate_tip_angle(line, (600, 800), pivot=(400, 450))
            assert 0.0 <= a < 360.0, f"Angle {a} out of range for tip ({tx},{ty})"


# ===========================================================================
# gauge_mapping
# ===========================================================================

class TestAngleToPercent:
    def test_at_empty(self):
        assert angle_to_percent(30, 30, 150) == 0.0

    def test_at_full(self):
        assert angle_to_percent(150, 30, 150) == 100.0

    def test_midpoint(self):
        assert abs(angle_to_percent(90, 30, 150) - 50.0) < 0.01

    def test_clamped_below(self):
        assert angle_to_percent(0, 30, 150) == 0.0

    def test_clamped_above(self):
        assert angle_to_percent(200, 30, 150) == 100.0

    def test_zero_span(self):
        # Should not raise ZeroDivisionError
        assert angle_to_percent(90, 90, 90) == 0.0

    def test_descending_scale(self):
        # Some gauges run FULL=low angle, EMPTY=high angle
        assert angle_to_percent(30, 150, 30) == 100.0
        assert angle_to_percent(150, 150, 30) == 0.0


class TestCalibrateFromAngles:
    def test_raises_on_too_few(self):
        with pytest.raises(CalibrationError, match="at least 10"):
            calibrate_from_angles([45.0] * 5)

    def test_raises_on_too_small_range(self):
        # 10 angles all within a 2-degree band
        with pytest.raises(CalibrationError, match="too small"):
            calibrate_from_angles([90.0 + i * 0.1 for i in range(10)])

    def test_returns_p5_p95(self):
        # A clean linear sweep 0 -> 90 degrees, many samples
        angles = [float(i) for i in range(0, 91)]
        p5, p95 = calibrate_from_angles(angles)
        # p5 should be near 0, p95 near 90 (not exact due to percentile)
        assert p5 < 10.0
        assert p95 > 80.0


class TestGaugeScaleLinear:
    def _make_cfg(self, empty=30.0, full=150.0, tick_map=None):
        return GaugeConfig(
            pivot=(400, 300), dial_radius=200,
            empty_angle=empty, full_angle=full,
            tick_map=tick_map or [],
        )

    def test_linear_at_empty(self):
        s = GaugeScale(self._make_cfg(30, 150))
        val, unit = s.angle_to_value(30)
        assert abs(val - 0.0) < 0.01
        assert unit == "percent"

    def test_linear_at_full(self):
        s = GaugeScale(self._make_cfg(30, 150))
        val, unit = s.angle_to_value(150)
        assert abs(val - 100.0) < 0.01

    def test_to_percent_midpoint(self):
        s = GaugeScale(self._make_cfg(30, 150))
        assert abs(s.to_percent(90) - 50.0) < 0.01


class TestGaugeScaleOCR:
    def _make_cfg_with_ticks(self):
        ticks = [(30.0, 0.0), (90.0, 50.0), (150.0, 100.0)]
        return GaugeConfig(
            pivot=(400, 300), dial_radius=200,
            empty_angle=30.0, full_angle=150.0,
            tick_map=ticks,
        )

    def test_ocr_at_known_tick(self):
        s = GaugeScale(self._make_cfg_with_ticks())
        val, unit = s.angle_to_value(90.0)
        assert abs(val - 50.0) < 0.01
        assert unit == "physical"

    def test_ocr_interpolates(self):
        s = GaugeScale(self._make_cfg_with_ticks())
        val, unit = s.angle_to_value(60.0)
        assert abs(val - 25.0) < 0.01
        assert unit == "physical"

    def test_ocr_clamps_below(self):
        s = GaugeScale(self._make_cfg_with_ticks())
        val, unit = s.angle_to_value(0.0)
        assert abs(val - 0.0) < 0.01

    def test_ocr_clamps_above(self):
        s = GaugeScale(self._make_cfg_with_ticks())
        val, unit = s.angle_to_value(200.0)
        assert abs(val - 100.0) < 0.01

    def test_single_tick_falls_back_to_percent(self):
        # One tick is not enough for interpolation; must fall back to linear %
        ticks = [(90.0, 50.0)]
        cfg = GaugeConfig(
            pivot=(400, 300), dial_radius=200,
            empty_angle=30.0, full_angle=150.0,
            tick_map=ticks,
        )
        s = GaugeScale(cfg)
        _, unit = s.angle_to_value(90.0)
        assert unit == "percent"


# ===========================================================================
# config
# ===========================================================================

class TestGaugeConfig:
    def test_any_fallback_false_by_default(self):
        cfg = GaugeConfig(pivot=(400, 300), dial_radius=200,
                          empty_angle=30.0, full_angle=150.0)
        assert cfg.any_fallback is False

    def test_any_fallback_true_when_radius_fallback(self):
        cfg = GaugeConfig(pivot=(400, 300), dial_radius=200,
                          empty_angle=30.0, full_angle=150.0,
                          radius_is_fallback=True)
        assert cfg.any_fallback is True

    def test_describe_includes_fallback_tag(self):
        cfg = GaugeConfig(pivot=(400, 300), dial_radius=200,
                          empty_angle=30.0, full_angle=150.0,
                          pivot_is_fallback=True)
        assert "[FALLBACK]" in cfg.describe()