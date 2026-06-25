"""
Alert classification for gauge percent readings.

Fix log (see code review):
- Thresholds used to be duplicated and INCONSISTENT between this module
  (classify_range used <25) and main.py's dashboard arc_color (<20/<40).
  They now live in exactly one place: LOW_ALERT_PERCENT / WARNING_PERCENT
  below. dashboard.py imports them instead of redefining its own numbers.
- classify_range/AlertStateMachine return an AlertStatus enum instead of
  bare strings, so a typo'd status string can't silently fail to match
  anywhere it's compared.
- The old "low_streak" debounce in main.py was asymmetric: N consecutive
  low frames were required to RAISE an alert, but a single normal frame
  immediately cleared it, causing flapping on noisy detections.
  AlertStateMachine now requires N consecutive frames to change tier in
  EITHER direction.
"""
from enum import Enum


class AlertStatus(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    LOW_ALERT = "LOW ALERT"


# Single source of truth for tier boundaries (percent of full angular sweep).
LOW_ALERT_PERCENT = 20.0
WARNING_PERCENT = 40.0


def classify_range(percent: float) -> AlertStatus:
    """Raw (non-debounced) tier classification for a single reading."""
    if percent < LOW_ALERT_PERCENT:
        return AlertStatus.LOW_ALERT
    if percent < WARNING_PERCENT:
        return AlertStatus.WARNING
    return AlertStatus.NORMAL


def get_interval(percent, major_step=20):
    minor_step = major_step / 10.0
    lower = (percent // minor_step) * minor_step
    upper = lower + minor_step
    return round(lower, 2), round(upper, 2)


class AlertStateMachine:
    """
    Debounced alert state with symmetric hysteresis: a tier change (in
    either direction) only takes effect after `streak_needed` consecutive
    frames classify into that tier. This replaces the old one-directional
    `low_streak` counter that could clear an alert on a single noisy frame.
    """

    def __init__(self, streak_needed: int = 5):
        self.streak_needed = streak_needed
        self._current = AlertStatus.NORMAL
        self._candidate = AlertStatus.NORMAL
        self._streak = 0

    def update(self, percent: float) -> AlertStatus:
        raw = classify_range(percent)

        if raw == self._candidate:
            self._streak += 1
        else:
            self._candidate = raw
            self._streak = 1

        if self._streak >= self.streak_needed:
            self._current = raw

        return self._current

    def reset(self):
        self._current = AlertStatus.NORMAL
        self._candidate = AlertStatus.NORMAL
        self._streak = 0