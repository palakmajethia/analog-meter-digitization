"""
GaugeConfig: single calibration state object threaded through the pipeline.

Replaces the old pattern of passing (PIVOT, DIAL_RADIUS, EMPTY_ANGLE,
FULL_ANGLE, TICK_MAP, SCALE) as 5-6 separate positional arguments into
every function call. Having one object means:
  - a failed calibration can be represented as CalibrationError rather than
    silently substituting wrong-but-plausible numbers;
  - adding a new calibration field doesn't require touching every call site.
"""
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


class CalibrationError(RuntimeError):
    """Raised when calibration cannot produce a usable result."""


@dataclass
class GaugeConfig:
    # Geometry
    pivot: Tuple[float, float]
    dial_radius: int
    empty_angle: float
    full_angle: float

    # Scale (populated after OCR pass; None = use linear percent fallback)
    tick_map: List[Tuple[float, float]] = field(default_factory=list)

    # Calibration quality flags: True = we used a hardcoded default,
    # False = was actually detected. Exposed so callers can warn the user.
    pivot_is_fallback: bool = False
    radius_is_fallback: bool = False
    angles_are_fallback: bool = False

    @property
    def any_fallback(self) -> bool:
        return self.pivot_is_fallback or self.radius_is_fallback or self.angles_are_fallback

    def describe(self) -> str:
        lines = [
            f"  Pivot        : {self.pivot[0]:.1f}, {self.pivot[1]:.1f}"
            + (" [FALLBACK]" if self.pivot_is_fallback else ""),
            f"  Dial radius  : {self.dial_radius}px"
            + (" [FALLBACK]" if self.radius_is_fallback else ""),
            f"  Empty angle  : {self.empty_angle:.2f}°",
            f"  Full angle   : {self.full_angle:.2f}°"
            + (" [FALLBACK]" if self.angles_are_fallback else ""),
            f"  OCR ticks    : {len(self.tick_map)} labeled tick marks found",
        ]
        return "\n".join(lines)