"""Baseline-relative retraining trigger for the live engine.

Instead of absolute health thresholds, retraining is judged RELATIVE to how
the currently deployed model was doing when it was deployed:

- when a model version becomes active, its MHS/CDS/QDS at that window are
  frozen as the DeploymentBaseline;
- a window "breaches" when MHS falls more than `health_drop_threshold` below
  the baseline, OR CDS/QDS rises more than `drift_rise_threshold` above it;
- a candidate is generated only when the breach is sustained for
  `sustain_windows` consecutive windows AND at least `cooldown_windows` have
  passed since the last generated candidate;
- approving a candidate resets the baseline to the new version's scores.

Deterministic: same baseline + same window sequence in -> same decisions out.
No LLM involvement (CLAUDE.md) — the `reason` strings are plain templates.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantumguard.config import Config, load_config


@dataclass(frozen=True)
class DeploymentBaseline:
    version: str
    window_id: int
    mhs: float
    cds: float
    qds: float

    @property
    def hybrid(self) -> float:
        return (self.cds + self.qds) / 2.0

    def describe(self) -> str:
        return (
            f"{self.version} deployed at window {self.window_id}, baseline "
            f"health={self.mhs:.2f}, cds={self.cds:.2f}, qds={self.qds:.2f}, "
            f"hybrid={self.hybrid:.2f}"
        )


@dataclass(frozen=True)
class RelativeTriggerDecision:
    """One window's evaluation against the deployment baseline — every input
    that shaped the outcome is recorded so the reasoning log can replay it."""

    window_id: int
    baseline_version: str
    health_drop: float       # baseline.mhs - current mhs (positive = degraded)
    cds_rise: float          # current cds - baseline.cds
    qds_rise: float          # current qds - baseline.qds
    breached: bool           # this window exceeds a relative margin
    sustained_windows: int   # consecutive breach windows, including this one
    cooldown_remaining: int  # windows until a new candidate may be generated
    triggered: bool
    drift_type: str | None           # "sudden" | "gradual" when triggered
    retrain_window_width: int | None
    reason: str


class RelativeRetrainTrigger:
    def __init__(self, config: Config | None = None):
        cfg = config if config is not None else load_config()
        rt = cfg.relative_trigger
        self._health_drop_thr = rt.health_drop_threshold
        self._drift_rise_thr = rt.drift_rise_threshold
        self.sustain_windows = rt.sustain_windows
        self._cooldown = rt.cooldown_windows
        # sudden-vs-gradual classification reuses the trend detector's config
        # so the retrain window widths stay consistent across both triggers
        td = cfg.trend_detector
        self._sudden_drop = td.sudden_drop_threshold
        self._widths = dict(td.retrain_window_width)

        self.baseline: DeploymentBaseline | None = None
        self._streak = 0
        self._last_trigger_window: int | None = None
        self._recent_mhs: list[float] = []

    def set_baseline(self, version: str, window_id: int, components) -> DeploymentBaseline:
        """Freeze the active model's scores as the new reference point; called
        at run start and again on every approval (requirement 3)."""
        self.baseline = DeploymentBaseline(
            version=version,
            window_id=window_id,
            mhs=components.mhs,
            cds=components.cds,
            qds=components.qds,
        )
        self._streak = 0  # a fresh baseline starts with a clean slate
        return self.baseline

    def evaluate(self, window_id: int, components) -> RelativeTriggerDecision:
        if self.baseline is None:
            raise RuntimeError("set_baseline must be called before evaluate")

        self._recent_mhs.append(components.mhs)
        self._recent_mhs = self._recent_mhs[-(self.sustain_windows + 1):]

        # degradation measured RELATIVE to the deployed version's baseline
        health_drop = self.baseline.mhs - components.mhs
        cds_rise = components.cds - self.baseline.cds
        qds_rise = components.qds - self.baseline.qds
        breached = (
            health_drop >= self._health_drop_thr
            or cds_rise >= self._drift_rise_thr
            or qds_rise >= self._drift_rise_thr
        )
        self._streak = self._streak + 1 if breached else 0

        cooldown_remaining = 0
        if self._last_trigger_window is not None:
            cooldown_remaining = max(
                0, self._cooldown - (window_id - self._last_trigger_window)
            )

        triggered = (
            breached and self._streak >= self.sustain_windows and cooldown_remaining == 0
        )

        drift_type = width = None
        if triggered:
            self._last_trigger_window = window_id
            # sharp fall across the sustain span -> sudden (narrow retrain
            # window of very recent data); slow bleed -> gradual (wide window)
            recent_drop = (
                self._recent_mhs[0] - self._recent_mhs[-1]
                if len(self._recent_mhs) > 1 else 0.0
            )
            drift_type = "sudden" if recent_drop >= self._sudden_drop else "gradual"
            width = int(self._widths[drift_type])

        return RelativeTriggerDecision(
            window_id=window_id,
            baseline_version=self.baseline.version,
            health_drop=round(health_drop, 4),
            cds_rise=round(cds_rise, 4),
            qds_rise=round(qds_rise, 4),
            breached=breached,
            sustained_windows=self._streak,
            cooldown_remaining=cooldown_remaining,
            triggered=triggered,
            drift_type=drift_type,
            retrain_window_width=width,
            reason=self._describe(health_drop, cds_rise, qds_rise, breached,
                                  cooldown_remaining, triggered),
        )

    # -- log-line rendering ----------------------------------------------

    def _dominant_signal(self, health_drop: float, cds_rise: float, qds_rise: float) -> str:
        """Name the margin that is furthest past its threshold."""
        signals = [
            (health_drop / self._health_drop_thr,
             f"Health degraded {health_drop:.2f} below {self.baseline.version} baseline"),
            (cds_rise / self._drift_rise_thr,
             f"Classical drift rose {cds_rise:.2f} above {self.baseline.version} baseline"),
            (qds_rise / self._drift_rise_thr,
             f"Quantum drift rose {qds_rise:.2f} above {self.baseline.version} baseline"),
        ]
        return max(signals, key=lambda s: s[0])[1]

    def _describe(self, health_drop, cds_rise, qds_rise, breached,
                  cooldown_remaining, triggered) -> str:
        if not breached:
            return (
                f"within {self.baseline.version} baseline margins "
                f"(ΔMHS {-health_drop:+.2f}, ΔCDS {cds_rise:+.2f}, ΔQDS {qds_rise:+.2f})"
            )
        signal = self._dominant_signal(health_drop, cds_rise, qds_rise)
        if triggered:
            return f"{signal} — threshold breached, cooldown expired — generating retraining candidate."
        if self._streak < self.sustain_windows:
            return f"{signal} ({self._streak}/{self.sustain_windows} sustained windows)."
        return (
            f"{signal} — threshold breached, cooldown active "
            f"({cooldown_remaining} windows remaining) — holding."
        )
