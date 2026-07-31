#!/usr/bin/env python3
"""Phase 4 (minimal): synthetic-injection label factory over real normal windows.

Every injector takes a real decimated window (400 Hz, [N,3] or [N] in g, DC
retained) and returns (modified_window, params). Labels are the injection
parameters themselves - the section-2.1 axis values plus one severity - so no
teacher model appears anywhere. Severity is calibrated RELATIVE to the
window's own background (SNR / ratio terms, not absolute g), so the same
recipe stays meaningful across quiet and windy backgrounds and across
domains (section-9 domain-randomization doctrine).

Axis/severity mapping table (the ground-truth definition for the whole
label factory):

  injection            spectral_character  impulsiveness  persistence  amplitude_change
  ------------------   ------------------  -------------  -----------  ----------------
  none (healthy)       unchanged           none           sustained    none
  tone / harmonics     new_tone            none           sustained*   from rms ratio
  decaying sinusoid    unchanged**         single         transient    from rms ratio
  repeated impulses    unchanged**         repeated       recurring    from rms ratio
  broadband noise      broadband_rise      none           sustained*   from rms ratio
  gain                 unchanged           none           sustained    from gain
  spectral shift       shifted             none           sustained    none (energy kept)

  *  partial-window variants become transient
  ** ring-down energy is narrowband but transient; spectral_character describes
     the sustained spectrum, so it stays "unchanged"

  severity: none | mild | significant, from the per-injection strength draw
  (mild and significant parameter ranges are disjoint by construction).

Flat-label projection (for the Phase 0.5 C+ probe, whose referee truth is the
existing SENSOR_LABELS): none -> normal_relative_to_baseline,
mild -> mild_deviation, significant -> significant_local_deviation.

Self-test: run this file on a Phase 2 npz - every injector must measurably
manifest (PSD line present, kurtosis raised, rms ratio hit, dominant band
moved, energy-neutral tone truly energy-neutral) on REAL backgrounds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

FS = 400.0

SEVERITY_TO_FLAT = {
    "none": "normal_relative_to_baseline",
    "mild": "mild_deviation",
    "significant": "significant_local_deviation",
}

# Disjoint strength ranges per severity. Tones/noise in dB relative to the
# background AC rms of the band they land in; gains as plain ratios.
TONE_SNR_DB = {"mild": (3.0, 8.0), "significant": (10.0, 18.0)}
IMPULSE_PEAK_RATIO = {"mild": (3.0, 5.0), "significant": (7.0, 14.0)}
NOISE_RISE_DB = {"mild": (3.0, 6.0), "significant": (8.0, 14.0)}
GAIN_RATIO = {"mild": (1.6, 2.4), "significant": (3.0, 6.0)}
SHIFT_FRACTION = {"mild": (0.03, 0.06), "significant": (0.08, 0.15)}

TONE_HZ = (1.0, 45.0)        # C+ probe is later downsampled to 100 Hz (Nyquist 50)
RING_HZ = (4.0, 40.0)
RING_TAU_S = (0.15, 0.8)


def detrend(w: np.ndarray) -> np.ndarray:
    return w - np.mean(w, axis=0, keepdims=True) if w.ndim == 2 else w - float(np.mean(w))


def ac_rms(w: np.ndarray) -> float:
    x = detrend(np.asarray(w, dtype=np.float64))
    return float(np.sqrt(np.mean(np.square(x))))


def _draw(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    return float(rng.uniform(*lo_hi))


@dataclass
class Injection:
    kind: str
    severity: str
    params: dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)

    @property
    def axes(self) -> dict[str, str]:
        spectral = {
            "none": "unchanged",
            "tone": "new_tone",
            "harmonics": "new_tone",
            "impulse": "unchanged",
            "repeated_impulse": "unchanged",
            "broadband_noise": "broadband_rise",
            "gain": "unchanged",
            "spectral_shift": "shifted",
        }[self.kind]
        impulsiveness = {"impulse": "single", "repeated_impulse": "repeated"}.get(self.kind, "none")
        persistence = {
            "impulse": "transient",
            "repeated_impulse": "recurring",
        }.get(self.kind, "transient" if self.params.get("partial") else "sustained")
        rms_ratio = float(self.params.get("rms_ratio", 1.0))
        amplitude = "none" if rms_ratio < 1.3 else ("mild" if rms_ratio < 2.5 else "large")
        return {
            "amplitude_change": amplitude,
            "spectral_character": spectral,
            "impulsiveness": impulsiveness,
            "persistence": persistence,
        }

    @property
    def flat_label(self) -> str:
        return SEVERITY_TO_FLAT[self.severity]

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "flat_label": self.flat_label,
            "axes": self.axes,
            "params": self.params,
            "quality_flags": self.quality_flags,
        }


def _finish(w0: np.ndarray, out: np.ndarray, kind: str, severity: str, params: dict[str, Any]) -> tuple[np.ndarray, Injection]:
    params = dict(params)
    base = ac_rms(w0)
    params["rms_ratio"] = round(ac_rms(out) / base, 4) if base > 0 else None
    return out.astype(w0.dtype, copy=False), Injection(kind=kind, severity=severity, params=params)


def inject_none(w: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, Injection]:
    return w.copy(), Injection(kind="none", severity="none", params={"rms_ratio": 1.0})


def _time(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / FS


def _add_to_axis(w: np.ndarray, sig: np.ndarray, axis: int) -> np.ndarray:
    out = np.array(w, dtype=np.float64, copy=True)
    if out.ndim == 2:
        out[:, axis] = out[:, axis] + sig
    else:
        out = out + sig
    return out


def _axis_rms(w: np.ndarray, axis: int) -> float:
    x = w[:, axis] if w.ndim == 2 else w
    return ac_rms(x)


def dominant_axis(w: np.ndarray) -> int:
    if w.ndim == 1:
        return 0
    return int(np.argmax([ac_rms(w[:, a]) for a in range(w.shape[1])]))


def inject_tone(
    w: np.ndarray,
    rng: np.random.Generator,
    severity: str,
    *,
    harmonics: bool = False,
    energy_neutral: bool = False,
    partial: bool = False,
) -> tuple[np.ndarray, Injection]:
    """Sustained narrowband tone (machinery-like), optional harmonic set.

    Amplitude set as SNR in dB relative to the target axis background rms.
    energy_neutral rescales the result back to the original axis rms - the
    "same energy, different spectrum" hard negative (still new_tone).
    """
    n = w.shape[0]
    axis = dominant_axis(w)
    f0 = _draw(rng, TONE_HZ)
    snr_db = _draw(rng, TONE_SNR_DB[severity])
    amp = _axis_rms(w, axis) * (10.0 ** (snr_db / 20.0)) * np.sqrt(2.0)
    t = _time(n)
    phase = rng.uniform(0, 2 * np.pi)
    sig = amp * np.sin(2 * np.pi * f0 * t + phase)
    n_harm = 0
    if harmonics:
        n_harm = int(rng.integers(2, 4))
        for k in range(2, 2 + n_harm):
            if k * f0 < TONE_HZ[1]:
                sig += (amp / k) * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
    lo = 0
    if partial:
        seg = int(n * rng.uniform(0.25, 0.5))
        lo = int(rng.integers(0, n - seg))
        mask = np.zeros(n)
        mask[lo : lo + seg] = np.hanning(seg) ** 0.25
        sig = sig * mask
    out = _add_to_axis(w, sig, axis)
    if energy_neutral:
        x0 = detrend(np.asarray(w, np.float64)[:, axis] if w.ndim == 2 else w)
        x1 = detrend(out[:, axis] if out.ndim == 2 else out)
        scale = float(np.sqrt(np.mean(x0**2) / max(np.mean(x1**2), 1e-30)))
        mean1 = np.mean(out[:, axis]) if out.ndim == 2 else np.mean(out)
        if out.ndim == 2:
            out[:, axis] = mean1 + (out[:, axis] - mean1) * scale
        else:
            out = mean1 + (out - mean1) * scale
    return _finish(
        w, out, "harmonics" if harmonics else "tone", severity,
        {"f0_hz": round(f0, 3), "snr_db": round(snr_db, 2), "axis": axis,
         "n_harmonics": n_harm, "energy_neutral": energy_neutral, "partial": partial},
    )


def inject_impulse(
    w: np.ndarray, rng: np.random.Generator, severity: str, *, repeated: bool = False
) -> tuple[np.ndarray, Injection]:
    """Decaying-sinusoid ring-down(s); peak set relative to background rms."""
    n = w.shape[0]
    axis = dominant_axis(w)
    f = _draw(rng, RING_HZ)
    tau = _draw(rng, RING_TAU_S)
    peak_ratio = _draw(rng, IMPULSE_PEAK_RATIO[severity])
    peak = _axis_rms(w, axis) * peak_ratio
    count = int(rng.integers(3, 6)) if repeated else 1
    t = _time(n)
    sig = np.zeros(n)
    starts = np.sort(rng.uniform(0.05, 0.85, size=count)) * (n / FS)
    for t0 in starts:
        dt = t - t0
        env = np.where(dt >= 0, np.exp(-np.maximum(dt, 0) / tau), 0.0)
        sig += peak * env * np.sin(2 * np.pi * f * np.maximum(dt, 0))
    out = _add_to_axis(w, sig, axis)
    return _finish(
        w, out, "repeated_impulse" if repeated else "impulse", severity,
        {"ring_hz": round(f, 3), "tau_s": round(tau, 3), "peak_ratio": round(peak_ratio, 2),
         "count": count, "axis": axis, "start_s": [round(s, 2) for s in starts.tolist()]},
    )


def inject_broadband_noise(
    w: np.ndarray, rng: np.random.Generator, severity: str, *, partial: bool = False
) -> tuple[np.ndarray, Injection]:
    """Band-limited noise floor rise (0.5-45 Hz), level relative to axis rms."""
    from scipy.signal import butter, sosfiltfilt

    n = w.shape[0]
    axis = dominant_axis(w)
    rise_db = _draw(rng, NOISE_RISE_DB[severity])
    target_rms = _axis_rms(w, axis) * (10.0 ** (rise_db / 20.0))
    noise = rng.standard_normal(n)
    sos = butter(4, [0.5 / (FS / 2), 45.0 / (FS / 2)], btype="band", output="sos")
    noise = sosfiltfilt(sos, noise)
    noise *= target_rms / max(float(np.sqrt(np.mean(noise**2))), 1e-30)
    lo = 0
    if partial:
        seg = int(n * rng.uniform(0.25, 0.5))
        lo = int(rng.integers(0, n - seg))
        mask = np.zeros(n)
        mask[lo : lo + seg] = 1.0
        noise = noise * mask
    out = _add_to_axis(w, noise, axis)
    return _finish(
        w, out, "broadband_noise", severity,
        {"rise_db": round(rise_db, 2), "axis": axis, "partial": partial},
    )


def inject_gain(w: np.ndarray, rng: np.random.Generator, severity: str) -> tuple[np.ndarray, Injection]:
    """Multiply the AC part of every axis - same spectrum, different energy."""
    g = _draw(rng, GAIN_RATIO[severity])
    x = np.asarray(w, dtype=np.float64)
    mean = np.mean(x, axis=0, keepdims=True) if x.ndim == 2 else float(np.mean(x))
    out = mean + (x - mean) * g
    return _finish(w, out, "gain", severity, {"gain": round(g, 3)})


def inject_spectral_shift(w: np.ndarray, rng: np.random.Generator, severity: str) -> tuple[np.ndarray, Injection]:
    """Shift ALL spectral content by a fraction (modal-shift analog): resample
    the AC part in time, energy-preserving, same window length."""
    from scipy.signal import resample

    frac = _draw(rng, SHIFT_FRACTION[severity]) * (1 if rng.uniform() < 0.5 else -1)
    x = np.asarray(w, dtype=np.float64)
    n = x.shape[0]
    mean = np.mean(x, axis=0, keepdims=True) if x.ndim == 2 else float(np.mean(x))
    ac = x - mean
    # FFT-resample to m samples, replay at the original rate: f_new = f * n/m.
    # m = n/(1+frac) gives f_new = f*(1+frac). The resampled signal is periodic
    # with period m, so wrapping to n samples is seam-consistent.
    m = int(round(n / (1 + frac)))
    stretched = resample(ac, m, axis=0)
    out_ac = np.take(stretched, np.arange(n) % m, axis=0)
    rms0 = float(np.sqrt(np.mean(ac**2)))
    rms1 = float(np.sqrt(np.mean(out_ac**2)))
    if rms1 > 0:
        out_ac *= rms0 / rms1
    return _finish(w, mean + out_ac, "spectral_shift", severity, {"shift_fraction": round(frac, 4)})


# ---- sensor-quality corruptions (flags, not deviation labels) ----------------

# Frozen corruption-extent levels, preregistered in NATIVE_VIBRATION_LLM_PLAN.md
# section 12.2 (2026-07-13). "Extent" = affected sample fraction. The run-4
# quota builder and the hash-bound quality challenge MUST draw extents from
# this table only; the legacy extent-free defaults below stay bit-identical
# (including RNG consumption) because the v3 diagnostic replay depends on them.
QUALITY_EXTENTS = {
    "clipping": (0.005, 0.02, 0.05),
    "flatline": (0.10, 0.20, 0.30),
}


def corrupt_clipping(
    w: np.ndarray, rng: np.random.Generator, extent: float | None = None
) -> tuple[np.ndarray, Injection]:
    """Clip at the amplitude quantile that affects ~`extent` of samples.

    For the builder's 1-D float32 windows, clipping is computed in float64
    after mean centering. A sample is realized-affected only when its absolute
    centered value is strictly above the default-linear ``1 - extent``
    quantile; ties at the threshold can therefore make the realized fraction
    smaller than the requested fraction. The result is cast back to the input
    dtype.

    For a 2-D ``[N, C]`` window, centering is per column, while the quantile and
    realized count pool all ``N * C`` scalar components. Thus clipping extent
    is a scalar-component fraction, not a fraction of time rows.

    extent=None reproduces the legacy fixed 98th-percentile behavior exactly
    (affected fraction ~0.02). No RNG draws on either path."""
    x = np.asarray(w, dtype=np.float64)
    q = 0.98 if extent is None else 1.0 - float(extent)
    lim = float(np.quantile(np.abs(detrend(x)), q))
    mean = np.mean(x, axis=0, keepdims=True) if x.ndim == 2 else float(np.mean(x))
    out = mean + np.clip(x - mean, -lim, lim)
    params = {"clip_level_g": round(lim, 6), "rms_ratio": 1.0}
    if extent is not None:
        params["affected_fraction"] = float(extent)
        params["clip_quantile"] = q
        affected = int(np.count_nonzero(np.abs(detrend(x)) > lim))
        denominator = int(x.size)
        params["realized_extent"] = {
            "definition": "strictly_clipped_scalar_samples",
            "numerator": affected,
            "denominator": denominator,
            "fraction": affected / denominator,
        }
    inj = Injection(kind="none", severity="none", params=params)
    inj.quality_flags.append("clipping")
    return out.astype(w.dtype, copy=False), inj


def corrupt_dropout(
    w: np.ndarray, rng: np.random.Generator, extent: float | None = None
) -> tuple[np.ndarray, Injection]:
    """Flatline a contiguous segment of `extent` fraction of the window.

    For the builder's 1-D float32 windows, the configured span is
    ``round(N * extent)`` consecutive samples. Its start is drawn uniformly
    from the high-exclusive interval ``[0, N - span)`` and every sample in
    ``[start, start + span)`` is assigned the original value at ``start``.
    Realized extent is the configured span divided by ``N``, not the fraction
    of bytes that differ: the first held sample is already unchanged and
    repeated input values can reduce the changed-byte fraction further.

    For a 2-D ``[N, C]`` window, the span still counts time rows and every
    component in each selected row is held.

    extent=None reproduces the legacy behavior exactly, INCLUDING its two RNG
    draws (uniform length then start). The extent path consumes exactly one
    draw (start position)."""
    x = np.array(w, dtype=np.float64, copy=True)
    n = x.shape[0]
    seg = int(n * rng.uniform(0.1, 0.3)) if extent is None else int(round(n * float(extent)))
    lo = int(rng.integers(0, n - seg))
    hold = x[lo].copy()
    x[lo : lo + seg] = hold
    params = {"dropout_s": round(seg / FS, 2), "rms_ratio": 1.0}
    if extent is not None:
        params["affected_fraction"] = float(extent)
        params["realized_extent"] = {
            "definition": "configured_flatline_time_rows",
            "numerator": int(seg),
            "denominator": int(n),
            "fraction": seg / n,
        }
    inj = Injection(kind="none", severity="none", params=params)
    inj.quality_flags.append("flatline")
    return x.astype(w.dtype, copy=False), inj


INJECTORS = {
    "none": lambda w, rng, sev: inject_none(w, rng),
    "tone": lambda w, rng, sev: inject_tone(w, rng, sev),
    "harmonics": lambda w, rng, sev: inject_tone(w, rng, sev, harmonics=True),
    "tone_energy_neutral": lambda w, rng, sev: inject_tone(w, rng, sev, energy_neutral=True),
    "tone_partial": lambda w, rng, sev: inject_tone(w, rng, sev, partial=True),
    "impulse": lambda w, rng, sev: inject_impulse(w, rng, sev),
    "repeated_impulse": lambda w, rng, sev: inject_impulse(w, rng, sev, repeated=True),
    "broadband_noise": lambda w, rng, sev: inject_broadband_noise(w, rng, sev),
    "gain": lambda w, rng, sev: inject_gain(w, rng, sev),
    "spectral_shift": lambda w, rng, sev: inject_spectral_shift(w, rng, sev),
}


# ---- self-test on real backgrounds -------------------------------------------

def _psd_line_snr_db(x: np.ndarray, f0: float) -> float:
    from scipy.signal import welch

    f, p = welch(detrend(np.asarray(x, np.float64)), fs=FS, nperseg=1024)
    near = (f > f0 - 1.5) & (f < f0 + 1.5)
    floor = (f > max(0.5, f0 - 12)) & (f < min(FS / 2, f0 + 12)) & ~near
    return 10 * np.log10(np.max(p[near]) / np.median(p[floor]))


def _kurtosis(x: np.ndarray) -> float:
    x = detrend(np.asarray(x, np.float64))
    s = float(np.std(x))
    return float(np.mean((x / max(s, 1e-30)) ** 4))


def self_test(npz_path: str) -> None:
    data = np.load(npz_path)
    windows = data["windows"]
    rng = np.random.default_rng(20260707)
    picks = rng.choice(len(windows), size=min(24, len(windows)), replace=False)
    fails: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  {'PASS' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            fails.append(name)

    w = windows[picks[0]]
    for sev in ("mild", "significant"):
        out, inj = inject_tone(w, np.random.default_rng(1), sev)
        ax = inj.params["axis"]
        snr = _psd_line_snr_db(out[:, ax], inj.params["f0_hz"])
        check(f"tone/{sev}", snr > 8, f"PSD line SNR {snr:.1f} dB at {inj.params['f0_hz']:.1f} Hz")

    out, inj = inject_tone(w, np.random.default_rng(2), "significant", energy_neutral=True)
    r = inj.params["rms_ratio"]
    check("tone energy-neutral", abs(r - 1.0) < 0.02, f"rms_ratio {r}")

    out, inj = inject_impulse(w, np.random.default_rng(3), "significant")
    ax = inj.params["axis"]
    k0, k1 = _kurtosis(w[:, ax]), _kurtosis(out[:, ax])
    check("impulse", k1 > k0 + 2, f"kurtosis {k0:.1f} -> {k1:.1f}")

    out, inj = inject_impulse(w, np.random.default_rng(4), "mild", repeated=True)
    check("repeated impulse", inj.params["count"] >= 3, f"count {inj.params['count']}")

    out, inj = inject_gain(w, np.random.default_rng(5), "significant")
    check("gain", abs(inj.params["rms_ratio"] - inj.params["gain"]) < 0.05,
          f"gain {inj.params['gain']} rms_ratio {inj.params['rms_ratio']}")

    out, inj = inject_broadband_noise(w, np.random.default_rng(6), "significant")
    check("broadband noise", inj.params["rms_ratio"] > 2.0, f"rms_ratio {inj.params['rms_ratio']}")

    # spectral shift must move a known embedded line by exactly (1+frac) on a
    # real background - tests the operator itself, not the background's own
    # (unstable) dominant peak
    from scipy.signal import welch

    moved = 0
    tried = 0
    for i in picks[:12]:
        wi, _ = inject_tone(windows[i], np.random.default_rng(200 + int(i)), "significant")
        ax = dominant_axis(wi)
        out, inj = inject_spectral_shift(wi, np.random.default_rng(100 + int(i)), "significant")
        f, p0 = welch(detrend(wi[:, ax].astype(np.float64)), fs=FS, nperseg=4000)
        _, p1 = welch(detrend(out[:, ax].astype(np.float64)), fs=FS, nperseg=4000)
        sel = f > 1.0
        d0, d1 = f[sel][np.argmax(p0[sel])], f[sel][np.argmax(p1[sel])]
        tried += 1
        expect = d0 * (1 + inj.params["shift_fraction"])
        if abs(d1 - expect) < 0.6:
            moved += 1
    check("spectral shift", moved >= tried * 0.8, f"embedded line tracked shift in {moved}/{tried}")

    out, inj = corrupt_clipping(w, np.random.default_rng(7))
    check("clipping flag", inj.quality_flags == ["clipping"], str(inj.quality_flags))
    out, inj = corrupt_dropout(w, np.random.default_rng(8))
    check("dropout flag", inj.quality_flags == ["flatline"], str(inj.quality_flags))

    ex = inject_tone(w, np.random.default_rng(9), "mild")[1]
    print("  example label json:", json.dumps(ex.to_json(), sort_keys=True)[:200], "...")
    if fails:
        raise SystemExit(f"SELF-TEST FAILED: {fails}")
    print("self-test: all pass")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="injection library self-test")
    parser.add_argument("--npz", required=True, help="Phase 2 windows npz to test against")
    args = parser.parse_args()
    self_test(args.npz)


def randomize_excitation(w: np.ndarray, rng: np.random.Generator,
                         static_db: float = 12.0, ramp_db: float = 6.0) -> np.ndarray:
    """Domain-randomize ambient excitation level (plan section 9).

    Independent draws for reference and targets teach the invariance the C+
    v1 run measurably lacked (trained on one calm morning -> LUMO's 10x wind
    swings read as damage): a static gain up to +-static_db plus a slow
    within-window ramp up to +-ramp_db, applied to the AC part. Injection
    severities are background-relative (SNR), so labels are unaffected.
    """
    x = np.asarray(w, dtype=np.float64)
    mean = np.mean(x, axis=0, keepdims=True) if x.ndim == 2 else float(np.mean(x))
    g0 = 10.0 ** (rng.uniform(-static_db, static_db) / 20.0)
    span = rng.uniform(-ramp_db, ramp_db)
    n = x.shape[0]
    ramp = 10.0 ** ((np.linspace(-span / 2, span / 2, n)) / 20.0)
    if x.ndim == 2:
        ramp = ramp[:, None]
    return (mean + (x - mean) * g0 * ramp).astype(w.dtype, copy=False)
