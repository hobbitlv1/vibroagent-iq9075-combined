# Building Vibration Pipeline

## 1. Register sensors

`SensorRegistry` loads `config/sensors.live.yaml` and identifies one reference sensor plus the selected targets. Every sensor ID maps to one location, acquisition folder, component name, and default axis.

## 2. Read a common window

For every selected sensor, the pipeline reads the same requested duration and start position. Live requests require actively updating acquisition data. Saved or replay data must be requested explicitly.

The returned window contains calibrated acceleration, sampling rate, timestamps when available, source metadata, and currentness information.

## 3. Validate data quality

The numerical precheck identifies conditions such as missing samples, too-short windows, non-finite values, flatline behavior, stale data, or failed reads. Quality failures constrain the labels the model may return.

## 4. Extract compact metrics

Each window is converted to building-monitoring metrics:

- acceleration RMS, peak, and peak-to-peak;
- crest factor and kurtosis;
- estimated peak particle velocity;
- dominant frequency;
- relative frequency-band powers;
- sample count, duration, axis, and quality flags.

Raw samples stay inside the process and are not included in model prompts.

## 5. Compare targets with the reference

For every target, the pipeline calculates relative RMS, peak, PPV, dominant-frequency difference, and spectral distance. The interpretation is comparative: it describes how each target differs from the synchronized reference during the same interval.

## 6. Deterministic precheck

Rule-based logic produces a sensor label and confidence from the measured ratios, spectral difference, impulsiveness, and data quality. A second deterministic step proposes the overall network label and affected sensor IDs.

## 7. Bounded model decision

One model request receives only compact metrics, deterministic prechecks, allowed labels, and the expected JSON schema. It returns one report per target plus the fused network assessment.

The result is accepted only when:

- all requested sensor IDs are present exactly once;
- every label belongs to the fixed building vocabulary;
- quality-gated labels are not contradicted;
- numeric fields are finite and valid;
- explanatory text stays inside the building-monitoring domain.

Otherwise, the deterministic precheck is used.

## 8. Present the result

The webchat displays the building-level finding, affected sensors, relative evidence, and a practical monitoring step. The host does not perform an extra free-form model rewrite after a structured tool result is available.

## 9. PSD path

The spectrum interface independently reads the selected registered board and computes:

1. optional zero-phase analysis filter;
2. mean removal;
3. periodic-Hann Welch PSD by default;
4. one-sided density scaling in `g²/Hz`;
5. peak extraction and integrated PSD RMS;
6. plot aggregation for the browser.

A persistent marker shows the primary peak. Pointer movement displays frequency and PSD level at arbitrary positions on the plotted curve.
