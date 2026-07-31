# Current Project Goal

Build a reliable edge application that compares synchronized vibration measurements across a building using one reference IIS3DWB sensor and multiple target sensors.

The application must:

- preserve a stable mapping between each physical board and its configured location;
- acquire and read common time windows across all selected sensors;
- compute deterministic vibration and spectral metrics locally;
- compare every target with the reference sensor;
- classify only with the fixed building sensor and network labels;
- use one bounded language-model decision for compact structured evidence;
- reject out-of-domain output before it reaches the operator;
- continue with deterministic results when the model is unavailable;
- provide live waveforms, a floating periodic-Hann Welch PSD view, optional analysis filters, persistent peak markers, and pointer-based frequency inspection;
- avoid modifying stored data when filters or display transformations are selected;
- report monitoring evidence without making a certified structural-safety judgment.

Success means that repeated requests consistently use the selected board, the logger remains stable, graph rendering remains responsive, and webchat responses remain strictly within building-vibration relative monitoring.
