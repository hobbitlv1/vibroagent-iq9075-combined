# LUMO demo-window attribution

The two compact demo files are derived from the **LUMO — Leibniz University Test Structure for Monitoring** dataset:

> Stefan Wernitz, Benedikt Hofmeister, Clemens Jonscher, Tanja Grießmann, and Raimund Rolfes (2021), *LUMO — Leibniz University Test Structure for Monitoring*, LUIS, <https://doi.org/10.25835/0027803>.

The source dataset is distributed under the [Creative Commons Attribution 3.0 licence](https://creativecommons.org/licenses/by/3.0/).

## Included derivatives

| Demo target | LUMO condition | Original source | Original SHA-256 | Included derivative | Included SHA-256 |
|---|---|---|---|---|---|
| `target_3` | `DAM4_010` | `SHMTS_202106010009.mat`, 520–530 s | `31a625e2a0327e3d488993234d11b7747cd8d38d56bd646b7b37cc1a3fec3e87` | `10_DAM4_010/target_3_DAM4_010_520_530s.npz` | `cd1e068e1d060210f426f1a0049a97bf359865d7957ef9e06b9c1d647de73402` |
| `target_5` | `DAM6_010` | `SHMTS_202105050003.mat`, 520–530 s | `1f34c2946f24d997fc33e41c7317726a7b4113d1491c1ef2b475d27631dc4d21` | `08_DAM6_010/target_5_DAM6_010_520_530s.npz` | `d2c612767ad51ab711192d661e14a7422c5fbb652b695759dba1895d21b56150` |

Only the synchronized X/Y acceleration channels for the six-panel mapping are retained: reference `accel04/ML4`; targets `accel02/ML2`, `accel05/ML5`, `accel06/ML6`, `accel08/ML8`, and `accel09/ML9`. The files are converted from MATLAB to compressed NumPy containers without resampling. LUMO has no measured Z channel for this panel, so the runtime supplies zeros and an explicit `Z=false` axis mask.

These windows are demonstrations and regression fixtures, not certified structural-safety evidence.
