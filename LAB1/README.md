# Simulation of a golf swing in MuJoCo

## Necessary libraries

```bash
pip install mujoco numpy
```

## Files

- **`main_final.xml`** — MuJoCo model of the player, club and golf ball.
- **`golf_ball.stl`** — 3D mesh of the golf ball, used by the model.
- **`dof_demonstracoes.py`** — simulates a brief demonstration of the degrees of freedoom of this model.
- **`tacada_final.py`** — visualization model of the golf swing.
- **`tacada_perturbacoes.py`** — same script with sinusoidal perturbation for disturbance comprehension.

## How to Run

```bash
python3 dof_demonstracoes.py
python3 tacada_final.py
python3 tacada_perturbacoes.py
```

Run from this folder (the scripts use `main_final.xml`).
