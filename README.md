# Simulação de uma tacada de golfe em MuJoCo

## Pacotes necessários

```bash
pip install mujoco numpy
```

## Ficheiros

- **`main_final.xml`** — modelo MuJoCo do jogador, taco e bola.
- **`golf_ball.stl`** — malha 3D da bola, usada pelo modelo.
- - **`dof_demonstracoes.py`** — simula uma breve demonstração dos graus de liberdade que o modelo possui.
- **`tacada_final.py`** — corre a tacada completa, com visualização.
- **`tacada_perturbacoes.py`** — a mesma tacada, com ruído aplicado aos DOF do jogador (Task 2).

## Como correr

```bash
python3 dof_demonstracoes.py
python3 tacada_final.py
python3 tacada_perturbacoes.py
```

Correr a partir desta pasta (os scripts procuram o `main_final.xml`).
