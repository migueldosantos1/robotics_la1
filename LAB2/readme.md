# Robótica - Projeto de Laboratório 2

### Scripts Python

- `main_script.py` - Ficheiro de sessão de laboratório, usado para correr a trajetória
  - Flag `--traj-only`, corre a trajetória, ignorando a validação (melhor opção para visualizar apenas o movimento)
  - Flag `--analyze`, apenas analisa o ficheiro ik_data.npz, obtido durante a execução da trajetória
  - Sem flags, corre uma validação seguida da trajetória
- `trajectory_def.py` - Definição da trajetória da Tarefa 2; a função traj(t) devolve a pose desejada no instante t.
- `trajectory_exec.py` - Responsável pela execução da trajetória e registo dos logs.
- `matrices.py` - Implementação da cinemática direta do Lite 6 desenvolvida pelo grupo
- `data_4_X.py` - Responsável pelos cálculos e plots utilizados na secção 4.X
- `plot_ik_joints.py` - Elabora os plots dos ângulos das juntas devolvidos pela IK durante o teste estático
- `plot_joints.py` - Elabora os plots dos ângulos das juntas obtidos durante a trajetória executada pelo robô

### Outros ficheiros (todos obtidos no `main_script.py`):

- `ik_data.npz`: Contém ts (instantes), poses (poses desejadas), q_ik (juntas da IK, graus) e codes (código de retorno da IK por ponto).
- `ik_errors.npz`: Contém ts, desired, calc (DK sobre as juntas da IK) e err (calc − desired, mm e graus).
- `traj_run_slow1.npz`: Contém ts e desired (os 20 waypoints enviados) e log (registo da execução, uma linha por amostra).
- `fk_validation.npy`: Contém Uma linha por configuração, com colunas q(6); pose do robô(6); pose da DK(6); diferença(6).

Unidades: posições em mm e ângulos em graus.

Dependências: `numpy`, `scipy`, `matplotlib` e `xarm-python-sdk`.

Correr os scripts a partir da pasta principal, dado que os scripts usam os ficheiros `.npz`