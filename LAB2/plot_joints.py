import sys, os
import numpy as np
import matplotlib.pyplot as plt

# files come from the command line, if no arguments are given, the script uses the default ones
files = sys.argv[1:] or ["traj_run_slow1.npz", "traj_run_slow2.npz", "traj_run_slow5.npz"]

# one subplot per run, stacked vertically
fig, axs = plt.subplots(len(files), 1, figsize=(11, 3.3 * len(files)), squeeze=False)
# all samples of all runs, for the CSV file
rows = []

for ax, f in zip(axs[:, 0], files):
    d = np.load(f)
    log = d["log"]
    t = log[:, 0]
    q_actual = log[:, 7:13]
    t_exp = d["ts"][-1]

    # one curve per joint
    for j in range(6):
        ax.plot(t, q_actual[:, j], label=f"J{j+1}")
        
    # dashed vertical line at the expected end, to compare with when the log really ends
    ax.axvline(t_exp, color="k", linestyle="--", linewidth=1, label=f"Final esperado ({t_exp:.0f} s)")
    # plot legends
    ax.set_title(f"{os.path.basename(f)}")
    ax.set_xlabel("Tempo (s)", fontsize=12)
    ax.set_ylabel("Ângulo (graus)", fontsize=12)
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=12)

    # collect the samples
    for ti, qi in zip(t, q_actual):
        rows.append([os.path.basename(f), ti, *qi])

plt.tight_layout()
plt.savefig("juntas_reais.png", dpi=200)

# export the raw values
np.savetxt("juntas_reais.csv", np.array(rows, dtype=object), delimiter=",", fmt="%s", header="run,t_s,j1_deg,j2_deg,j3_deg,j4_deg,j5_deg,j6_deg", comments="")