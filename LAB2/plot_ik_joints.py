import sys
import numpy as np
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "ik_data.npz"
d = np.load(path)

# time in seconds and IK joint angles in degrees, q shape [N, 6]
ts, q = d["ts"], d["q_ik"]

# joint variables over time
fig, ax = plt.subplots(figsize=(10, 4))
for j in range(6):
    ax.plot(ts, q[:, j], label=f"J{j+1}")

# plot legends
ax.set_title(f"Joint variables from built-in IK (N = {len(ts)})")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angles (degrees)")
ax.grid(True)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
plt.tight_layout()
plt.savefig("juntas_ik.png", dpi=200)

# extremes and rates (numerical derivatives) for the limits table
dt = ts[1] - ts[0]
qd = np.gradient(q, dt, axis=0)
qdd = np.gradient(qd, dt, axis=0)
qddd = np.gradient(qdd, dt, axis=0)

# print final information about the joints
print("min  [deg]    ", q.min(0).round(1))
print("max  [deg]    ", q.max(0).round(1))
print("max |vel|  [deg/s]  ", np.abs(qd).max().round(1))
print("max |acc|  [deg/s2] ", np.abs(qdd).max().round(1))
print("max |jerk| [deg/s3] ", np.abs(qddd[2:-2]).max().round(1))