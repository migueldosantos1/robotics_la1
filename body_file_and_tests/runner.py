import mujoco
import mujoco.viewer    

model = mujoco.MjModel.from_xml_path("main.xml")
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        v.sync()