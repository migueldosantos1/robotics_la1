import mujoco
import mujoco.viewer    

model = mujoco.MjModel.from_xml_path("main_v6.xml")
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        v.sync()

mujoco.mj_forward(model, data)
print("contactos ativos:", data.ncon)
for i in range(data.ncon):
    c = data.contact[i]
    print(model.geom(c.geom1).name, "colide com", model.geom(c.geom2).name)