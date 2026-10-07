import time
import math
import mujoco
import mujoco.viewer
import os

folder = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(folder, "main_v6.xml")

model = mujoco.MjModel.from_xml_path(file_path)
data = mujoco.MjData(model)

# Ângulo entre os dois braços, com base na geometria do tronco e dos braços
ANGULO_BRACOS = math.asin(3 / 7)

DELAY_INICIAL = 3 # Tempo para o robô se posicionar
TEMPO_ESPERA = 1 # Esperar um pouco antes da tacada
DURACAO = 0.2 # Tempo que a tacada demora

# Atuadores:
act_rot_hip_z = model.actuator("act_rot_hip_z").id
act_rot_hip_y = model.actuator("act_rot_hip_y").id
act_rot_hip_x = model.actuator("act_rot_hip_x").id
act_left_arm_x = model.actuator("act_left_arm_x").id
act_left_arm_y = model.actuator("act_left_arm_y").id
act_right_arm_x = model.actuator("act_right_arm_x").id
act_right_arm_y = model.actuator("act_right_arm_y").id
act_wrist = model.actuator("act_wrist").id

# Posições das juntas:
adr_rot_hip_z = int(model.joint("rot_hip_z").qposadr[0])
adr_rot_hip_y = int(model.joint("rot_hip_y").qposadr[0])
adr_rot_hip_x = int(model.joint("rot_hip_x").qposadr[0])
adr_left_arm_x = int(model.joint("left_arm_x").qposadr[0])
adr_left_arm_y = int(model.joint("left_arm_y").qposadr[0])
adr_right_arm_x = int(model.joint("right_arm_x").qposadr[0])
adr_right_arm_y = int(model.joint("right_arm_y").qposadr[0])
adr_wrist = int(model.joint("wrist").qposadr[0])

# Posição inicial - tronco inclinado para a frente, braços em baixo, a olhar em frente
data.qpos[adr_rot_hip_z] = 0.0
data.qpos[adr_rot_hip_y] = math.radians(10)
data.qpos[adr_rot_hip_x] = 0.0
data.qpos[adr_left_arm_x] = -ANGULO_BRACOS
data.qpos[adr_left_arm_y] = math.radians(-20)
data.qpos[adr_right_arm_x] = ANGULO_BRACOS
data.qpos[adr_right_arm_y] = math.radians(-20)
data.qpos[adr_wrist] = -ANGULO_BRACOS

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        inicio = time.time()
        t = data.time

        # PRIMEIRA ETAPA - PREPARAÇÃO
        # O robô está inicialmente com o tronco um pouco inclinado para a frente, e com os braços em baixo
        # Ao longo desta etapa inicial, ele coloca-se lentamente em posição para efetuar o swing
        # Esta preparação é feita com uma função cosseno, que começa em 0 e se move suavemente para 1
        if t < DELAY_INICIAL:
            prep = 0.5 * (1 - math.cos(math.pi * t / DELAY_INICIAL))
        else:
            prep = 1

        # SEGUNDA ETAPA - TEMPO DE ESPERA
        # O robô fica simplesmente à espera durante um certo período de tempo antes de efetuar o swing
        if t < DELAY_INICIAL + TEMPO_ESPERA:
            fase = 0

        # TERCEIRA ETAPA - SWING
        # O robô executa o swing, que está descrito em detalhe no final
        elif t < DELAY_INICIAL + TEMPO_ESPERA + DURACAO:
            fase = (t - (DELAY_INICIAL + TEMPO_ESPERA)) / DURACAO
        else:
            fase = 1

        onda_1 = math.sin(math.pi * fase / 2) # Função que substitui a sinusoide por ser um pouco mais agressiva
        onda_2 = math.sin(math.pi * fase) # Para movimentos de dois sentidos (subida e descida dos braços)

        # Os sinais dos atuadores são calculados através da combinação do valor estático inicial, da onda correspondente
        #  à etapa de preparação, e das ondas correspondentes ao movimento principal. 
        data.ctrl[act_rot_hip_z] = - math.radians(60)*prep + math.radians(120)*onda_1
        data.ctrl[act_rot_hip_y] = math.radians(10) + math.radians(45)*onda_2
        data.ctrl[act_rot_hip_x] = math.radians(20)*onda_1
        data.ctrl[act_left_arm_x] = -ANGULO_BRACOS
        data.ctrl[act_left_arm_y] = -math.radians(45) -math.radians(75)*prep + math.radians(35)*onda_2
        data.ctrl[act_right_arm_x] = ANGULO_BRACOS 
        data.ctrl[act_right_arm_y] = -math.radians(45) -math.radians(75)*prep+ math.radians(35)*onda_2
        data.ctrl[act_wrist] = -ANGULO_BRACOS + (-math.radians(100)  * prep) + math.radians(150)*onda_1

        mujoco.mj_step(model, data)
        v.sync()

        # Mantém o tempo real (o programa pausa durante o tempo que sobra após os cálculos para esse passo)
        resto = model.opt.timestep - (time.time() - inicio)
        if resto > 0:
            time.sleep(resto)


        # Descrição completa de cada articulação durante o movimento (tudo em graus):

        # Rotação do tronco (segundo o eixo vertical) - começa a preparação em 0 e termina em -60; começa o swing em -60 e termina em 60

        # Inclinação do tronco para a frente ou para trás - durante a preparação mantém-se em 10 graus (inclinação para a frente);
        #  durante o swing aumenta desde 10 até 55 (quando bate na bola), regressando a 10 graus no final do movimento

        # Inclinação lateral do tronco - nula até ao início do swing; começa o swing em 0 e termina em 20 graus, inclinado para a sua direita (X positivo)

        # Os braços mantêm o mesmo ãngulo entre si durante todo o movimento
        # Quando ao ângulo que fazem com a vertical: começa a preparação em -45 e termina em -120, ou seja, 60 graus com a vertical positiva;
        #  começa o swing em 60, desce até aos 25 (quando bate na bola), e volta a subir até aos 60

        # Quando ao taco, este começa alinhado com o eixo X (de frente para o jogardor), terminando a preparação com um ângulo de 100 graus
        #  em relação a esse mesmo eixo; começa o swing em 100 e termina em -50, ou seja, descreve um arco de 150 durante o movimento
        #  (é de notar que o pulso tem sempre o fator corretivo associado ao ângulo entre os braços, já que este é descrito como estando segundo
        #  o eixo do braço direito - como tal, para colocá-lo segundo o eixo X, é necessário subtrair esse mesmo ângulo, como verificado nos cálculos)