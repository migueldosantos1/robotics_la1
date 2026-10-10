import time
import math
import mujoco
import mujoco.viewer
import os

folder = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(folder, "main_v7.xml")

model = mujoco.MjModel.from_xml_path(file_path)
data = mujoco.MjData(model)

# Ângulo entre os dois braços, com base na geometria do tronco e dos braços
ANGULO_BRACOS = math.asin(30 / 70)

DELAY_INICIAL = 2 # Tempo para o robô se posicionar
TEMPO_ESPERA = 1 # Esperar um pouco antes da tacada
DURACAO = 0.25 # Tempo que a tacada demora

# Integrador utilizado - IMPLICIT / IMPLICITFAST são os mais adequados (por experiência)
model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT

# Atuadores:
act_rot_hip_z = model.actuator("act_rot_hip_z").id
act_rot_hip_y = model.actuator("act_rot_hip_y").id
act_rot_hip_x = model.actuator("act_rot_hip_x").id
act_left_arm_x = model.actuator("act_left_arm_x").id
act_left_arm_y = model.actuator("act_left_arm_y").id
act_right_arm_x = model.actuator("act_right_arm_x").id
act_right_arm_y = model.actuator("act_right_arm_y").id
act_left_elbow = model.actuator("act_left_elbow").id
act_right_elbow = model.actuator("act_right_elbow").id
act_wrist_y = model.actuator("act_wrist_y").id
act_wrist_x = model.actuator("act_wrist_x").id

# Posições das juntas:
adr_rot_hip_z = int(model.joint("rot_hip_z").qposadr[0])
adr_rot_hip_y = int(model.joint("rot_hip_y").qposadr[0])
adr_rot_hip_x = int(model.joint("rot_hip_x").qposadr[0])
adr_left_arm_x = int(model.joint("left_arm_x").qposadr[0])
adr_left_arm_y = int(model.joint("left_arm_y").qposadr[0])
adr_right_arm_x = int(model.joint("right_arm_x").qposadr[0])
adr_right_arm_y = int(model.joint("right_arm_y").qposadr[0])
adr_left_elbow = int(model.joint("left_elbow").qposadr[0])
adr_right_elbow = int(model.joint("right_elbow").qposadr[0])
adr_wrist_y = int(model.joint("wrist_y").qposadr[0])
adr_wrist_x = int(model.joint("wrist_x").qposadr[0])

# Ligação entre as mãos - configurada para aguentar movimentos bruscos
eq_maos = model.equality("hands_connected").id
model.eq_solref[eq_maos] = [0.001, 1.0]
model.eq_solimp[eq_maos] = [0.99, 0.99, 0.001, 0.5, 2]

# Posição inicial do robô:
# Cabeça e tronco a olhar em frente, ligeiramente inclinados para a frente
# Braços esticados e em baixo, com um ângulo de 45 graus em relação ao eixo vertical
# Pulso a segurar o taco no plano dos braços
def posicao_inicial():
    data.qpos[adr_rot_hip_z] = 0.0
    data.qpos[adr_rot_hip_y] = math.radians(15)
    data.qpos[adr_rot_hip_x] = 0.0
    data.qpos[adr_left_arm_x] = -ANGULO_BRACOS
    data.qpos[adr_left_arm_y] = math.radians(-45)
    data.qpos[adr_right_arm_x] = ANGULO_BRACOS
    data.qpos[adr_right_arm_y] = math.radians(-45)
    data.qpos[adr_left_elbow] = 0.0
    data.qpos[adr_right_elbow] = 0.0
    data.qpos[adr_wrist_y] = 0.0
    data.qpos[adr_wrist_x] = -ANGULO_BRACOS

posicao_inicial()
mujoco.mj_forward(model, data)

t_aux = 0.0

with mujoco.viewer.launch_passive(model, data) as v:

    # Configuração da posição inicial da câmara
    v.cam.type =    mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.lookat[:] = [0, 0, 1]
    v.cam.distance = 7.0
    v.cam.azimuth = -150
    v.cam.elevation = -6

    while v.is_running():
        inicio = time.time()
        t = data.time

        # Ativar a ligação entre as mãos (importante estar dentro do while)
        data.eq_active[eq_maos] = 1

        # Impôr a posição inicial quando a simulação dá reset
        if t < t_aux:
            posicao_inicial()
            t = data.time
        t_aux = t

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
        # Orobô executa o swing, que está descrito em detalhe no final
        elif t < DELAY_INICIAL + TEMPO_ESPERA + DURACAO:
            fase = (t - (DELAY_INICIAL + TEMPO_ESPERA)) / DURACAO
        else:
            fase = 1

        onda_1 = math.sin(math.pi * fase / 2) # Para movimentos de um sentido (ex. rotação do corpo)
        onda_2 = math.sin(math.pi * fase) # Para movimentos de dois sentidos (ex. subida e descida dos braços)

        # Sinais dos atuadores
        data.ctrl[act_rot_hip_z] = - math.radians(75)*prep + math.radians(150)*onda_1
        data.ctrl[act_rot_hip_y] = math.radians(15) + math.radians(35)*onda_2
        data.ctrl[act_rot_hip_x] = math.radians(30)*onda_1
        data.ctrl[act_left_arm_x] = -ANGULO_BRACOS
        data.ctrl[act_left_arm_y] = -math.radians(45) -math.radians(55)*prep + math.radians(15)*onda_2
        data.ctrl[act_right_arm_x] = ANGULO_BRACOS 
        data.ctrl[act_right_arm_y] = -math.radians(45) -math.radians(55)*prep+ math.radians(15)*onda_2  
        data.ctrl[act_right_elbow] = -math.radians(20)*prep + math.radians(20)*onda_2
        data.ctrl[act_wrist_y] = (-math.radians(100) * prep) + math.radians(100)*onda_2
        data.ctrl[act_wrist_x] = -ANGULO_BRACOS

        mujoco.mj_step(model, data)
        v.sync()

        # Mantém o tempo real (o programa pausa durante o tempo que sobra após os cálculos para esse passo)
        resto = model.opt.timestep - (time.time() - inicio)
        if resto >0:
                time.sleep(resto)

        # Descrição completa de cada articulação durante o movimento (tudo em graus):

        # Rotação do tronco (segundo o eixo vertical) - começa a preparação em 0 e termina em -75; começa o swing em -75 e termina em 75

        # Inclinação do tronco para a frente ou para trás - durante a preparação mantém-se em 15 graus (inclinação para a frente);
        #  durante o swing aumenta desde 15 até 50 (quando bate na bola), regressando a 15 graus no final do movimento

        # Inclinação lateral do tronco - nula até ao início do swing; começa o swing em 0 e termina em 30 graus, inclinado para a sua direita (X positivo)

        # O ângulo que os braços fazem com a vertical começa a preparação em -45 e termina em -100, ou seja, 80 graus com a vertical positiva;
        #  começa o swing em 80, desce até aos 65 (quando bate na bola), e volta a subir até aos 80

        # O antebraço direito controla o movimento do ponto de vista da simulação. O braço esquerdo apenas o segue. 
        # Os antebraços começam a preparação alinhados com o resto dos braços e acabam com um ângulo de 20 graus
        # em relação à posição inicial. Começam o swing em 20, voltam a 0 a meio do swing, para bater a bola, e regressam aos 20.

        # Quando ao taco, este começa alinhado com o eixo X (de frente para o jogardor), terminando a preparação com um ângulo de 100 graus
        #  em relação ao seu eixo Y. O movimento dos antebraços também acaba por rodá-lo em X.