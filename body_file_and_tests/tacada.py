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
ANGULO_FLEXAO_COTOVELO = math.radians(-60)
START = 3 # Tempo de espera até se começar a mover
DELAY_INICIAL = 3 # Tempo para o robô se posicionar
TEMPO_ESPERA = 1 # Esperar um pouco antes da tacada
DURACAO_SWING = 4 # Tempo que a tacada demora
DELAY_FINAL = DELAY_INICIAL # Tempo para regressar à posição inicial

# Atuadores:
act_rot_hip_z = model.actuator("act_rot_hip_z").id
act_rot_hip_y = model.actuator("act_rot_hip_y").id
act_rot_hip_x = model.actuator("act_rot_hip_x").id
act_left_arm_x = model.actuator("act_left_arm_x").id
act_left_arm_y = model.actuator("act_left_arm_y").id
act_left_elbow = model.actuator("act_left_elbow").id
act_right_arm_x = model.actuator("act_right_arm_x").id
act_right_arm_y = model.actuator("act_right_arm_y").id
act_right_elbow = model.actuator("act_right_elbow").id
act_wrist = model.actuator("act_wrist").id

# Posições das juntas:
adr_rot_hip_z = int(model.joint("rot_hip_z").qposadr[0])
adr_rot_hip_y = int(model.joint("rot_hip_y").qposadr[0])
adr_rot_hip_x = int(model.joint("rot_hip_x").qposadr[0])
adr_left_arm_x = int(model.joint("left_arm_x").qposadr[0])
adr_left_arm_y = int(model.joint("left_arm_y").qposadr[0])
adr_left_elbow = int(model.joint("left_elbow").qposadr[0])
adr_right_arm_x = int(model.joint("right_arm_x").qposadr[0])
adr_right_arm_y = int(model.joint("right_arm_y").qposadr[0])
adr_right_elbow = int(model.joint("right_elbow").qposadr[0])
adr_wrist = int(model.joint("wrist").qposadr[0])

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        inicio = time.time()
        t = data.time

        # ETAPA ZERO
        if t < 0:
            zero = 0
        elif t < START:
            zero = math.sin(math.pi * (t / START) / 2)
        else:
            zero = 1

        # PRIMEIRA ETAPA - PREPARAÇÃO
        # O robô está inicialmente com o tronco um pouco inclinado para a frente, e com os braços em baixo
        # Ao longo desta etapa inicial, ele coloca-se lentamente em posição para efetuar o swing
        # Esta preparação é feita com uma função cosseno, que começa em 0 e se move suavemente para 1
        if t < START:
            prep = 0
        elif t < DELAY_INICIAL + START:
            t_t0 = t - START
            prep = math.sin(math.pi * (t_t0 / DELAY_INICIAL) / 2)
        else:
            prep = 1

        # SEGUNDA ETAPA - TEMPO DE ESPERA
        # O robô fica simplesmente à espera durante um certo período de tempo antes de efetuar o swing

        # TERCEIRA ETAPA - SWING
        # O robô executa o swing, que está descrito em detalhe no final
        t0 = START + DELAY_INICIAL + TEMPO_ESPERA
        if t < t0:
            onda_swing = 0
            onda_swing_sec = 0
        elif t < t0 + DURACAO_SWING:
            t_t0 = t - t0
            onda_swing = math.sin(math.pi * (t_t0 / DURACAO_SWING) / 2) # Movimento principal do swing (tronco)
            onda_swing_sec = math.sin(math.pi * (t_t0 / DURACAO_SWING)) # Movimento secundário do swing (antebraços)
        else:
            onda_swing = 1
            onda_swing_sec = 0

        # QUARTA ETAPA - VOLTA AO 0
        # O robô volta para a posição inicial
        t1 = START + DELAY_INICIAL + TEMPO_ESPERA + DURACAO_SWING + TEMPO_ESPERA
        if t < t1:
            finish = 0
        elif t < t1 + DELAY_FINAL:
            t_t0 = t - t1
            finish = math.sin(math.pi * (t_t0 / DELAY_FINAL) / 2)
        else:
            finish = 1

        # Fator multiplicativo de retorno à posição neutra (1 durante o swing, reduz até 0 no finish)
        fator_posicao = 1 - finish

        # Os sinais dos atuadores são calculados através da combinação do valor estático inicial, da onda correspondente
        #  à etapa de preparação, e das ondas correspondentes ao movimento principal. 
        # data.ctrl[act_rot_hip_z] = (-math.radians(90) * prep + math.radians(180) * onda_swing) * fator_posicao
        # data.ctrl[act_rot_hip_y] = (math.radians(15) * onda_swing_sec) * fator_posicao
        # data.ctrl[act_rot_hip_x] = (math.radians(20) * onda_swing) * fator_posicao
        # data.ctrl[act_left_arm_x] = -ANGULO_BRACOS * fator_posicao
        # data.ctrl[act_left_arm_y] = (-math.radians(75) * prep + math.radians(35) * onda_swing_sec) * fator_posicao
        # data.ctrl[act_right_arm_x] = ANGULO_BRACOS * fator_posicao
        # data.ctrl[act_right_arm_y] = (-math.radians(75) * prep + math.radians(35) * onda_swing_sec) * fator_posicao
        # data.ctrl[act_wrist] = ( (-math.radians(20) * prep) +  math.radians(-40)*onda_swing) * fator_posicao
        # data.ctrl[act_left_elbow] = (ANGULO_FLEXAO_COTOVELO * prep ) * fator_posicao
        # data.ctrl[act_right_elbow] = (ANGULO_FLEXAO_COTOVELO * prep ) * fator_posicao

        data.ctrl[act_rot_hip_z] = (-math.radians(90) * prep + math.radians(180) * onda_swing) * fator_posicao
        data.ctrl[act_rot_hip_y] = math.radians(40)*zero + (math.radians(20) * onda_swing) * fator_posicao
        data.ctrl[act_rot_hip_x] = 0
        data.ctrl[act_left_arm_x] = 0
        data.ctrl[act_left_arm_y] = 0
        data.ctrl[act_right_arm_x] = 0
        data.ctrl[act_right_arm_y] = 0
        data.ctrl[act_wrist] = 0
        data.ctrl[act_left_elbow] = 0
        data.ctrl[act_right_elbow] = 0




        mujoco.mj_step(model, data)
        v.sync()

        # Mantém o tempo real (o programa pausa durante o tempo que sobra após os cálculos para esse passo)
        resto = model.opt.timestep - (time.time() - inicio)
        if resto > 0:
            time.sleep(resto)


        # Descrição completa de cada articulação durante o movimento (tudo em graus):

        # Rotação do tronco (segundo o eixo vertical) - começa a preparação em 0 e termina em -60; começa o swing em -60 e termina em 60

        # Inclinação do tronco para a frente ou para trás - durante a preparação mantém-se em 20 graus (inclinação para a frente);
        #  durante o swing aumenta desde 20 até 55 (quando bate na bola), regressando a 20 graus no final do movimento

        # Inclinação lateral do tronco - nula até ao início do swing; começa o swing em 0 e termina em 20 graus, inclinado para a sua direita (X positivo)

        # Os braços mantêm o mesmo ãngulo entre si durante todo o movimento
        # Quanto ao ângulo que fazem com a vertical: começa a preparação em -45 e termina em -120, ou seja, 60 graus com a vertical positiva;
        #  começa o swing em 60, desce até aos 25 (quando bate na bola), e volta a subir até aos 60

        # Quando ao taco, este começa alinhado com o eixo X (de frente para o jogardor), terminando a preparação com um ângulo de 100 graus
        #  em relação a esse mesmo eixo; começa o swing em 100 e termina em -50, ou seja, descreve um arco de 150 durante o movimento
        #  (é de notar que o pulso tem sempre o fator corretivo associado ao ângulo entre os braços, já que este é descrito como estando segundo
        #  o eixo do braço direito - como tal, para colocá-lo segundo o eixo X, é necessário subtrair esse mesmo ângulo, como verificado nos cálculos)