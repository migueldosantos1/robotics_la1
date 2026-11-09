
#### Intro

**Direct Kinematics** - a partir de um conjunto de ângulos das juntas, obtém-se uma pose
- implementada pela função das matrizes `tip_position`
- Serve para construir a pose real a partir dos ângulos das juntas fornecidos
- Serve para validar o modelo na função de validação

**Inverse Kinematics** - a partir de uma pose, obtém-se um conjunto de ângulos de juntas
- Usa-se a IK do robô
- Serve para converter a trajetória da Task 2 em ângulos de juntas

Processo:
1. Pose desejada
2. IK do robô
3. Ângulos de juntas
4. DK das matrizes
5. Pose calculada


#### `extrair_data.py`

`extrair_data.py` 
- Calcula o erro entre a pose desejada e a pose calculada pela nossa DK a partir da IK do robô
- Mede a consistência entre DK e IK

Resultados da `extrair_data.py`
- Os erros são baixos, da ordem de 1.5 mm - é esta a comparação pedida na Task 4


#### `analysis.py`

`analysis.py`
- Calcula o erro entre a pose desejada e a pose efetivamente alcançada pelo robô durante o movimento (aplica a DK aos ângulos das juntas reportados no log do robô)
- Mede a precisão do seguimento de trajetória por parte do robô
- O papel da IK é interno ao robô

Resultados da `analysis.py`
- O primeiro gráfico mostra que a trajetória desejada foi seguida com uma precisão elevada (mais tarde é calculado como sendo da ordem de 1 mm, e como tal o seguimento é tão bom quanto o modelo permite medir), mas que o robô não chegou ao final, tendo ficado aproximadamente 17 mm atrás do ponto final da trajetória desejada
- O erro de yaw e o erro de posição variam de forma semelhante
- Os dois pontos anteriores podem indicar que existe algum lag entre a posição desejada e a posição atingida pelo robô (quanto ao segundo ponto, se o robô estiver atrasado, os valores medidos são aqueles de um ponto anterior) - por isso adicionou-se um bloco de código novo
- No bloco novo, demonstra-se que provavelmente existe um atraso na ordem dos 0.3 segundos, e após compensar esse atraso o valor do erro desce consideravelmente, para 6.47 mm (este cálculo é apenas uma estimativa para ver o que acontece, então é normal que o resultado não seja perfeito)

O erro entre a posição desejada e a posição calculada (1.549 mm) e o erro entre a DK e a posição interna do robô (1.49 mm) são muito semelhantes - poderá indicar que o erro principal vem do modelo, e que a IK do robô é ainda mais precisa do que o estimado
