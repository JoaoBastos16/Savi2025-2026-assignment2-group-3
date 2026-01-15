# Savi2025-2026-assignment2-group-3

#Tarefa 2
1. Geração de Dataset: generate_dataset.py
Objetivo

O script generate_dataset.py é responsável por gerar o dataset de "Cenas" com os dígitos do MNIST. Ele cria imagens maiores (128x128) e posiciona os dígitos aleatoriamente, com ou sem variação de escala. O dataset é gerado nas versões A, B, C, e D, conforme os requisitos da tarefa.

Como Funciona

Importação de Dependências:
O script usa bibliotecas como os, random, json, cv2 (para manipulação de imagens), torchvision.datasets (para carregar o dataset MNIST), e tqdm (para exibir o progresso).

Função generate_scene:
A função principal que gera uma cena (imagem) com dígitos do MNIST. Ela posiciona os dígitos aleatoriamente na tela, ajustando seu tamanho conforme a versão do dataset. Ela também verifica se há sobreposição entre os dígitos, evitando que eles se sobreponham.

Função generate_dataset:
Essa função coordena a geração do dataset, criando um diretório de saída para cada versão e gerando imagens de treino e teste. As imagens e suas anotações (bounding boxes e rótulos) são salvas em formato PNG e JSON, respectivamente.

Execução:
A função generate_dataset é chamada para cada versão do dataset (A, B, C, D), gerando 60.000 imagens de treino e 10.000 imagens de teste para cada versão.

Visualização e Estatísticas: main_dataset_stats.py
Objetivo

O script main_dataset_stats.py tem como objetivo visualizar uma amostra aleatória de imagens e suas anotações, além de gerar estatísticas sobre o dataset. Ele exibe informações como a distribuição de classes (dígitos de 0 a 9), o número médio de dígitos por imagem, o tamanho médio das caixas de anotação, e histogramas de tamanho das caixas e número de dígitos.

Como Funciona

Função load_sample:
Carrega uma amostra aleatória de 9 imagens e suas anotações (salvas em arquivos JSON). Ele seleciona 9 imagens do diretório de treino (data/scenes_D/train) e carrega suas caixas de anotação.

Função visualize:
Exibe as 9 imagens amostradas em uma grade 3x3. Para cada imagem, as caixas de anotação (bounding boxes) são desenhadas em vermelho.

Função statistics:
Gera estatísticas sobre o dataset:

Número de dígitos por imagem.

Tamanho das caixas (largura).

Distribuição das classes (dígitos de 0 a 9).

Histogramas de número de dígitos por imagem e do tamanho das caixas.

Funções de Geração de Imagens: generate_database.py
Objetivo

O script generate_database.py contém funções de utilidade para gerar as imagens do dataset. Ele cuida de como as imagens são manipuladas, redimensionadas, e como as caixas de anotação são geradas e verificadas para evitar sobreposição.

Como Funciona

Função iou:
Calcula a Interseção sobre a União (IoU) entre duas caixas de delimitadores, que é usada para verificar se há sobreposição entre as caixas.

Função non_overlapping:
Verifica se uma nova caixa de anotação não se sobrepõe às caixas já presentes na imagem.

Função generate_scene:
Gera uma "cena" com dígitos do MNIST dispostos aleatoriamente, usando a função non_overlapping para evitar sobreposição entre os dígitos.

Função generate_dataset:
Gera o dataset de imagens e anotações, salvando as imagens em formato PNG e as anotações em formato JSON.
