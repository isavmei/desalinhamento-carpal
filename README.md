# Detecção de desalinhamento carpal em radiografia PA de punho com CNN de pontos anatômicos

Isabela de Avellar Meireles, Universidade Federal Fluminense (UFF)

## Motivação/Objetivo

A dissociação escafolunar e outras instabilidades do carpo costumam passar despercebidas na radiografia inicial do punho, porque o sinal é uma pequena alteração de posição entre ossos e não uma linha de fratura visível. Quando o diagnóstico atrasa, a instabilidade evolui para artrose degenerativa do punho.

O objetivo do projeto é medir automaticamente o alinhamento do carpo numa radiografia PA. Em vez de treinar um classificador direto (normal x anormal), que exigiria centenas de exemplos positivos rotulados que não existem em bases públicas, uma CNN aprende a localizar 11 pontos anatômicos. A partir desses pontos são calculadas medidas geométricas, e cada medida é comparada com um corte obtido numa referência de punhos normais. Assim o modelo pode ser treinado majoritariamente com imagens normais, e a anormalidade fica definida por geometria, de forma interpretável.

## Descrição do Dataset

### Descrição dos dados

Foram usadas duas bases públicas de radiografia de punho.

**GRAZPEDWRI-DX** (Nagy et al., 2022, Scientific Data): 20.327 radiografias de punho de trauma do Hospital Universitário de Graz, com metadados por imagem (projeção, idade, lateralidade, gesso, material metálico, espaçamento de pixel). Foram elegíveis as imagens em projeção PA, idade a partir de 16 anos (esqueleto próximo do adulto), sem gesso e sem metal: 578 imagens. Delas foram sorteadas 237 de pacientes distintos.

**MURA v1** (Rajpurkar et al., 2017, Stanford): 10.415 imagens de punho rotuladas por estudo como normal ou anormal. Como a base mistura incidências, as imagens foram triadas visualmente e 65 radiografias PA foram incorporadas (38 normais, 27 anormais pelo rótulo MURA), aumentando a variedade de equipamento e contraste. A MURA não informa espaçamento de pixel, então medidas em milímetros só existem para as imagens GRAZ.

**Casos positivos**: 4 casos publicados de dissociação escafolunar (relatos de caso e material didático), lidos mais de uma vez, mais 1 caso medido externamente, totalizando 5 positivos. As imagens desses casos não são redistribuídas no repositório, apenas as coordenadas marcadas.

**Anotação**: todas as imagens foram marcadas manualmente pela autora com uma ferramenta de clique feita no Colab. Os 11 pontos são:

| # | Ponto | Uso |
|---|---|---|
| 1 | escafoide, borda ulnar (meio da interlinha escafolunar) | intervalo escafolunar |
| 2 | semilunar, borda radial (meio da interlinha escafolunar) | intervalo escafolunar |
| 3, 4 | capitato, extremidades proximal e distal | eixo e comprimento do capitato |
| 5 | base do 3º metacarpo | índice de Pires |
| 6 | superfície articular distal do rádio | índice de Pires |
| 7, 8 | escafoide, cantos proximais radial e ulnar | arco de Gilula |
| 9, 10 | semilunar, cantos proximais radial e ulnar | arco de Gilula |
| 11 | piramidal, canto proximal radial | arco de Gilula |

Para medir a variabilidade da própria marcação, 30 imagens foram marcadas uma segunda vez, às cegas, com nomes trocados.

### Etapas de preparação dos dados ou pré-processamento

1. Filtro de elegibilidade na GRAZPEDWRI-DX (PA, idade ≥ 16, sem gesso, sem metal) e triagem visual de incidência PA na MURA.
2. Normalização de intensidade por percentis: valores entre os percentis 1 e 99 levados para 0 a 255 em 8 bits. Isso uniformiza imagens de 16 bits e de contrastes diferentes.
3. Espelhamento horizontal dos punhos esquerdos, para que todas as imagens tenham a mesma orientação.
4. Exclusão de 4 imagens com erro de marcação, detectadas por filtro de plausibilidade (índice de Pires fora de 1,2 a 1,9 ou relação escafolunar fora de 0,01 a 0,20). Restaram 297 imagens (233 GRAZ, 64 MURA).
5. Redimensionamento para 384 x 384 em tons de cinza, valores em [0, 1], com as coordenadas dos pontos reescaladas na mesma proporção.
6. Geração dos alvos: um mapa de calor gaussiano 144 x 144 por ponto (σ = 3 px).
7. Aumento de dados, somente no treino e a cada época: rotação de ±12°, escala de 0,70 a 1,35, translação de ±8%, ganho de 0,82 a 1,18, brilho de ±18 níveis e gama de 0,80 a 1,25. As transformações geométricas são aplicadas também aos pontos, e são descartadas se algum ponto sair da imagem.

### Link para a versão disponibilizada do dataset

Bases originais: [GRAZPEDWRI-DX no Kaggle](https://www.kaggle.com/datasets/jasonroggy/grazpedwri-dx) e [MURA v1 no Kaggle](https://www.kaggle.com/datasets/tommyngx/mura-v1).

## Modelo de aprendizado de máquina

### Formalização matemática

**Entrada e saída.** Dada uma imagem $x \in [0,1]^{384 \times 384}$, a rede $f_\theta$ produz 11 mapas $\hat{H} = f_\theta(x) \in [0,1]^{144 \times 144 \times 11}$, um por ponto anatômico.

**Arquitetura.** O encoder é a MobileNetV2 pré-treinada na ImageNet, cortada na camada `block_13_expand_relu` (passo 16, saída 24 x 24 x 576). A imagem em cinza é replicada em 3 canais. O decoder tem dois blocos de upsampling 2x seguidos de convolução 3x3 com ReLU (128 e 64 filtros) e normalização em lote, um redimensionamento para 144 x 144 e uma convolução 3x3 final com 11 filtros e ativação sigmoide. Total de 1,36 milhão de parâmetros.

**Alvo.** Para o ponto $k$ com coordenadas $(x_k, y_k)$ na escala do mapa:

$$H_k(u,v) = \exp\left(-\frac{(u-x_k)^2 + (v-y_k)^2}{2\sigma^2}\right), \quad \sigma = 3$$

**Função de perda.** Erro quadrático médio entre mapas previstos e alvos:

$$\mathcal{L}(\theta) = \frac{1}{N \cdot 144^2 \cdot 11} \sum_{i=1}^{N} \sum_{u,v,k} \left(\hat{H}_{i,k}(u,v) - H_{i,k}(u,v)\right)^2$$

**Decodificação.** Para cada mapa, toma-se o pico $(u^*, v^*) = \arg\max \hat{H}_k$ e refina-se a posição pelo centróide ponderado numa janela 5 x 5 em torno dele. O valor do pico $c_k = \hat{H}_k(u^*, v^*)$ é usado como confiança, e medidas cujos pontos têm $c_k < 0{,}20$ são rejeitadas.

**Medidas geométricas.** Com os pontos $p_1, \dots, p_{11}$, o eixo do capitato $\hat{u} = (p_4 - p_3)/L$, de comprimento $L = \lVert p_4 - p_3 \rVert$, serve de régua interna, o que torna as medidas independentes da escala da imagem:

- relação escafolunar: $r_{SL} = \lVert p_1 - p_2 \rVert / L$
- índice de Pires (altura carpal sobre comprimento do capitato): $I_P = |(p_5 - p_6) \cdot \hat{u}| / L$
- arco de Gilula: um círculo de centro $c$ e raio $R$ é ajustado por mínimos quadrados aos pontos 7 a 11, e $d_j = \lVert p_j - c \rVert$. Daí saem o resíduo do arco $\max_j |d_j - R| / R$, o degrau escafolunar $|d_8 - d_9| / R$ e o degrau lunopiramidal $|d_{10} - d_{11}| / R$.

**Regra de decisão.** Uma medida é considerada anormal quando ultrapassa o percentil 95 da mesma medida nos 233 punhos GRAZ de referência (cortes: $r_{SL} > 0{,}1404$, resíduo > 0,0398, degrau SL > 0,0597, degrau LT > 0,0242). Para o índice de Pires é usada a faixa de 1,46 a 1,67.

### Método de validação

Divisão hold-out 80/20 **por paciente**, estratificada pela origem (GRAZ ou MURA), com semente fixa. Nenhum paciente aparece nos dois conjuntos. O conjunto de validação foi usado no early stopping (paciência de 25 épocas sobre a perda de validação), portanto participou da seleção do modelo.

O treino tem duas fases: na primeira o encoder fica congelado (Adam, taxa 1e-3, até 200 épocas); na segunda as últimas 30 camadas do encoder são liberadas (Adam, taxa 1e-5, até 80 épocas). Lote de 4 imagens.

A avaliação é feita em três níveis: (1) erro de localização de cada ponto; (2) concordância entre as medidas calculadas com pontos do modelo e com pontos manuais, comparada com a concordância da própria anotadora ao remarcar as mesmas imagens; (3) desempenho preliminar da regra de decisão nos casos positivos publicados contra a referência normal.

### Medidas de desempenho

- Erro de localização: distância euclidiana entre ponto previsto e ponto manual, em pixels do mapa 144 (mediana e percentil 90).
- Erro de cada medida geométrica: erro absoluto mediano e percentil 90, viés (média de modelo menos manual) e limites de concordância de Bland-Altman (viés ± 1,96 desvio padrão).
- Razão entre o erro do modelo e o erro intra-observador (p90). Valor perto de 1 indica que o modelo erra tanto quanto a anotadora ao remarcar a mesma imagem.
- Coeficiente de correlação intraclasse ICC(1,1) da remarcação.
- Concordância de classificação (normal x anormal) entre modelo e anotação manual: concordância percentual e kappa de Cohen.
- Sensibilidade e especificidade da regra de decisão, com intervalo de confiança exato de Clopper-Pearson para a sensibilidade.

## Avaliação

### Amostras usadas para treinamento, validação e teste

| Conjunto | GRAZ | MURA | Total | Uso |
|---|---|---|---|---|
| Treino | 184 | 51 | 235 | ajuste dos pesos |
| Validação | 49 | 13 | 62 | early stopping e todas as métricas do modelo |
| Referência normal | 233 | | 233 | cálculo dos cortes (pontos manuais) |
| Teste de detecção | | | 5 positivos + 233 normais | sensibilidade e especificidade preliminares |
| Repetibilidade | 30 | | 30 | erro intra-observador |

Não há um conjunto de teste independente para o localizador de pontos: as métricas de localização vêm da validação, que também foi usada no early stopping.

### Medidas de desempenho

**Localização dos pontos (validação, n = 62).** Erro mediano de 1,12 px no mapa 144 (cerca de 3 px na entrada de 384), percentil 90 de 2,57 px. Por origem: GRAZ 1,10 px (p90 2,58), MURA 1,28 px (p90 2,41). O aumento de dados reduziu o erro mediano de 1,34 para 1,14 px e o erro médio absoluto do intervalo escafolunar em mm de 2,85 para 0,80 mm, eliminando falhas grosseiras.

**Medidas geométricas, modelo contra marcação manual (validação).**

| Medida | Aceitas | Erro mediano | Erro p90 | Viés | Corte | p90/corte |
|---|---|---|---|---|---|---|
| índice de Pires | 60/62 | 0,0667 | 0,1819 | -0,0488 | faixa 1,46 a 1,67 | |
| relação SL/capitato | 62/62 | 0,0317 | 0,0679 | -0,0169 | 0,1404 | 0,48 |
| resíduo do arco | 59/62 | 0,0095 | 0,0212 | +0,0046 | 0,0398 | 0,53 |
| degrau escafolunar | 59/62 | 0,0168 | 0,0352 | -0,0028 | 0,0597 | 0,59 |
| degrau lunopiramidal | 59/62 | 0,0093 | 0,0301 | +0,0095 | 0,0242 | 1,25 |

Intervalo escafolunar em mm (GRAZ, n = 49): erro médio absoluto de 0,80 mm, viés de -0,24 mm, limites de concordância de -2,03 a +1,54 mm.

**Modelo comparado com a variabilidade da anotadora (30 imagens remarcadas).**

| Medida | p90 anotadora x anotadora | p90 modelo x anotadora | Razão | ICC da remarcação |
|---|---|---|---|---|
| índice de Pires | 0,0896 | 0,2244 | 2,50 | 0,647 |
| relação SL/capitato | 0,0547 | 0,0679 | 1,24 | 0,776 |
| resíduo do arco | 0,0207 | 0,0227 | 1,10 | 0,209 |
| degrau escafolunar | 0,0350 | 0,0364 | 1,04 | 0,243 |
| degrau lunopiramidal | 0,0158 | 0,0362 | 2,30 | 0,185 |

**Concordância de classificação, modelo contra manual (validação, n = 62).**

| Critério | Manual positivo | Modelo positivo | Concordância | Kappa |
|---|---|---|---|---|
| Pires fora da faixa | 13 | 15 | 67,7% | 0,08 |
| escafolunar alargado | 5 | 0 | 91,9% | 0,00 |
| arco descontínuo | 2 | 5 | 95,2% | 0,55 |
| degrau escafolunar | 3 | 1 | 93,5% | -0,02 |
| degrau lunopiramidal | 4 | 20 | 71,0% | 0,16 |

**Detecção preliminar pela relação SL/capitato (pontos manuais).** Nos 233 normais a mediana é 0,0724, o percentil 95 é 0,1404 e o máximo é 0,1834. Os 5 casos positivos ficaram entre 0,2473 e 0,4348 (mediana 4,2 vezes a dos normais), sem sobreposição com os normais: o menor positivo está 35% acima do maior normal, e mesmo usando a pior leitura de cada caso a margem é de 29%. No corte p95, sensibilidade de 5/5 (100%, IC95% de 48% a 100%) e especificidade de 221/233 (94,8%). No corte pelo máximo dos normais, 5/5 e 233/233. A remarcação dos casos positivos variou mais que a dos normais (diferença mediana de 0,036 contra 0,017).

## Conclusão

A CNN localiza os 11 pontos com erro mediano em torno de 3 px na imagem de entrada, e para a relação escafolunar e as medidas do arco de Gilula o erro do modelo fica próximo do erro da própria anotadora ao remarcar a imagem (razões de 1,04 a 1,24). Isso mostra que a abordagem por pontos anatômicos, treinada quase só com punhos normais, consegue reproduzir medidas clínicas de alinhamento carpal. O aumento de dados geométrico foi decisivo para eliminar as falhas grosseiras.

A relação escafolunar normalizada pelo capitato separou completamente os casos publicados de dissociação escafolunar dos punhos normais quando medida com pontos manuais, o que sustenta o critério geométrico escolhido.

Os resultados ainda não sustentam uso como classificador automático. O modelo subestima levemente a relação escafolunar (viés de -0,017) e não sinalizou nenhuma das 5 imagens que a marcação manual classificou como alargadas na validação; o índice de Pires e o degrau lunopiramidal erram 2,3 a 2,5 vezes mais que a anotadora, e a concordância de classificação é baixa (kappa até 0,55). As medidas do arco têm baixa repetibilidade mesmo entre marcações manuais (ICC de 0,19 a 0,24). A avaliação de detecção usa apenas 5 positivos, vindos de casos publicados (tipicamente evidentes), com pontos manuais e cortes derivados da mesma amostra de normais, de modo que a especificidade no p95 é próxima de 95% por construção. Os normais da GRAZPEDWRI-DX vêm de atendimentos de trauma e são normais apenas presumidamente quanto ao alinhamento carpal. Em teste exploratório, o modelo teve confiança baixa em imagens recortadas de publicações, diferentes das radiografias completas do treino.

Os próximos passos são: reunir casos positivos reais com radiografia completa (base em DICOM do professor) para medir sensibilidade com os pontos do modelo, separar um conjunto de teste independente, obter uma segunda anotação por especialista e ampliar o número de imagens MURA para reduzir a dependência de uma única base.

## Estrutura do repositório

```
README.md                                   esta documentação
demonstracao_desalinhamento_carpal.ipynb    notebook executável (Colab)
```

Para executar, abra o notebook no Google Colab com GPU, coloque a versão anotada do dataset no Google Drive na estrutura descrita na seção 1 do notebook e execute todas as células. O treino leva poucos minutos numa GPU T4.

## Referências

- Nagy, E. et al. A pediatric wrist trauma X-ray dataset (GRAZPEDWRI-DX) for machine learning. Scientific Data, 9, 222, 2022.
- Rajpurkar, P. et al. MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs. arXiv:1712.06957, 2017.
- Sandler, M. et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR, 2018.
- Gilula, L. A. Carpal injuries: analytic approach and case exercises. American Journal of Roentgenology, 133(3), 503-517, 1979.
