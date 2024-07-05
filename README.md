# Feedforward Neural Network

## Что такое Feedforward Neural Network (FNN)?

**FNN** - это тип нейронной сети, где информация проходит через сеть в одном направлении, без циклов или обратной связи. Она состоит из нескольких слоёв:

- **Входной слой:** Получает данные и передаёт их первому скрытому слою.
- **Скрытые слои:** Проводят нелинейные преобразования данных, извлекая более сложные признаки.
- **Выходной слой:** Выдаёт результат обработки данных.

## Как работает FNN?

1. **Входные данные:** Вводятся во входной слой.
2. **Передача данных:** Информация передаётся через каждый слой сети.
3. **Нелинейные преобразования:** В каждом нейроне каждого скрытого слоя применяются нелинейные функции активации, например, ReLU или сигмоида. 
4. **Весовые коэффициенты:** Каждая связь между нейронами имеет свой весовой коэффициент, который определяет силу влияния нейрона на следующий.
5. **Вычисление:** На каждом слое нейроны вычисляют взвешенную сумму своих входов, а затем применяют функцию активации.
6. **Выход:** Результат вычислений выходного слоя является выходными данными FNN.

## Преимущества FNN:

- **Простая реализация:** Относительно легка в реализации и понимании.
- **Эффективность:** Может быть очень эффективной для решения широкого круга задач.
- **Универсальность:** Применима для задач классификации, регрессии, прогнозирования и других.

## Применения FNN:

- **Распознавание образов:** Классификация изображений, распознавание рукописного текста.
- **Обработка естественного языка:** Перевод текста, анализ тональности.
- **Прогнозирование:** Прогнозирование спроса, цен на акции.
- **Игры:** Искусственный интеллект для игр.

**FNN** - это мощный инструмент для решения многих задач машинного обучения. Однако важно помнить о её ограничениях и правильно выбирать параметры для достижения наилучших результатов.

# Процесс обучения FNN

## Training v.1

### Гиперпараметры нейросети:

| input_size | output_size |  lr   | weight_decay | patience | factor | fc1 | fc2 | dropout | torch.randn | torch.randint | batch_size | shuffle | total trainings |
| ---------- | ----------- | ----- | ------------ | -------- | ------ | --- | --- | ------- | ----------- | ------------- | ---------- | ------- | --------------- |
|     5      |      10     | 0.001 |     None     |    10    |   0.1  | 128 | 128 |  None   |   1000, 5   |0, 10, (1000,) |     32     |   True  |        3        |

### Прогресс обучения:

0 - 100 Epochs

|      Epoch      |    Progress    |      Loss      |         Time         |
| --------------- | -------------- | -------------- | -------------------- |
|        0        |    [0/1000]    |    2.316299    | 05/07/2024, 18:33:43 |
|        5        |    [0/1000]    |    2.283491    | 05/07/2024, 18:33:44 |
|        75       |    [0/1000]    |    2.266108    | 05/07/2024, 18:33:45 |

100 - 1 000 Epochs

|      Epoch      |    Progress    |      Loss      |         Time         |
| --------------- | -------------- | -------------- | -------------------- |
|        0        |    [0/1000]    |    2.319755    | 05/07/2024, 18:45:41 |
|        13       |    [0/1000]    |    2.283897    | 05/07/2024, 18:45:41 |
|        41       |    [0/1000]    |    2.273901    | 05/07/2024, 18:45:41 |
|        62       |    [0/1000]    |    2.255302    | 05/07/2024, 18:45:41 |
|        164      |    [0/1000]    |    2.254197    | 05/07/2024, 18:45:42 |
|        472      |    [0/1000]    |    2.247628    | 05/07/2024, 18:45:43 |
|        593      |    [0/1000]    |    2.247453    | 05/07/2024, 18:45:43 |

1 000 - 500 000 Epochs

|      Epoch      |    Progress    |      Loss      |         Time         |
| --------------- | -------------- | -------------- | -------------------- |
|        0        |    [0/1000]    |    2.269815    | 05/07/2024, 18:53:21 |
|        67       |    [0/1000]    |    2.268115    | 05/07/2024, 18:53:22 |
|        71       |    [0/1000]    |    2.248463    | 05/07/2024, 18:53:22 |
|        183      |    [0/1000]    |    2.243957    | 05/07/2024, 18:53:22 |
|        199      |    [0/1000]    |    2.239024    | 05/07/2024, 18:53:22 |
|        870      |    [0/1000]    |    2.227217    | 05/07/2024, 18:53:24 |
|        2121     |    [0/1000]    |    2.204968    | 05/07/2024, 18:53:28 |
|        16568    |    [0/1000]    |    2.201313    | 05/07/2024, 18:54:15 |
|        17027    |    [0/1000]    |    2.200411    | 05/07/2024, 18:54:17 |
|        130731   |    [0/1000]    |    2.176197    | 05/07/2024, 19:00:18 |
|        176348   |    [0/1000]    |    2.173238    | 05/07/2024, 19:02:52 |
|        274047   |    [0/1000]    |    2.155731    | 05/07/2024, 19:08:08 |


## Training v.2

### Гиперпараметры нейросети:

| input_size | output_size |    lr    | weight_decay | patience | factor | fc1   | fc2   |dropout| torch.randn | torch.randint | batch_size | shuffle | total trainings |
| ---------- | ----------- | -------- | ------------ | -------- | ------ | ----- | ----- | ----- | ----------- | ------------- | ---------- | ------- | --------------- |
|     5      |      10     |**0.0001**|   **1e-5**   |  **5**   |**0.5** |**256**|**256**|**0.2**|   1000, 5   |0, 10, (1000,) |   **64**   |   True  |     **14**      |

### Экспоненциальное обучение:[^1]

| № of training | total Epochs |
| ------------- | ------------ |
|       1       |    100       |
|       2       |    200       |
|       3       |    400       |
|       4       |    800       |
|       5       |  16 000      |
|       6       |  32 000      |
|       7       |  64 000      |
|       8       |  128 000     |
|       9       |  256 000     |
|       10      |  512 000     |
|       11      |  102 400     |
|       12      |  204 800     |
|       13      |  409 600     |
|       14      |  819 200     |

## Использованные библиолтеки и версия Python

> [!IMPORTANT]
> Python = 3.10.0
> time: `pip install time`
> logging: `pip install logging`
> torch: `pip install torch`

[^1]: Основано на геометрической прогрессии эпох и уменьшении шага обучения по формуле `learning_rate_10 = 0.1 * exp(-0.01 * 10) ≈ 0.0905`[^2]
[^2]: Где **learning_rate_t** - шаг обучения на этапе **t**; **learning_rate_0** - начальный шаг обучения; **decay_rate** - скорость уменьшения шага обучения (обычно небольшое положительное число); **t** - номер текущего этапа обучения
