# Журнал направлений и границ поиска

Дата: 23.09.2026, часовой пояс команды Asia/Almaty. Это тематический журнал фактически выполненного поиска, а не поминутный лог всех HTTP-запросов. Проверенные адреса и результаты доступа находятся в [SOURCES.csv](SOURCES.csv).

## Кейс и организаторы

Проверялись запросы:

```text
"HackAlem" "Beeline"
"Beeline Tariff Marketing Campaigns"
"HackAlem" "Beeline" "agent.py"
"HackAlem" "тариф"
"HackAlem" site:t.me
"HackAlem" site:t.me/s/
HackAlem AI 2026 официальный сайт телеграм 23 сентября
"hackalem" "13:00" "18:00"
site:github.com "HackAlem" "Beeline"
```

Путь к первоисточникам: официальный лендинг → карточка платформы/положение; публичная Telegram-копия → единый Google Doc → официальные PDF. Web-инструмент не всегда открывал страницы; публичный HTML был дополнительно прочитан обычным HTTP. Это позволило проверить разделы положения и совпадение двух внешних PDF с локальными.

Прямой маршрут `t.me/s/hackalem` вернул карточку контакта вместо ленты. Официальный чат доступен как приглашение; история сообщений недоступна в этом исследовании. Telemetr использован только как вторичная копия с явно указанными ограничениями.

## GitHub

Проверены публичные repository-search запросы `HackAlem`, `"Beeline Tariff Marketing Campaigns" in:readme`, `HackAlem Beeline in:readme`, `HackAlem tariff in:readme`, `"run_pilot" "tariff" in:readme`. Общий запрос вернул 43 репозитория;32 корневых README получены,11 недоступны. Совпадение с конкретным кейсом в полученных README не обнаружено. Приватные репозитории, другие ветки и весь код не исследовались. Подробный результат по адресу каждого README сохранён в [sources_organizers.csv](sources_organizers.csv).

## Методы исследования и оптимизации

Поиск включал названия и сочетания:

```text
Bandits with Knapsacks
Resourceful Contextual Bandits
Best Arm Identification fixed budget Successive Rejects
Almost Optimal Exploration in Multi-Armed Bandits
A Tutorial on Thompson Sampling
Simple Bayesian Algorithms for Best Arm Identification
Knowledge Gradient correlated normal beliefs
misspecified priors multi armed bandits
hierarchical Bayesian bandits
batch arm pulls fixed budget
confidence sequences optional stopping
Conjugate Bayesian analysis Gaussian distribution
```

Отобраны 15 первичных источников: arXiv, авторские PDF, PMLR, NeurIPS, журнал/PMC. Уровень чтения — аннотация или проверенные разделы — указан отдельно для каждого. Гарантии исходных теорем не объявлялись гарантиями 20 пилотов этого кейса.

## Uplift, телеком и библиотеки

```text
Uplift Modeling for Multiple Treatments with Cost Optimization
Response Transformation and Profit Decomposition for Revenue Uplift Modeling
Doubly Robust Policy Evaluation and Learning
EconML unconfoundedness
CausalML multiple treatment cost optimization
Criteo uplift dataset randomized
scikit uplift fetch_megafon
telecom uplift marketing campaign optimization
Vowpal Wabbit contextual bandit action cost probability
Vowpal Wabbit offline policy evaluation
scipy optimize milp time_limit
Google OR-Tools multiple knapsack
scikit-learn common pitfalls data leakage
```

Использованы первичные публикации, официальные проекты и документация. В выдаче попадались Reddit, Wikipedia, обзоры и репозитории сторонних авторов; технические рекомендации на их неподтверждённых утверждениях не строились.

## Что сознательно не приравнивалось к ответу на кейс

- Действующие рыночные тарифы и бизнес-показатели Beeline: справочник хакатона синтетический.
- Общие анонсы мероприятия: они не раскрывают скрытые эффекты.
- Статьи про churn classification: вероятность ухода не равна эффекту тарифного предложения.
- Частота исторического перехода: не известная вероятность маркетингового назначения или отклика.
- Найденное Q&A-видео без расшифровки: не подтверждение конкретного ответа.
- Нулевой поисковый результат: не доказательство отсутствия материалов вообще.

## Какие пробелы остаются

Сообщения внутри чата; полный текст ответов Q&A; индивидуальные условия карточки Beeline после входа участника; доступность исторических файлов и версии библиотек в judge environment; точное межкейсное сопоставление технических результатов. Сформулированные вопросы находятся в [EXPERIMENTS_AND_DECISIONS.md](EXPERIMENTS_AND_DECISIONS.md).
