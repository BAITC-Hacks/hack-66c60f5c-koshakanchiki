# Исходный пакет кейса Beeline

Здесь сохранены 15 исходных файлов кейса без изменений; SHA-256 каждого записан в [манифесте](../SHA256SUMS.txt). Это **входные материалы задачи**. Выбранный подход и проверки команды находятся в [final-research/](../../final-research/README.md).

| Группа | Файлы |
|---|---|
| Постановка | `PARTICIPANT_GUIDE.md`, `PARTICIPANT_GUIDE.pdf` |
| Текущая аудитория и справочники | `customer_profile.csv`, `tariff_dictionary.csv`, `feature_dictionary.csv` |
| Исторические данные | `data/change_tariff.csv`, `data/traffic.csv`, `data/arpu_monthly.csv`, `data/dict_tariff.csv` |
| API и оценка | `environment.py`, `scoring_core.py`, `mock_environment.py` |
| Шаблон и локальная проверка | `agent_template.py`, `local_eval.py`, `make_submission.py` |

Скрипты локальной проверки ожидают `agent.py` рядом с собой и относительные пути к данным. Для разработки лучше собрать отдельную рабочую копию пакета в репозитории, выданном для сдачи, и запускать команды из её корня. В эту папку не записывать экспериментальные результаты и `submission.csv`: это исходный архив контекста.
