# Исходный пакет кейса Beeline

Эти 15 файлов скопированы без изменений из `amir-research/`; SHA-256 каждого сохранён в [манифесте](../SHA256SUMS.txt). Здесь лежат **входные материалы задачи**, а не готовое решение команды.

| Группа | Файлы |
|---|---|
| Постановка | `PARTICIPANT_GUIDE.md`, `PARTICIPANT_GUIDE.pdf` |
| Текущая аудитория и справочники | `customer_profile.csv`, `tariff_dictionary.csv`, `feature_dictionary.csv` |
| Исторические данные | `data/change_tariff.csv`, `data/traffic.csv`, `data/arpu_monthly.csv`, `data/dict_tariff.csv` |
| API и оценка | `environment.py`, `scoring_core.py`, `mock_environment.py` |
| Шаблон и локальная проверка | `agent_template.py`, `local_eval.py`, `make_submission.py` |

Скрипты локальной проверки ожидают `agent.py` рядом с собой и относительные пути к данным. Для разработки лучше собрать отдельную рабочую копию пакета в репозитории, выданном для сдачи, и запускать команды из её корня. В эту папку не записывать экспериментальные результаты и `submission.csv`: это исходный архив контекста.
