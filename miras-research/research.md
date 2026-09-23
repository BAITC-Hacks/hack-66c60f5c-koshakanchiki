casual, [9/23/26 2:43 PM]
# 🎯 Финальный стек агента смены тарифа — что, подо что, как

Архитектура в одну строку: scikit-uplift считает, кому воздействие даёт плюс → ω-UCB (OmegaUCB) раздаёт пилотный бюджет между кандидатами → impactiq profit-cutoff + knapsack выбирает ≤10 без downsell → LLM только генерит кандидатов и объясняет. Всё MIT, ноль тяжёлых фреймворков.

---

## Компонент 1 — scikit-uplift

- Репо: https://github.com/maks-sh/scikit-uplift · MIT · 817★ · ветка master · docs: https://www.uplift-modeling.com/en/latest/
- Зависимости: лёгкие (scikit-learn, numpy, pandas, matplotlib, requests, tqdm)
- ПОД ЧТО: подзадача B — оценить инкремент выручки на контакт для каждого (сегмент × тариф), со знаком (downsell = `û<0`).
- КАК:
  
  from sklift.models import TwoModels
  from xgboost import XGBRegressor
  # target = изменение выручки/ARPU (регрессия!), treat = 0/1
  tm = TwoModels(estimator_trmnt=XGBRegressor(), estimator_ctrl=XGBRegressor(), method='vanilla')
  tm.fit(X_train, y_revenue, treat_train)
  uplift = tm.predict(X_cand)        # û на кандидата, может быть < 0
  
  ⚠️ Учи на RCT-части, оценивай на holdout. qini_auc_score из sklift — для бинарного отклика; для непрерывной выручки оценивай profit/policy-value (компонент 3), Qini держи вторичным.

## Компонент 2 — heymarco/OmegaUCB (ω-UCB)

- Репо: https://github.com/heymarco/OmegaUCB · MIT · 3★ (за статьёй KDD 2024) · ветка master
- Зависимости: чистый numpy + scipy (проверено по исходнику)
- Забрать:
  - https://raw.githubusercontent.com/heymarco/OmegaUCB/master/components/bandits/wucb.py
  - https://raw.githubusercontent.com/heymarco/OmegaUCB/master/components/bandits/abstract.py
- ПОД ЧТО: подзадача A — раздать ограниченный пилотный бюджет туда, где лучшее отношение «прирост выручки / стоимость контакта» при высокой неопределённости.
- КАК: reward, cost нормируем в [0,1]; клип downsell по нулю (знак храним отдельно для отбора).
  
  from vendor.wucb import WUCBArm
  arms = [WUCBArm(confidence=0.95) for _ in candidates]
  for t in range(pilot_budget):                      # лимит пилотов
      i = max(range(len(arms)), key=lambda k: arms[k].sample())   # UCB_r / LCB_c
      r, c = run_pilot(candidates[i])                # шумный тест на малой выборке
      for k, a in enumerate(arms):
          a.update(c if k == i else 0, r if k == i else 0, was_pulled=(k == i))
  

## Компонент 3 — abhi-iitg/impactiq

- Репо: https://github.com/abhi-iitg/impactiq-campaign-uplift-optimization · MIT · ветка main
- Зависимости: лёгкие (sklearn, statsmodels, scikit-uplift; без causalml)
- Забрать:
  - src/uplift_models.py — https://raw.githubusercontent.com/abhi-iitg/impactiq-campaign-uplift-optimization/main/src/uplift_models.py (X-learner, если T-learner слаб на дисбалансе)
  - src/power_analysis.py — https://raw.githubusercontent.com/abhi-iitg/impactiq-campaign-uplift-optimization/main/src/power_analysis.py (сколько клиентов в пилоте под нужный MDE)
  - notebooks/06_targeting_policy.ipynb — https://github.com/abhi-iitg/impactiq-campaign-uplift-optimization/blob/main/notebooks/06_targeting_policy.ipynb (profit-кривая, cutoff)
- ПОД ЧТО: целевая функция отбора — Σ(û·value − contact_cost), отсечение negative-effect, размер пилота.
- КАК (финальный отбор ≤10):
  
  scored = [(i, u*value[i] - cost[i], cost[i]) for i, u in enumerate(uplift)]
  scored = [s for s in scored if s[1] > 0]                 # downsell / sleeping dogs — вон
  scored.sort(key=lambda s: s[1]/s[2], reverse=True)       # net на тенге контакта
  picked, spent = [], 0
  for i, net, c in scored:
      if len(picked) < 10 and spent + c <= contact_budget:
          picked.append(i); spent += c
  

## Компонент 4 — CLAHRCWessex/subset-selection-problem (OCBA-m) — опционально

casual, [9/23/26 2:43 PM]
- Репо: https://github.com/CLAHRCWessex/subset-selection-problem · MIT · ветка master
- Забрать: OCBA_m_Experiments.ipynb + пакет bootcomp/
- ПОД ЧТО: если пилоты идут батчами, а не последовательно — принципиальный выбор «лучших m=10» под фиксированным бюджетом. Даёт сильную методологическую защиту («это OCBA-m: пилоты = репликации, m = 10»).
- КАК: заменяет цикл ω-UCB — раздаёт весь бюджет разом ∝ дисперсии/зазору кандидатов.

## Компонент 5 — LLM-слой (без репо, свои промпты)

- ПОД ЧТО: (0) генерация кандидатов сегмент×тариф×канал + приоры; (2) сжатие истории пилотов; (4) NL-объяснение портфеля.
- КАК / граница: LLM никогда не выбирает арм. Обоснование в README: [2403.15371](https://arxiv.org/abs/2403.15371), [2502.00225](https://arxiv.org/abs/2502.00225), [2608.16707](https://arxiv.org/abs/2608.16707) — LLM в роли селектора проигрывает линейной регрессии и ловит semantic bias.

---

## Сборка проекта


pip install scikit-uplift xgboost numpy pandas scipy scikit-learn
mkdir -p campaign_agent/vendor && cd campaign_agent/vendor
curl -sO https://raw.githubusercontent.com/heymarco/OmegaUCB/master/components/bandits/wucb.py
curl -sO https://raw.githubusercontent.com/heymarco/OmegaUCB/master/components/bandits/abstract.py
git clone --depth 1 https://github.com/abhi-iitg/impactiq-campaign-uplift-optimization impactiq
git clone --depth 1 https://github.com/CLAHRCWessex/subset-selection-problem ocba   # опц.


campaign_agent/
├── candidates.py   # 5 — LLM-генератор (опц.)
├── uplift.py       # 1 — scikit-uplift TwoModels(XGBRegressor)
├── allocate.py     # 2 — vendor/wucb.py::WUCBArm
├── select.py       # 3 — impactiq profit-cutoff + knapsack
├── explain.py      # 5 — LLM-объяснение (опц.)
└── vendor/         # wucb.py, abstract.py, impactiq/, ocba/


## Одним взглядом

| # | Репо | Лиц. | Стадия | Что берём |
|---|---|---|---|---|
| 1 | github.com/maks-sh/scikit-uplift | MIT | uplift (B) | TwoModels(XGBRegressor) |
| 2 | github.com/heymarco/OmegaUCB | MIT | аллокация (A) | wucb.py → WUCBArm |
| 3 | github.com/abhi-iitg/impactiq-campaign-uplift-optimization | MIT | отбор + sizing | profit-cutoff, X-learner, power_analysis |
| 4 | github.com/CLAHRCWessex/subset-selection-problem | MIT | best-10 (батч) | OCBA-m |

НЕ ставить: causalml, econml (тяжёлая сборка) · не копировать код: tkarim45/uplift-targeting-engine (нет лицензии + скрыто тянет causalml) — берём только threshold = np.quantile(scores, 1−budget).

---

Это финальный стек. Хочешь — материализую campaign_agent/ (5 модулей + vendor + запуск на Hillstrom из коробки) прямо в /myapp/pentest/campaign_agent/, либо сохраню этот вариант как README.md в проект.