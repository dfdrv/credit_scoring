# HW6 - Обучение моделей

## Постановка задачи

Цель работы - используя признаки, которые были в предыдущих дз, обучить 4 вида моделей машинного обучения, подобрать гиперпараметры, провести анализ, сравнить результаты и метрики. 

**Целевая переменная:** `TARGET`.

Обучение проводилось на обучающей части датасета `application` (`dataset_source = 'train'`).  
Для финальной оценки качества использовалась тестовая выборка.

## Подход к формированию признакового пространства

На момент выполнения задания признаки, разработанные в HW4, ещё не прошли полноценную проверку.  

В модели вошли:
- признаки из таблицы `application`
- проверенные признаки из HW2

## Использованные признаки

### Финансовый блок
- `AMT_INCOME_TOTAL`
- `AMT_CREDIT`
- `AMT_ANNUITY`
- `AMT_GOODS_PRICE`

### Блок кредитной и жизненной истории
- `DAYS_BIRTH`
- `DAYS_EMPLOYED`
- `DAYS_REGISTRATION`
- `DAYS_ID_PUBLISH`
- `DAYS_LAST_PHONE_CHANGE`

### Социальный блок
- `CNT_CHILDREN`
- `CNT_FAM_MEMBERS`
- `OBS_30_CNT_SOCIAL_CIRCLE`
- `DEF_30_CNT_SOCIAL_CIRCLE`

### Внешние скоринговые источники
- `EXT_SOURCE_1`
- `EXT_SOURCE_2`
- `EXT_SOURCE_3`

### Блок кредитного бюро
- `AMT_REQ_CREDIT_BUREAU_MON`
- `AMT_REQ_CREDIT_BUREAU_QRT`
- `AMT_REQ_CREDIT_BUREAU_YEAR`

### Категориальные признаки
- `CODE_GENDER`, `NAME_CONTRACT_TYPE`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`
- `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`

### Собственные фичи из HW2 (5 шт.)
- `credit_to_income_ratio`
- `annuity_to_income_ratio`
- `active_credit_count`
- `credit_history_days`
- `prev_application_count`

## Предобработка данных

Были созданы отдельные пайплайны предобработки:

- **Для линейных моделей:**  
  числовые признаки — `SimpleImputer(median)` + `StandardScaler`  
  категориальные признаки — `SimpleImputer(most_frequent)` + `OneHotEncoder`

- **Для моделей на основе деревьев:**  
  числовые признаки — `SimpleImputer(median)` (без масштабирования)  
  категориальные признаки — `SimpleImputer(most_frequent)` + `OrdinalEncoder`

## Обученные модели

- Logistic Regression
- Decision Tree
- Random Forest
- HistGradientBoostingClassifier 

Подбор гиперпараметров выполнялся с помощью **Optuna**.  
Валидация - **Stratified K-Fold Cross Validation** (учитывая сильный дисбаланс классов).

**Ключевая метрика оптимизации:** ROC-AUC

## Результаты

| Модель                  | ROC-AUC   | PR-AUC    | F1     | Precision | Recall  | Accuracy |
|-------------------------|-----------|-----------|--------|-----------|---------|----------|
| Gradient Boosting       | **0.7601** | 0.2501    | 0.0407 | 0.5556    | 0.0211  | 0.9196   |
| Random Forest           | 0.7508    | 0.2331    | 0.2308 | 0.3274    | 0.1782  | 0.9041   |
| Logistic Regression     | 0.7475    | 0.2240    | 0.2602 | 0.1609    | **0.6798** | 0.6879 |
| Decision Tree           | 0.7177    | 0.1923    | 0.0247 | 0.4532    | 0.0127  | 0.9191   |

## Анализ результатов

- Лучшее качество по основной метрике **ROC-AUC = 0.7601** показала модель **Gradient Boosting**.
- Модель демонстрирует наилучшую способность ранжировать клиентов по уровню кредитного риска.
- Logistic Regression показала высокий Recall (0.6798), но очень низкий Precision (0.1609) - это может привести к большому количеству ложных отказов, что негативно влияет на бизнес.
- Random Forest показал результаты, близкие к бустингу.

## Финальный выбор модели

**Итоговая модель: Gradient Boosting (HistGradientBoostingClassifier)**

## Вывод

По совокупности качества и практической применимости в качестве финальной модели выбран **Gradient Boosting**.

## Подбор порога классификации (Threshold Tuning)

Стандартный порог `0.5` оказался неоптимальным
Для моделей были подобраны пороги:

- Logistic Regression — 0.48  
- Decision Tree — 0.50  
- Random Forest — 0.10 
- Gradient Boosting — 0.10  


## Результаты после подбора порога (на test)

| Модель               | Threshold | Precision | Recall  | F1     | F2     | Accuracy |
|----------------------|----------|-----------|---------|--------|--------|----------|
| Logistic Regression  | 0.48     | 0.1514    | 0.6967  | 0.2487 | 0.4050 | 0.6603   |
| Decision Tree        | 0.50     | 0.1545    | 0.5716  | 0.2433 | 0.3712 | 0.7130   |
| Random Forest        | 0.10     | 0.1800    | 0.5559  | 0.2720 | 0.3922 | 0.7598   |
| Gradient Boosting    | 0.10     | 0.1844    | 0.5897  | 0.2809 | 0.4096 | 0.7563   |

## Вывод

- ROC-AUC почти не изменился, но поведение моделей существенно поменялось
- Вырос Recall → модели начали выявлять дефолты
- Увеличилось число ложных отказов (FP)

