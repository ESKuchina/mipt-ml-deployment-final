# Финальный проект по дисциплине «Развертывание ML-моделей»

## О проекте

В этом проекте я спроектировала и реализовала ML-систему для **прогнозирования риска дефицита товара на складе** на горизонте 7 дней.

Бизнес-цель проекта — **снизить долю SKU, ушедших в дефицит**, за счет более раннего выявления рискованных позиций и последующей автоматизации принятия решений о пополнении.

Модель решает задачу **бинарной классификации**:
- `1` — товар уйдет в дефицит в ближайшие 7 дней;
- `0` — товар не уйдет в дефицит в ближайшие 7 дней.

## Заявленный уровень зрелости

Проект реализован как **ML-система уровня зрелости 2**.

В рамках проекта предусмотрены:
- версионирование кода в GitHub;
- CI/CD-контур;
- feature store в упрощенном виде;
- сервинг модели через API;
- мониторинг сервиса и модели;
- система управления экспериментами;
- оркестратор переобучения.

## Архитектура решения

Система состоит из следующих компонентов:

- **PostgreSQL** — хранение исходных данных и признаков;
- **Feature store** — отдельный слой признаков в PostgreSQL;
- **Training pipeline** — подготовка данных, обучение, оценка и регистрация модели;
- **MLflow** — управление экспериментами и версиями моделей;
- **Airflow** — оркестрация пайплайна переобучения;
- **FastAPI** — API сервиса инференса;
- **Nginx** — blue-green переключение трафика;
- **Prometheus** — сбор метрик;
- **Grafana** — дашборд мониторинга;
- **Docker Compose** — инфраструктура как код.

## Структура репозитория

```text
.
├── airflow/
│   ├── dags/
│   │   └── retrain_pipeline.py
│   ├── Dockerfile
│   └── requirements.txt
├── docs/
│   ├── architecture.md
│   ├── business_goal.md
│   ├── manifest.md
│   ├── mdd_adr.md
│   └── sli_slo.md
├── infra/
│   ├── docker-compose.yml
│   ├── prometheus.yml
│   ├── grafana/
│   │   └── stockout_monitoring_dashboard.json
│   └── nginx/
│       ├── nginx.active.conf
│       ├── nginx.blue.conf
│       └── nginx.green.conf
├── screenshots/
├── service/
│   ├── app.py
│   ├── model_loader.py
│   ├── metrics.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── models/
│       └── current/
└── training/
    ├── preprocess.py
    ├── train.py
    ├── evaluate.py
    ├── register_model.py
    └── artifacts/
```

## Основные текстовые артефакты

В репозитории подготовлены отдельные документы по ключевым шагам задания:

- `docs/business_goal.md` — постановка цели;
- `docs/manifest.md` — одностраничный ML-манифест;
- `docs/architecture.md` — описание архитектуры уровня 2;
- `docs/sli_slo.md` — SLI/SLO на техническом, модельном и бизнесовом уровнях;
- `docs/mdd_adr.md` — ADR с решением по Metrics Driven Development.

## Публичный сервис

Сервис развернут в облаке и доступен по адресу:

- https://stockout-risk-service.onrender.com

### Проверка доступности

`GET /health`

Пример:

```bash
curl https://stockout-risk-service.onrender.com/health
```

### Получение предсказания

`POST /predict`

Пример:

```bash
curl -X POST https://stockout-risk-service.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sku_id": 101,
    "warehouse_id": 3,
    "current_stock": 40,
    "safety_stock": 60,
    "demand_7d": 120,
    "demand_14d": 220,
    "demand_30d": 430,
    "orders_7d": 18,
    "orders_14d": 35,
    "orders_30d": 70,
    "avg_daily_demand_7d": 17.14,
    "avg_daily_demand_30d": 14.33,
    "demand_std_30d": 11.5,
    "demand_trend_7d_vs_prev_7d": 1.3,
    "days_since_replenishment": 12,
    "last_replenishment_qty": 80,
    "stock_to_weekly_demand_ratio": 0.33,
    "stock_to_safety_ratio": 0.67,
    "days_of_cover": 2.3,
    "day_of_week": 2,
    "month": 5,
    "category_id": 4,
    "warehouse_region_id": 2,
    "warehouse_turnover_rate": 1.8
  }'
```

## Как запустить проект локально

### 1. Подготовить первую модель

Из корня проекта:

```bash
mkdir -p service/models/current
mkdir -p training/artifacts

python3 training/preprocess.py
python3 training/train.py
python3 training/evaluate.py
python3 training/register_model.py
```

### 2. Поднять стек

```bash
docker compose -f infra/docker-compose.yml up -d --build
```

### 3. Проверить состояние контейнеров

```bash
docker ps
```

### 4. Проверить сервис

```bash
curl http://localhost/health
```

## Используемые локальные порты

Из-за занятых стандартных портов часть сервисов опубликована на альтернативные порты:

- API через Nginx — `http://localhost`
- PostgreSQL — `5433`
- MLflow — `http://localhost:5001`
- Airflow — `http://localhost:8081`
- Prometheus — `http://localhost:9091`
- Grafana — `http://localhost:3001`

## Airflow

Airflow используется как оркестратор переобучения.

### DAG
`retrain_stockout_model`

### Шаги DAG
- `preprocess`
- `train`
- `evaluate`
- `register_model`

### Доступ
- URL: `http://localhost:8081`
- логин: `admin`
- пароль: `admin`

## MLflow

MLflow используется как система управления экспериментами:
- логируются параметры обучения;
- сохраняются метрики;
- сохраняется артефакт модели;
- регистрируются версии моделей.

URL: `http://localhost:5001`

## Мониторинг

### Prometheus
Prometheus собирает метрики с FastAPI-сервисов.

URL: `http://localhost:9091`

### Grafana
Grafana отображает дашборд мониторинга:
- p95 времени отклика `/predict`;
- число успешных запросов;
- число ошибочных запросов;
- ошибки сервиса по типам.

URL: `http://localhost:3001`

Экспортированный JSON дашборда сохранен в:
- `infra/grafana/stockout_monitoring_dashboard.json`

## Blue-Green переключение

Для выкладки новых версий используется схема **blue-green**:

- `fastapi-blue`
- `fastapi-green`

Текущий активный маршрут задается через `infra/nginx/nginx.active.conf`.

Переключение можно выполнять заменой активного конфига и перезагрузкой Nginx.

## Скриншоты

В папке `screenshots/` сохранены подтверждающие материалы:

- [docker ps](screenshots/01_docker_ps.png)
- [health-check](screenshots/02_health_check.png)
- [успешный predict](screenshots/03_predict_response.png)
- [список DAG в Airflow](screenshots/04_airflow_dag_list.png)
- [успешный запуск DAG](screenshots/05_airflow_successful_run.png)
- [эксперимент в MLflow](screenshots/06_mlflow_experiment.png)
- [метрики run в MLflow](screenshots/07_mlflow_run_metrics.png)
- [targets в Prometheus](screenshots/08_prometheus_targets.png)
- [дашборд Grafana](screenshots/09_grafana_dashboard.png)
- [статус сервиса в Render](screenshots/10_render_live_service.png)
- [проверка cloud health/predict](screenshots/11_render_health_or_predict.png)

## Что реализовано в проекте

На текущем этапе реализованы:
- постановка задачи и полный комплект текстовых артефактов;
- инфраструктурный каркас уровня 2;
- сервис инференса;
- обучение модели;
- Airflow DAG;
- MLflow;
- мониторинг в Prometheus и Grafana;
- blue-green схема маршрутизации;
- облачное развертывание API-сервиса на Render.

## Вывод

В проекте реализована цельная ML-система, которая соответствует заявленному уровню зрелости 2 и покрывает весь жизненный цикл модели: подготовку данных, обучение, оценку качества, регистрацию версии, онлайн-инференс, мониторинг и контролируемую замену версии в производственной среде.
