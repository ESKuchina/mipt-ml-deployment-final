from prometheus_client import Counter, Histogram


REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "Время обработки HTTP-запросов",
    ["endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

PREDICT_COUNTER = Counter(
    "predict_requests_total",
    "Общее число запросов на предсказание",
    ["status"],
)

ERROR_COUNTER = Counter(
    "service_errors_total",
    "Ошибки сервиса по типам",
    ["endpoint", "error_type"],
)
