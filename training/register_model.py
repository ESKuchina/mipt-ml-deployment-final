from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "training" / "artifacts"
SERVICE_MODEL_DIR = PROJECT_ROOT / "service" / "models" / "current"
SERVICE_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    src = ARTIFACTS_DIR / "model.joblib"
    if not src.exists():
        raise FileNotFoundError(f"Не найдена обученная модель: {src}")

    dst = SERVICE_MODEL_DIR / "model.joblib"
    shutil.copy2(src, dst)
    print(f"Модель зарегистрирована для сервиса: {dst}")


if __name__ == "__main__":
    main()
