# Веб-сервис классификации одежды (CNN + Fashion-MNIST, Flask)

## Состав проекта

- `train_model.py` — загрузка `Fashion-MNIST`, подготовка тензоров `(28, 28, 1)`, обучение свёрточной сети `build_cnn()`, оценка на тестовом наборе и сохранение модели в `artifacts/garment_cnn.keras` и `artifacts/garment_cnn.tflite`.
- `server.py` — Flask-приложение (`GET /`, `POST /api/classify`) на лёгком рантайме `ai-edge-litert` (LiteRT / TFLite, без полного TensorFlow), с проверкой формата и размера изображения (до 5 МБ), функцией `prepare_image()` и выдачей предсказанного класса.
- `templates/home.html`, `static/theme.css`, `static/client.js` — веб-интерфейс в тёмной двухколоночной теме.
- `requirements.txt` — лёгкие зависимости только для веб-сервера и деплоя (`ai-edge-litert`, `flask`, `gunicorn` для Linux и др.).
- `requirements-train.txt` — зависимости для локального обучения `train_model.py` (`tensorflow`, `numpy`).

## Запуск

1. Обучение модели (при необходимости переобучить; требует `requirements-train.txt`):
   ```powershell
   .\.venv\Scripts\python.exe train_model.py
   ```
2. Запуск веб-сервиса локально:
   ```powershell
   .\.venv\Scripts\python.exe server.py
   ```
3. Открыть в браузере: `http://127.0.0.1:8001`
