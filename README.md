# EfficientNet Prediction API

## What this project does

- loads `best_model.keras`
- reads `class_names.pkl`
- exposes a FastAPI endpoint to predict disease labels from an uploaded image
- optionally calls a Gemini-compatible LLM for disease explanation

## Local setup

1. Install dependencies:

```bash
cd e:\efficientnet_newdataset
py -m pip install -r requirements.txt
```

2. Run the API:

```bash
python api.py
```

3. Open docs:

```
http://127.0.0.1:8000/docs
```

## Docker deployment

Build the image:

```bash
docker build -t efficientnet-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 \
  -e GEMINI_API_KEY=your_api_key_here \
  efficientnet-api
```

If you do not need Gemini description support, omit `-e GEMINI_API_KEY`.

## API endpoints

- `POST /predict` - upload an image and get prediction results
- `GET /describe/{label}` - get disease description and care steps for one label

Use `POST /predict?describe=true` to get both predictions and Gemini disease guidance in one response.

## Why Docker?

Docker packages your app with its runtime and dependencies into a container. That means:

- the app runs the same way on development, staging, and production
- you do not need to install Python packages manually on the target machine
- you avoid dependency version conflicts
- deployment is easier and more portable

In short: Docker makes your model API more reliable and simpler to deploy.
