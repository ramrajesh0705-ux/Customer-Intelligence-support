# Customer Intelligence Support

## Run locally

The project uses two local processes:

1. Install frontend dependencies: `python -m pip install -r requirements-frontend.txt`.
2. Install API dependencies: `python -m pip install -r requirements-api.txt`.
3. Place trained artifacts under `models/`, or set `CATEGORY_MODEL_PATH`, `LABEL_ENCODER_PATH`, `RESOLUTION_MODEL_PATH`, and `PRIORITY_MODEL_PATH`.
4. Start the API from the project root: `uvicorn prediction_api.main:app --reload --port 8000`.
5. In a second terminal, start Streamlit: `streamlit run app/main.py --server.port 8501`.

Open `http://localhost:8501` for the UI. API documentation is available at `http://localhost:8000/docs`, and status is available at `http://localhost:8000/health`.

Set `PREDICTION_API_URL` when the frontend needs to call a different API address. Docker Compose can later use the same setting with the API service name.

## Run with Docker Compose

Install Docker Desktop and place the trained model artifacts in the root `models/` directory. Then run from the project root:

`docker compose up --build`

Open `http://localhost:8501` for Streamlit. FastAPI is available at `http://localhost:8000/docs`. Stop the services with:

`docker compose down`

The model directory is mounted read-only into the API container. It is intentionally excluded from the image build context, so the images do not contain model binaries.
