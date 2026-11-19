# Sentinel-2 Crop Classifier

A web application that classifies crops from a field boundary and a growing-season date range. It combines a React map frontend, a FastAPI backend, and a trained 18-class PyTorch time-series model.

## Try the deployed project

Open **http://15.206.67.161/**. The deployment's model-information endpoint is **http://15.206.67.161/api/v1/model-info**. The map and statistics pages use the deployed backend; the Live Validation page displays sample parcels and can request fresh predictions.

See [EVALUATION_RESULTS.md](./EVALUATION_RESULTS.md) for the dated production metrics and the important distinction between the deployed model report and the training history that was not saved.

## Run locally

Requirements: Python 3.10+ and Node.js/npm.

1. In a terminal, install and start the backend:

   ```powershell
   cd backend
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   Copy-Item .env.example .env
   # Edit .env and enter your own Copernicus Data Space OAuth client credentials.
   uvicorn app.main:app --reload --port 9000
   ```

2. In a second terminal, start the frontend:

   ```powershell
   cd frontend
   npm ci
   npm run dev
   ```

3. Open the Vite URL printed in the frontend terminal (normally `http://localhost:5173`).

The frontend's local API URL is `http://localhost:9000/api/v1`. Live predictions require valid Copernicus Data Space credentials in `backend/.env`, network access, and usable Sentinel-2 scenes in the selected date range. The deployed link is the simplest way to demonstrate live behavior without configuring local credentials.

## What's included

- `frontend/`: React/Vite map and project pages.
- `backend/app/`: FastAPI endpoints, inference, Sentinel-2 fetching, and demo data.
- `backend/*.pth`: saved model checkpoints. The backend is configured to load `multicountry_india_custom_18class.pth`.
- `backend/training/`: training notebooks and Python training/evaluation pipeline.
- `EVALUATION_RESULTS.md`: production API metrics, test-set context, and training-history disclosure.

Large downloaded training/fetch datasets and private hand-labeled parcel inputs are not included. See the results report before interpreting the training metrics or attempting a full retrain.

## API routes

- `GET /health` — backend health check.
- `GET /api/v1/model-info` — model and held-out evaluation information.
- `GET /api/v1/demo/parcels` — sample validation parcels.
- `POST /api/v1/demo/repredict` — request fresh predictions for the sample parcels.
- `POST /api/v1/predict` — classify a supplied AOI and date range.

The FastAPI application is started from `backend/`; its interactive API documentation is available at `http://localhost:9000/docs`.
