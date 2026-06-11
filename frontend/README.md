# Vector Search Engine Frontend

React + Vite UI for the FastAPI vector search backend.

## Run locally

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

By default, Vite proxies `/api` to `http://localhost:8000`. Start the backend separately, for example:

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

To point the frontend at another backend URL:

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

The Search page calls `POST /search`. The status widget calls `GET /openapi.json`.
