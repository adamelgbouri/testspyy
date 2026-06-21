# Commodity Trading Desk — Web Platform

Production-style rebuild of the Streamlit prototype.

| Layer       | Stack                                  |
|-------------|----------------------------------------|
| Frontend    | Next.js 14 (App Router) · TypeScript · Tailwind · Recharts · Lucide |
| Backend     | FastAPI · Pydantic v2 · Uvicorn        |
| Engine      | NumPy · Pandas · SciPy · scikit-learn  |
| Live data   | yfinance (Yahoo) with graceful fallback |
| Deployment  | Vercel (frontend) + Fly.io/Railway (backend) |

```
web/
├── backend/
│   ├── commodity_engine/        # Pure-Python analytics package
│   │   ├── config.py            # Commodity templates
│   │   ├── data.py              # Synthetic + live data
│   │   ├── balance.py           # S&D balance engine
│   │   ├── fair_value.py        # Inventory-to-price regression
│   │   └── options.py           # Black-76 pricer + Greeks
│   ├── main.py                  # FastAPI entry point
│   └── requirements.txt
└── frontend/
    ├── app/
    │   ├── page.tsx             # Landing page
    │   ├── dashboard/           # Market analytics
    │   └── options/             # Black-76 pricer UI
    ├── components/              # Sidebar, KPICard, TickerTape, BalanceChart…
    ├── lib/                     # Typed API client + helpers
    └── tailwind.config.ts       # Design system
```

## Quick start (local)

### 1. Backend

```bash
cd web/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Visit http://localhost:8000/docs for the auto-generated OpenAPI explorer.

### 2. Frontend

In a new terminal:

```bash
cd web/frontend
cp .env.example .env.local        # leave the default API URL
npm install
npm run dev
```

Visit http://localhost:3000.

You should see:
- A landing page with a hero and feature grid
- `/dashboard` — commodity selector, KPI strip, supply/demand chart, world balance + regional table
- `/options` — Black-76 pricer with Greeks and a payoff chart that updates live

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET    | `/api/health` | Sanity check |
| GET    | `/api/commodities` | List supported commodities |
| GET    | `/api/spot/{key}` | Live or reference spot price |
| GET    | `/api/balance/{key}` | Monthly S&D balance + fair value |
| GET    | `/api/regional/{key}` | Regional supply/demand split |
| GET    | `/api/curve/{key}` | Live or synthetic futures curve |
| POST   | `/api/options/price` | Black-76 price + Greeks |

All schemas are documented at `/docs` and `/redoc`.

## Extending the platform

### Add a commodity
Edit `commodity_engine/config.py` — one dict entry. The API and frontend pick
it up automatically.

### Add a new page
1. Drop a new route under `frontend/app/<slug>/page.tsx`.
2. Add a link to `frontend/components/Sidebar.tsx`.
3. Add the matching endpoint in `backend/main.py` if you need new data.

### Replace synthetic data with real feeds
The `commodity_engine/data.py` functions already check `DISABLE_YF=1` to skip
live fetches. For production, swap them for a real-feed adapter (Refinitiv,
Polygon, TradingView via `tvdatafeed`) — same return shapes, same downstream
code.

## Deployment hints

- **Frontend → Vercel**: zero-config. Set `NEXT_PUBLIC_API_URL` to your backend
  domain in the Vercel project settings.
- **Backend → Fly.io / Railway**: a 3-line `Dockerfile` runs `uvicorn main:app
  --host 0.0.0.0 --port $PORT`. Don't forget to whitelist the Vercel domain
  in the CORS middleware in `main.py`.
- **DB layer** (positions, users): drop in Supabase or Neon Postgres and add a
  SQLAlchemy session in FastAPI.
- **Auth**: Clerk or Supabase Auth in the Next.js middleware.

## What's ported from the Streamlit app

The pure-Python analytics layer is **identical**: same balance engine, same
fair-value regression, same Black-76 model. Only the presentation layer is
new. To migrate the remaining Streamlit pages (Spreads, Risk, Monte Carlo,
Events, Macro…), copy the relevant function from `commodity_sd_dashboard.py`
into the `commodity_engine` package, expose a new endpoint in `main.py`, and
build the corresponding Next.js route.
