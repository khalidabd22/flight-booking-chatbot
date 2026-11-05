# ✈️ Flight Ticket Booking Chatbot (Python + Streamlit)

Flight search, ticket booking (PNR), status check aur cancel karne wala chatbot (SQLite database).

Ye **Google Gemini** ya **free LLM** (Groq / Ollama) se chalta hai, aur aap apna **data upload ya paste** kar sakte hain.

Sample data: `flights.csv`

## ✨ Features

- **LLM fallback:** Gemini → Groq (free Llama) → Ollama (free, local). Agar ek fail ho jaye (galat key, rate limit, timeout) to agla provider apne aap try hota hai.
- **Data input:** CSV, TSV, Excel (.xlsx), JSON, TXT, MD, PDF upload kar sakte hain, ya CSV/JSON/text seedha paste kar sakte hain (format auto-detect hota hai).
- **Smart context:** Badi files se sirf relevant rows LLM ko bheji jaati hain, saath me schema aur numeric summary bhi.
- **Verified facts:** PNR lookup aur flight search Python me exact hote hain, isliye LLM flights ya prices nahi banata.
- **Error handling:** Galat/khali files, encoding issues, missing keys, rate limits, network errors, aur booking validation (email, seats, sold-out, past date) sab handle hote hain. Inme se kuch bhi ho to app crash nahi hota, sirf saaf message dikhata hai.
- **Booking validation:** Email, seats (1-9), class, sold-out flight aur past date sab check hote hain.
- **Offline tests:** pytest tests bina API key ke chalte hain, aur GitHub Actions CI bhi included hai.

## 📁 Structure

```
flight-booking-chatbot/
├── app.py                 # Streamlit UI
├── bots/
│   ├── base.py            # shared chatbot logic
│   └── flight.py        # is bot ka logic
├── core/
│   ├── config.py          # .env / Streamlit secrets se settings
│   ├── llm.py             # Gemini / Groq / Ollama + fallback + retry
│   ├── data_loader.py     # CSV/Excel/JSON/PDF/paste loader
│   ├── context.py         # relevant rows/chunks chunna
│   ├── booking_db.py      # SQLite flight bookings
│   └── errors.py          # friendly error messages
├── sample_data/           # demo data
├── tests/                 # pytest
├── .env.example           # isko copy karke .env banayein
└── .github/workflows/ci.yml
```

## 🚀 Setup (Windows)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Mac/Linux par: `source venv/bin/activate` aur `cp .env.example .env`.

Ab `.env` file kholein aur kam se kam **ek** key daalein:

- **Gemini (free tier):** https://aistudio.google.com/apikey se key lekar `GEMINI_API_KEY=...` me daalein.
- **Groq (free, Llama models):** https://console.groq.com/keys se key lekar `GROQ_API_KEY=...` me daalein.
- **Ollama (100% free, bina key ke):** https://ollama.com install karein, phir `ollama pull llama3.2` chalayein.

Run:

```bash
streamlit run app.py
```

Browser me http://localhost:8501 khulega. Bina kisi key ke try karna ho to sidebar me **LLM settings → Offline demo** select karein.

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 📤 GitHub par push kaise karein

1. GitHub par naya **empty** repository banayein (README add na karein), jaise `flight-booking-chatbot`.
2. Project folder me ye commands chalayein:

```bash
git init
git add .
git commit -m "Flight Ticket Booking Chatbot"
git branch -M main
git remote add origin https://github.com/<your-username>/flight-booking-chatbot.git
git push -u origin main
```

> ⚠️ `.env` file `.gitignore` me hai, isliye aapki API keys GitHub par **nahi** jaayengi. Push karne se pehle `git status` chalakar confirm kar lein ki `.env` list me nahi dikh rahi.

## ☁️ Free deploy (Streamlit Community Cloud)

1. https://share.streamlit.io par jaakar GitHub repo connect karein. Main file `app.py` rakhein.
2. **Settings → Secrets** me ye daalein:
   ```toml
   GEMINI_API_KEY = "..."
   GROQ_API_KEY = "..."
   LLM_PROVIDER_ORDER = "gemini,groq"
   ```
   Cloud par Ollama nahi chalta, isliye wahan Gemini ya Groq use karein.

## 📝 Apna data format

- `flight_id, origin, destination, date, price` zaroori hain. `airline, departure_time, arrival_time, seats_available` optional hain. Milte-julte column naam (`from`, `to`, `fare`...) bhi chal jaate hain.
