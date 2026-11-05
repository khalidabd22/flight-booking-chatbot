"""Offline tests - no API keys or internet needed (uses the 'mock' provider and fake models)."""
import json

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from bots import FlightBookingBot
from core.booking_db import BookingStore, normalize_flights
from core.config import BASE_DIR, get_settings
from core.context import build_context
from core.data_loader import load_bytes, load_pasted, load_path
from core.errors import BookingError, DataLoadError, LLMError
from core.llm import LLMRouter


@pytest.fixture
def mock_router():
    return LLMRouter(get_settings(provider_order="mock"))


@pytest.fixture
def store():
    return BookingStore(":memory:")


@pytest.fixture
def flights():
    return normalize_flights(load_path(BASE_DIR / "sample_data/flights.csv").df)


# ---------------------------------------------------------------- data loading
def test_all_samples_load():
    for f in (BASE_DIR / "sample_data").glob("*.csv"):
        ds = load_path(f)
        assert ds.df is not None and len(ds.df) > 0


def test_csv_semicolon_and_encoding():
    raw = "name;amount\nCafé;10\nTea;5\n".encode("cp1252")
    ds = load_bytes(raw, "x.csv")
    assert list(ds.df.columns) == ["name", "amount"] and len(ds.df) == 2


def test_json_records_and_nested():
    ds = load_bytes(json.dumps({"records": [{"a": 1}, {"a": 2}]}).encode(), "d.json")
    assert len(ds.df) == 2


@pytest.mark.parametrize("raw,name", [(b"", "a.csv"), (b"abc", "a.exe"), (b"{bad json", "a.json")])
def test_bad_files_raise_friendly(raw, name):
    with pytest.raises(DataLoadError) as e:
        load_bytes(raw, name)
    assert e.value.user_message


def test_pasted_auto_detect():
    assert load_pasted("a,b\n1,2\n3,4").kind == "table"
    assert load_pasted('[{"x": 1}]').kind == "table"
    assert load_pasted("Hello this is a policy note.").kind == "text"
    with pytest.raises(DataLoadError):
        load_pasted("   ")


def test_context_picks_relevant_rows():
    df = pd.DataFrame({"item": [f"thing{i}" for i in range(100)] + ["special-widget"], "v": range(101)})
    ds = load_pasted(df.to_csv(index=False))
    ctx = build_context([ds], "tell me about special-widget", max_chars=5000)
    assert "special-widget" in ctx


# ---------------------------------------------------------------- llm router
class _Boom:
    def invoke(self, _):
        raise RuntimeError("401 API key not valid")


def test_router_falls_back(monkeypatch):
    router = LLMRouter(get_settings(provider_order="gemini,mock", max_retries=0))
    router._models["gemini"] = _Boom()
    reply = router.chat([AIMessage(content="hi")])
    assert reply.provider == "Offline demo"


def test_router_all_fail_gives_friendly_error():
    router = LLMRouter(get_settings(provider_order="gemini", max_retries=0, gemini_api_key=""))
    router.settings.gemini_api_key = ""
    with pytest.raises(LLMError) as e:
        router.chat([AIMessage(content="hi")])
    assert "GEMINI_API_KEY" in e.value.user_message


def test_list_content_from_gemini_is_flattened():
    class Parts:
        def invoke(self, _):
            return AIMessage(content=[{"type": "text", "text": "Hello "}, {"type": "text", "text": "World"}])

    router = LLMRouter(get_settings(provider_order="gemini", max_retries=0))
    router._models["gemini"] = Parts()
    assert router.chat([AIMessage(content="x")]).text == "Hello World"


# ---------------------------------------------------------------- bot
def test_flight_bot_answers(mock_router, store):
    bot = FlightBookingBot(store)
    reply = bot.respond("Delhi to Mumbai flights", [], bot.load_samples(), mock_router)
    assert not reply.is_error


# ---------------------------------------------------------------- booking
def test_booking_flow(store, flights):
    bk = store.create_booking(flights, "AI101", "Rahul Sharma", "rahul@example.com", 2, "Business")
    assert bk["status"] == "CONFIRMED" and bk["total_price"] == 5400 * 2.5 * 2
    assert store.booked_seats("AI101") == 2
    assert store.cancel_booking(bk["pnr"])["status"] == "CANCELLED"
    assert store.booked_seats("AI101") == 0
    with pytest.raises(BookingError):
        store.cancel_booking(bk["pnr"])


@pytest.mark.parametrize("kwargs", [
    dict(flight_id="XX000", passenger_name="Ram", email="r@x.com"),
    dict(flight_id="AI101", passenger_name="R", email="r@x.com"),
    dict(flight_id="AI101", passenger_name="Ram", email="not-an-email"),
    dict(flight_id="AI101", passenger_name="Ram", email="r@x.com", seats=0),
    dict(flight_id="AI101", passenger_name="Ram", email="r@x.com", seat_class="Luxury"),
    dict(flight_id="UK303", passenger_name="Ram", email="r@x.com", seats=9, ),
])
def test_booking_validation(store, flights, kwargs):
    if kwargs["flight_id"] == "UK303":  # 12 seats: fill 9 then 9 more must fail
        store.create_booking(flights, **kwargs)
    with pytest.raises(BookingError):
        store.create_booking(flights, **kwargs)


def test_flight_bot_pnr_and_search_facts(store, flights):
    bot = FlightBookingBot(store)
    bk = store.create_booking(flights, "6E202", "Priya", "p@x.com")
    facts = bot.extra_context(f"status of {bk['pnr']} and Delhi to Mumbai 2026-12-20", bot.load_samples())
    assert bk["pnr"] in facts and "6E202" in facts and "AI101" in facts


def test_normalize_flights_rejects_bad_csv():
    with pytest.raises(BookingError):
        normalize_flights(pd.DataFrame({"foo": [1]}))
