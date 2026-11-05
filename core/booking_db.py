"""SQLite booking store for the flight bot. Flights come from a DataFrame (sample or uploaded CSV);
bookings are persisted in SQLite so they survive app restarts."""
from __future__ import annotations

import random
import re
import sqlite3
import string
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.context import find_column
from core.errors import BookingError

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PNR_RE = re.compile(r"\b[A-Z0-9]{6}\b")

SCHEMA = """
CREATE TABLE IF NOT EXISTS bookings (
    pnr TEXT PRIMARY KEY,
    flight_id TEXT NOT NULL,
    passenger_name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT,
    seats INTEGER NOT NULL,
    seat_class TEXT NOT NULL,
    total_price REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'CONFIRMED',
    created_at TEXT NOT NULL
);
"""

REQUIRED = {
    "flight_id": ("flight_id", "flight_no", "flight_number", "flight"),
    "origin": ("origin", "from", "source", "departure_city"),
    "destination": ("destination", "to", "dest", "arrival_city"),
    "date": ("date", "departure_date", "travel_date"),
    "price": ("price", "fare", "amount", "cost"),
}


def normalize_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Map any reasonable column names to the standard ones. Raises BookingError if impossible."""
    if df is None or df.empty:
        raise BookingError("Flights data khali hai.")
    mapping, missing = {}, []
    for std, candidates in REQUIRED.items():
        col = find_column(df, *candidates)
        if col is None:
            missing.append(std)
        else:
            mapping[col] = std
    if missing:
        raise BookingError(
            "Flights CSV me ye columns chahiye: flight_id, origin, destination, date, price "
            f"(missing: {', '.join(missing)})."
        )
    out = df.rename(columns=mapping).copy()
    for opt, candidates in {"airline": ("airline", "carrier"), "departure_time": ("departure_time", "dep_time", "time"),
                            "arrival_time": ("arrival_time", "arr_time"), "seats_available": ("seats_available", "seats", "available")}.items():
        if opt not in out.columns:
            col = find_column(df, *candidates)
            out[opt] = df[col] if col and col not in mapping else ("" if opt != "seats_available" else 50)
    out["flight_id"] = out["flight_id"].astype(str).str.strip().str.upper()
    out["origin"] = out["origin"].astype(str).str.strip()
    out["destination"] = out["destination"].astype(str).str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["price"] = pd.to_numeric(out["price"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
    out["seats_available"] = pd.to_numeric(out["seats_available"], errors="coerce").fillna(0).astype(int)
    out = out.dropna(subset=["date", "price"])
    if out.empty:
        raise BookingError("Flights data me valid date/price wali koi row nahi mili.")
    return out.drop_duplicates("flight_id").reset_index(drop=True)


class BookingStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False) if db_path == ":memory:" else None
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = self._mem_conn or sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise BookingError(f"Database error: {exc}") from exc
        finally:
            if conn is not self._mem_conn:
                conn.close()

    # ------------------------------------------------------------------
    def booked_seats(self, flight_id: str) -> int:
        with self._conn() as c:
            row = c.execute("SELECT COALESCE(SUM(seats),0) AS n FROM bookings WHERE flight_id=? AND status='CONFIRMED'",
                            (flight_id.upper(),)).fetchone()
            return int(row["n"])

    def with_availability(self, flights: pd.DataFrame) -> pd.DataFrame:
        f = flights.copy()
        f["seats_left"] = [max(0, int(s) - self.booked_seats(fid)) for fid, s in zip(f["flight_id"], f["seats_available"])]
        return f

    def search(self, flights: pd.DataFrame, origin: str = "", destination: str = "", date: str = "") -> pd.DataFrame:
        f = self.with_availability(flights)
        if origin:
            f = f[f["origin"].str.lower().str.contains(re.escape(origin.strip().lower()), na=False)]
        if destination:
            f = f[f["destination"].str.lower().str.contains(re.escape(destination.strip().lower()), na=False)]
        if date:
            f = f[f["date"] == str(date)]
        return f.sort_values(["date", "price"]).reset_index(drop=True)

    def _new_pnr(self, conn) -> str:
        for _ in range(20):
            # always mix letters + digits so the chat can recognise it as a PNR
            chars = random.choices(string.ascii_uppercase, k=3) + random.choices(string.digits, k=3)
            random.shuffle(chars)
            pnr = "".join(chars)
            if not conn.execute("SELECT 1 FROM bookings WHERE pnr=?", (pnr,)).fetchone():
                return pnr
        raise BookingError("PNR generate nahi ho paya, dobara try karein.")

    def create_booking(self, flights: pd.DataFrame, flight_id: str, passenger_name: str, email: str,
                       seats: int = 1, seat_class: str = "Economy", phone: str = "") -> dict:
        name = (passenger_name or "").strip()
        email = (email or "").strip()
        if len(name) < 2:
            raise BookingError("Passenger ka poora naam daalein.")
        if not EMAIL_RE.match(email):
            raise BookingError("Valid email address daalein.")
        try:
            seats = int(seats)
        except (TypeError, ValueError) as exc:
            raise BookingError("Seats number hona chahiye.") from exc
        if not 1 <= seats <= 9:
            raise BookingError("Ek booking me 1 se 9 seats tak book kar sakte hain.")
        multiplier = {"economy": 1.0, "premium economy": 1.4, "business": 2.5}.get(seat_class.lower())
        if multiplier is None:
            raise BookingError("Class Economy, Premium Economy ya Business honi chahiye.")

        match = flights[flights["flight_id"] == str(flight_id).strip().upper()]
        if match.empty:
            raise BookingError(f"Flight {flight_id} nahi mili.")
        flight = match.iloc[0]
        try:
            if datetime.strptime(flight["date"], "%Y-%m-%d").date() < datetime.now().date():
                raise BookingError("Yeh flight ki date nikal chuki hai.")
        except ValueError:
            pass

        with self._conn() as c:
            booked = c.execute("SELECT COALESCE(SUM(seats),0) AS n FROM bookings WHERE flight_id=? AND status='CONFIRMED'",
                               (flight["flight_id"],)).fetchone()["n"]
            left = int(flight["seats_available"]) - int(booked)
            if seats > left:
                raise BookingError(f"Sirf {max(left, 0)} seats bachi hain flight {flight['flight_id']} par.")
            pnr = self._new_pnr(c)
            total = round(float(flight["price"]) * multiplier * seats, 2)
            c.execute(
                "INSERT INTO bookings (pnr, flight_id, passenger_name, email, phone, seats, seat_class, total_price, status, created_at)"
                " VALUES (?,?,?,?,?,?,?,?, 'CONFIRMED', ?)",
                (pnr, flight["flight_id"], name, email, phone.strip(), seats, seat_class, total, datetime.now().isoformat(timespec="seconds")),
            )
        return self.get_booking(pnr)

    def get_booking(self, pnr: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM bookings WHERE pnr=?", ((pnr or "").strip().upper(),)).fetchone()
            return dict(row) if row else None

    def cancel_booking(self, pnr: str) -> dict:
        booking = self.get_booking(pnr)
        if not booking:
            raise BookingError(f"PNR {pnr} nahi mila.")
        if booking["status"] == "CANCELLED":
            raise BookingError(f"PNR {booking['pnr']} pehle se cancel hai.")
        with self._conn() as c:
            c.execute("UPDATE bookings SET status='CANCELLED' WHERE pnr=?", (booking["pnr"],))
        return self.get_booking(booking["pnr"])

    def list_bookings(self, email: str = "") -> pd.DataFrame:
        with self._conn() as c:
            if email:
                rows = c.execute("SELECT * FROM bookings WHERE lower(email)=lower(?) ORDER BY created_at DESC", (email.strip(),)).fetchall()
            else:
                rows = c.execute("SELECT * FROM bookings ORDER BY created_at DESC LIMIT 200").fetchall()
        return pd.DataFrame([dict(r) for r in rows])
