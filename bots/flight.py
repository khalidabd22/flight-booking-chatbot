import re

import pandas as pd

from bots.base import BaseBot
from core.booking_db import PNR_RE, BookingStore, normalize_flights
from core.context import tokens
from core.errors import BookingError


class FlightBookingBot(BaseBot):
    key = "flight"
    name = "Flight Ticket Booking"
    icon = "✈️"
    description = "Flights search, book, PNR status, cancel - apna flights CSV bhi upload kar sakte hain."
    sample_files = ["sample_data/flights.csv"]
    placeholder = "e.g. Delhi se Mumbai 2026-12-20 ki sabse sasti flight? / PNR AB12CD ka status?"

    system_prompt = (
        "You are a friendly airline booking assistant.\n"
        "- Help users find flights, compare price/time, explain baggage/cancellation basics, and check PNR status.\n"
        "- Only mention flights that appear in VERIFIED FACTS or DATA CONTEXT; never invent flights, prices or PNRs.\n"
        "- To actually book or cancel, tell the user to use the 'Book / Manage' tab (it needs name, email, seats and class) "
        "and mention the exact Flight ID to choose.\n"
        "- Prices are per seat in Economy; Premium Economy is 1.4x and Business is 2.5x."
    )

    def __init__(self, store: BookingStore | None = None):
        self.store = store

    def flights_df(self, datasets) -> pd.DataFrame | None:
        for ds in datasets:
            if ds.df is None:
                continue
            try:
                return normalize_flights(ds.df)
            except BookingError:
                continue
        return None

    def extra_context(self, query, datasets):
        if self.store is None:
            return ""
        lines = []
        for pnr in set(PNR_RE.findall(query.upper())):
            if not re.search(r"\d", pnr) or not re.search(r"[A-Z]", pnr):
                continue  # skip plain words / numbers
            booking = self.store.get_booking(pnr)
            lines.append(f"PNR {pnr}: {booking}" if booking else f"PNR {pnr}: NOT FOUND")

        flights = self.flights_df(datasets)
        if flights is not None:
            q = tokens(query)
            cities = {c for c in pd.concat([flights["origin"], flights["destination"]]).unique()}
            mentioned = [c for c in cities if c.lower() in query.lower() or any(t == c.lower() for t in q)]
            date = re.search(r"\d{4}-\d{2}-\d{2}", query)
            if mentioned:
                # order by position in the sentence: first city = origin
                mentioned.sort(key=lambda c: query.lower().find(c.lower()))
                origin = mentioned[0]
                dest = mentioned[1] if len(mentioned) > 1 else ""
                res = self.store.search(flights, origin, dest, date.group(0) if date else "")
                if res.empty and dest:  # maybe the user wrote destination first
                    res = self.store.search(flights, dest, origin, date.group(0) if date else "")
                cols = ["flight_id", "airline", "origin", "destination", "date", "departure_time", "arrival_time", "price", "seats_left"]
                if res.empty:
                    lines.append(f"Search {origin} -> {dest or 'any'} {date.group(0) if date else ''}: no flights found.")
                else:
                    lines.append(f"Search results ({len(res)}):\n" + res[[c for c in cols if c in res.columns]].head(15).to_csv(index=False))
        return "\n".join(lines)
