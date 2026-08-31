import datetime
import html
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = "https://ssr.finanstilsynet.no/api/v2/instruments/export-json"
SSR_HOME_URL = "https://ssr.finanstilsynet.no/"
DB_PATH = os.environ.get("SHORTSALG_DB_PATH", "shortsalg.db")
_DB_LOCK = threading.RLock()

_EXEMPT_FALLBACK = [
    {
        "issuerName": "Frontline PLC",
        "isin": "CY0200352116",
        "status": "Unntatt SSR-rapportering",
        "effectiveFrom": "2025-03-31",
    },
    {
        "issuerName": "Golden Ocean Group",
        "isin": "BMG396372051",
        "status": "Unntatt SSR-rapportering",
        "effectiveFrom": "2025-03-31",
    },
    {
        "issuerName": "Clean Seas Seafood Limited",
        "isin": "AU000000CSS3",
        "status": "Unntatt SSR-rapportering",
        "effectiveFrom": "2025-03-31",
    },
]


def _to_iso_date(value):
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return str(value)
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def _standardiser_shortpercent(value):
    """Normaliserer API-verdien til prosentpoeng, f.eks. 58 -> 0,58."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None

    if x > 20:
        return x / 100
    return x


def _get_first(data, candidates, default=None):
    if not isinstance(data, dict):
        return default
    lower_map = {str(key).lower(): value for key, value in data.items()}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return default


def _normaliser_unntatte_instrumenter(page_text):
    """Leser Finanstilsynets publiserte liste over aksjer unntatt SSR-rapportering."""
    columns = ["issuerName", "isin", "status", "effectiveFrom"]
    if not page_text:
        return pd.DataFrame(_EXEMPT_FALLBACK, columns=columns)

    section = re.search(
        r"Exempted shares from\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4}).*?<p>(.*?)</p>",
        page_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not section:
        return pd.DataFrame(_EXEMPT_FALLBACK, columns=columns)

    day, month_name, year, body = section.groups()
    month_lookup = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    month = month_lookup.get(month_name.lower())
    effective_from = (
        f"{int(year):04d}-{month:02d}-{int(day):02d}" if month else "2025-03-31"
    )

    clean_body = html.unescape(re.sub(r"<[^>]+>", " ", body))
    clean_body = re.sub(r"\s+", " ", clean_body).strip()
    if ":" in clean_body:
        clean_body = clean_body.split(":", 1)[1]

    rows = []
    for issuer, isin in re.findall(
        r"([^,]+?)\s*\(([A-Z]{2}[A-Z0-9]{10})\)", clean_body
    ):
        issuer = re.sub(r"^\s*(?:and\s+)?", "", issuer, flags=re.IGNORECASE).strip(" .")
        if issuer:
            rows.append(
                {
                    "issuerName": issuer,
                    "isin": isin,
                    "status": "Unntatt SSR-rapportering",
                    "effectiveFrom": effective_from,
                }
            )

    return pd.DataFrame(rows or _EXEMPT_FALLBACK, columns=columns).drop_duplicates(
        subset=["isin"]
    )


def _normaliser_payload(data):
    rows = []
    columns = ["isin", "issuerName", "positionHolder", "date", "shortPercent", "shares"]

    if not isinstance(data, list):
        return pd.DataFrame(columns=columns)

    for instrument in data:
        if not isinstance(instrument, dict):
            continue

        isin = _get_first(instrument, ["isin", "instrumentIsin"])
        issuer = _get_first(instrument, ["issuerName", "issuer", "instrumentName"])
        instrument_holder = _get_first(
            instrument,
            ["positionHolder", "positionHolderName", "holderName", "positionOwner", "ownerName", "holder"],
        )

        events = instrument.get("events", [])
        if not isinstance(events, list):
            continue

        for event in events:
            if not isinstance(event, dict):
                continue

            holder = _get_first(
                event,
                ["positionHolder", "positionHolderName", "holderName", "positionOwner", "ownerName", "holder"],
                default=instrument_holder,
            )

            row = {
                "isin": isin or _get_first(event, ["isin", "instrumentIsin"]),
                "issuerName": issuer or _get_first(event, ["issuerName", "issuer"]),
                "positionHolder": holder,
                "date": _to_iso_date(_get_first(event, ["date", "positionDate", "disclosureDate"])),
                "shortPercent": _standardiser_shortpercent(
                    _get_first(event, ["shortPercent", "netShortPosition", "positionPercent", "percent"])
                ),
                "shares": _get_first(event, ["shares", "shortPosition", "position", "numberOfShares"]),
            }

            if row["issuerName"] and row["date"] and row["shortPercent"] is not None:
                rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df["shortPercent"] = pd.to_numeric(df["shortPercent"], errors="coerce")
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
        df = df.dropna(subset=["issuerName", "date", "shortPercent"])
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def _normaliser_posisjonsholdere(data):
    """
    Lager ett separat datasett med individuelle offentlige shortposisjoner.
    Leser event["activePositions"] og holder dette adskilt fra de aggregerte
    event-radene, slik at eksisterende grafer og summer ikke dobbeltteller.
    """
    rows = []
    columns = ["isin", "issuerName", "positionHolder", "date", "shortPercent", "shares"]

    if not isinstance(data, list):
        return pd.DataFrame(columns=columns)

    for instrument in data:
        if not isinstance(instrument, dict):
            continue

        isin = _get_first(instrument, ["isin", "instrumentIsin"])
        issuer = _get_first(instrument, ["issuerName", "issuer", "instrumentName"])
        events = instrument.get("events", [])
        if not isinstance(events, list):
            continue

        for event in events:
            if not isinstance(event, dict):
                continue

            event_date = _to_iso_date(_get_first(event, ["date", "positionDate", "disclosureDate"]))
            active_positions = event.get("activePositions", [])
            if not isinstance(active_positions, list):
                continue

            for position in active_positions:
                if not isinstance(position, dict):
                    continue

                holder = _get_first(
                    position,
                    ["positionHolder", "positionHolderName", "holderName", "positionOwner", "ownerName", "holder"],
                )
                row = {
                    "isin": isin or _get_first(position, ["isin", "instrumentIsin"]),
                    "issuerName": issuer or _get_first(position, ["issuerName", "issuer"]),
                    "positionHolder": holder,
                    "date": _to_iso_date(
                        _get_first(position, ["date", "positionDate", "disclosureDate"], default=event_date)
                    ),
                    "shortPercent": _standardiser_shortpercent(
                        _get_first(position, ["shortPercent", "netShortPosition", "positionPercent", "percent"])
                    ),
                    "shares": _get_first(position, ["shares", "shortPosition", "position", "numberOfShares"]),
                }

                if row["issuerName"] and row["positionHolder"] and row["date"] and row["shortPercent"] is not None:
                    rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df["shortPercent"] = pd.to_numeric(df["shortPercent"], errors="coerce")
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
        df = df.dropna(subset=["issuerName", "positionHolder", "date", "shortPercent"])
        df = df.drop_duplicates().reset_index(drop=True)
    return df


@st.cache_data(ttl=3600, max_entries=1, show_spinner=False)
def _hent_api_payload(max_retries=3):
    """Henter rå JSON én gang per time og deler samme payload mellom datasett."""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(
                API_URL,
                timeout=(15, 120),
                headers={"User-Agent": "shortsalg-register/2.1"},
            )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list) or not data:
                raise ValueError("API-et svarte, men payloaden var tom eller ugyldig.")
            return data
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 + attempt)

    print(f"Klarte ikke hente data fra Finanstilsynet: {last_error}")
    return []


@st.cache_data(ttl=3600, max_entries=1, show_spinner=False)
def hent_fullt_register(max_retries=3):
    """Returnerer kun aggregerte event-rader for eksisterende analyser og grafer."""
    data = _hent_api_payload(max_retries=max_retries)
    df = _normaliser_payload(data)
    if df.empty:
        print("API-et svarte, men parseren fant ingen gyldige aggregerte rader.")
    return df


@st.cache_data(ttl=3600, max_entries=1, show_spinner=False)
def hent_posisjonsholdere(max_retries=3):
    """Returnerer individuelle offentlige shortposisjoner fra activePositions."""
    data = _hent_api_payload(max_retries=max_retries)
    df = _normaliser_posisjonsholdere(data)
    if df.empty:
        print("Ingen individuelle posisjonsholdere ble funnet i activePositions.")
    return df


@st.cache_data(ttl=3600, max_entries=1, show_spinner=False)
def hent_unntatte_instrumenter(max_retries=3):
    """Henter aksjer som Finanstilsynet uttrykkelig har unntatt SSR-rapportering."""
    for attempt in range(max_retries):
        try:
            response = requests.get(
                SSR_HOME_URL,
                timeout=(10, 30),
                headers={"User-Agent": "shortsalg-register/2.2"},
            )
            response.raise_for_status()
            return _normaliser_unntatte_instrumenter(response.text)
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)
            else:
                print(f"Klarte ikke hente SSR-unntakslisten: {exc}")

    return pd.DataFrame(_EXEMPT_FALLBACK)


def tving_ny_nedlasting():
    """Tømmer delte API-cacher. Neste kall laster data på nytt."""
    _hent_api_payload.clear()
    hent_fullt_register.clear()
    hent_posisjonsholdere.clear()
    hent_unntatte_instrumenter.clear()


def _connect(db_path=DB_PATH):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _ensure_schema(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS short_positions (
            isin TEXT,
            issuerName TEXT,
            positionHolder TEXT,
            date TEXT,
            shortPercent REAL,
            shares REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS updates_log (
            timestamp TEXT,
            new_rows INTEGER
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_short_issuer_date ON short_positions (issuerName, date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_short_isin ON short_positions (isin)"
    )
    conn.commit()


@st.cache_data(ttl=300, max_entries=1, show_spinner=False)
def hent_database_data(db_path=DB_PATH):
    """Leser SQLite-data én gang per fem minutter, delt mellom brukerne."""
    try:
        with _DB_LOCK:
            conn = _connect(db_path)
            _ensure_schema(conn)
            df = pd.read_sql_query(
                "SELECT isin, issuerName, positionHolder, date, shortPercent, shares FROM short_positions",
                conn,
            )
            conn.close()
        return df
    except Exception as exc:
        print(f"Feil ved lesing av database: {exc}")
        return pd.DataFrame(columns=["isin", "issuerName", "positionHolder", "date", "shortPercent", "shares"])


def _clear_database_cache():
    hent_database_data.clear()


def lagre_i_database(df, db_path=DB_PATH):
    """Lagrer bare nye rader. Skriving serialiseres for å unngå SQLite-låsing."""
    if df is None or df.empty:
        return 0

    columns = ["isin", "issuerName", "positionHolder", "date", "shortPercent", "shares"]
    clean = df.copy()
    for column in columns:
        if column not in clean.columns:
            clean[column] = None
    clean = clean[columns].drop_duplicates()

    with _DB_LOCK:
        conn = _connect(db_path)
        _ensure_schema(conn)
        try:
            existing = pd.read_sql_query(
                "SELECT isin, issuerName, positionHolder, date, shortPercent, shares FROM short_positions",
                conn,
            )

            compare_cols = columns
            left = clean.copy()
            right = existing.copy()
            for column in ["isin", "issuerName", "positionHolder", "date"]:
                left[column] = left[column].fillna("").astype(str)
                right[column] = right[column].fillna("").astype(str)
            for column in ["shortPercent", "shares"]:
                left[column] = pd.to_numeric(left[column], errors="coerce")
                right[column] = pd.to_numeric(right[column], errors="coerce")

            if not right.empty:
                marker = right.drop_duplicates(compare_cols)
                merged = left.merge(marker, on=compare_cols, how="left", indicator=True)
                new_mask = merged["_merge"].eq("left_only").to_numpy()
                clean = clean.loc[new_mask].copy()

            if clean.empty:
                new_rows = 0
            else:
                clean.to_sql("short_positions", conn, if_exists="append", index=False, method="multi", chunksize=1000)
                new_rows = len(clean)

            pd.DataFrame(
                [{
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "new_rows": int(new_rows),
                }]
            ).to_sql("updates_log", conn, if_exists="append", index=False)
            conn.commit()
        finally:
            conn.close()

    _clear_database_cache()
    return int(new_rows)


def hent_siste_oppdatering(db_path=DB_PATH):
    try:
        with _DB_LOCK:
            conn = _connect(db_path)
            _ensure_schema(conn)
            row = conn.execute(
                "SELECT timestamp FROM updates_log ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            total = conn.execute("SELECT COUNT(*) FROM short_positions").fetchone()[0]
            conn.close()
        return (row[0] if row else None), int(total)
    except Exception as exc:
        print(f"Feil ved henting av oppdateringsinfo: {exc}")
        return None, 0
