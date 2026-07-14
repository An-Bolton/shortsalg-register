import datetime
import os
import sqlite3
import threading
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = "https://ssr.finanstilsynet.no/api/v2/instruments/export-json"
DB_PATH = os.environ.get("SHORTSALG_DB_PATH", "shortsalg.db")
_DB_LOCK = threading.RLock()


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


@st.cache_resource(ttl=3600, max_entries=1, show_spinner=False)
def hent_fullt_register(max_retries=3):
    """
    Henter og normaliserer hele registeret én gang per time, delt mellom alle brukere.
    DataFrame-en skal behandles som skrivebeskyttet i appen.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(
                API_URL,
                timeout=(15, 120),
                headers={"User-Agent": "shortsalg-register/2.0"},
            )
            response.raise_for_status()
            df = _normaliser_payload(response.json())
            if df.empty:
                raise ValueError("API-et svarte, men parseren fant ingen gyldige rader.")
            return df
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 + attempt)

    print(f"Klarte ikke hente data fra Finanstilsynet: {last_error}")
    return pd.DataFrame(columns=["isin", "issuerName", "positionHolder", "date", "shortPercent", "shares"])


def tving_ny_nedlasting():
    """Tømmer den delte API-cachen. Neste kall laster data på nytt."""
    hent_fullt_register.clear()


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


@st.cache_resource(ttl=300, max_entries=1, show_spinner=False)
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
