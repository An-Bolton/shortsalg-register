# ssr_api_med_shortere_final.py
import streamlit as st
import requests
import pandas as pd
import sqlite3
import datetime
import time


API_URL = "https://ssr.finanstilsynet.no/api/v2/instruments/export-json"


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
    try:
        x = float(value)
    except Exception:
        return None

    if x <= 1.5:
        return x * 100
    if x > 100:
        return x / 100
    return x


def _get_first(d, candidates, default=None):
    if not isinstance(d, dict):
        return default
    lower_map = {str(k).lower(): v for k, v in d.items()}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return default


@st.cache_data(ttl=3600)
def _last_ned_data(max_retries=3):
    """
    Henter rådata fra Finanstilsynets instruments-endpoint.
    """
    print(f"Henter data fra {API_URL} ...")

    for attempt in range(max_retries):
        try:
            response = requests.get(API_URL, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Forsøk {attempt + 1}/{max_retries} feilet: {e}")
            time.sleep(3)

    return []


def _normaliser_payload(data):
    """
    Parser både aggregert og detaljert info fra instruments/export-json.

    Viktig:
    API-dokumentasjon ser ut til å indikere at event-objektene kan inneholde
    positionHolder. Derfor prøver vi både toppnivå og event-nivå.
    """
    rows = []

    if not isinstance(data, list):
        return pd.DataFrame(columns=["isin", "issuerName", "positionHolder", "date", "shortPercent", "shares"])

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

    return pd.DataFrame(rows)


def hent_fullt_register(max_retries=3, _progress_callback=None):
    """
    Henter shortregisteret fra Finanstilsynet.
    """
    if _progress_callback:
        _progress_callback(0.05, 0, 1)

    data = _last_ned_data(max_retries=max_retries)

    if not data:
        st.error("Klarte ikke hente data fra Finanstilsynet.")
        return pd.DataFrame()

    if _progress_callback:
        _progress_callback(0.85, 1, 1)

    df = _normaliser_payload(data)

    if df.empty:
        st.warning("Data ble hentet, men parseren fant ingen gyldige rader.")
        return df

    if _progress_callback:
        _progress_callback(1.0, 1, 1)

    print(f"Totalt {len(df):,} rader hentet.")
    print("Kolonner:", df.columns.tolist())
    return df


def _ensure_short_positions_schema(conn):
    """
    Sørger for at short_positions-tabellen har riktig struktur.
    Legger til manglende kolonner hvis databasen er fra en eldre versjon.
    """
    cur = conn.cursor()
    cur.execute(
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
    conn.commit()

    existing_cols = [row[1] for row in cur.execute("PRAGMA table_info(short_positions)").fetchall()]
    wanted = {
        "isin": "TEXT",
        "issuerName": "TEXT",
        "positionHolder": "TEXT",
        "date": "TEXT",
        "shortPercent": "REAL",
        "shares": "REAL",
    }

    for col, col_type in wanted.items():
        if col not in existing_cols:
            cur.execute(f"ALTER TABLE short_positions ADD COLUMN {col} {col_type}")
    conn.commit()


def lagre_i_database(df, db_path="shortsalg.db"):
    """Lagrer DataFrame i SQLite-database."""
    if df.empty:
        print("Ingen data å lagre.")
        return

    df = df.copy()
    if "positionHolder" not in df.columns:
        df["positionHolder"] = None

    conn = sqlite3.connect(db_path)
    _ensure_short_positions_schema(conn)

    try:
        eksisterende = pd.read_sql(
            "SELECT isin, issuerName, positionHolder, date, shortPercent, shares FROM short_positions",
            conn,
        )
    except Exception:
        eksisterende = pd.DataFrame(columns=["isin", "issuerName", "positionHolder", "date", "shortPercent", "shares"])

    if not eksisterende.empty:
        df_cmp = df.copy()
        ex_cmp = eksisterende.copy()

        for col in ["isin", "issuerName", "positionHolder", "date"]:
            df_cmp[col] = df_cmp[col].fillna("")
            ex_cmp[col] = ex_cmp[col].fillna("")

        merged = df_cmp.merge(
            ex_cmp,
            on=["isin", "issuerName", "positionHolder", "date", "shortPercent", "shares"],
            how="left",
            indicator=True,
        )
        df = df.loc[merged["_merge"] == "left_only"].copy()

    if not df.empty:
        df.to_sql("short_positions", conn, if_exists="append", index=False)
        print(f"Lagret {len(df)} nye rader i databasen {db_path}.")
    else:
        print("Ingen nye rader å lagre.")

    conn.close()


def oppdater_database_automatisk(db_path="shortsalg.db"):
    """Oppdaterer databasen og logger tidspunkt + antall nye rader."""
    print("🔄 Starter automatisk oppdatering av shortregisteret...")
    df_ny = hent_fullt_register()

    if df_ny.empty:
        print("Ingen nye data tilgjengelig.")
        return

    conn = sqlite3.connect(db_path)
    _ensure_short_positions_schema(conn)

    try:
        df_gammel = pd.read_sql("SELECT * FROM short_positions", conn)
    except Exception:
        df_gammel = pd.DataFrame()

    if not df_gammel.empty:
        nøkkelkolonner = [c for c in ["isin", "issuerName", "positionHolder", "date", "shortPercent", "shares"] if c in df_ny.columns and c in df_gammel.columns]
        gammel = df_gammel[nøkkelkolonner].drop_duplicates().copy()
        ny = df_ny.copy()

        for col in ["isin", "issuerName", "positionHolder", "date"]:
            if col in ny.columns:
                ny[col] = ny[col].fillna("")
            if col in gammel.columns:
                gammel[col] = gammel[col].fillna("")

        merged = ny.merge(gammel, on=nøkkelkolonner, how="left", indicator=True)
        df_ny = df_ny.loc[merged["_merge"] == "left_only"].copy()

    antall_nye = len(df_ny)
    if antall_nye > 0:
        df_ny.to_sql("short_positions", conn, if_exists="append", index=False)
        print(f"✅ Lagret {antall_nye} nye rader ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    else:
        print("🟡 Ingen nye rader å lagre.")

    log_df = pd.DataFrame([{
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "new_rows": antall_nye
    }])
    log_df.to_sql("updates_log", conn, if_exists="append", index=False)
    conn.close()
    print(f"Logget oppdatering: {antall_nye} nye rader.")


def hent_siste_oppdatering(db_path="shortsalg.db"):
    """Henter siste oppdateringstidspunkt og totalt antall rader i databasen."""
    try:
        conn = sqlite3.connect(db_path)

        siste_tid = None
        try:
            df_log = pd.read_sql("SELECT * FROM updates_log ORDER BY timestamp DESC LIMIT 1", conn)
            if not df_log.empty:
                siste_tid = df_log.iloc[0]["timestamp"]
        except Exception:
            pass

        try:
            total = pd.read_sql("SELECT COUNT(*) as antall FROM short_positions", conn).iloc[0]["antall"]
        except Exception:
            total = 0

        conn.close()
        return siste_tid, total

    except Exception as e:
        print(f"Feil ved henting av oppdateringsinfo: {e}")
        return None, 0
