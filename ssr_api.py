# ssr_api.py
import streamlit as st
import requests
import pandas as pd
import sqlite3
import datetime
import time
import json


# ---------- 1. CACHE SELVE API-KALLET ----------
@st.cache_data(ttl=3600)
def _last_ned_data(max_retries=3):
    """
    Denne funksjonen henter data fra API og returnerer JSON.
    Den inneholder INGEN Streamlit-elementer (kun ren logikk).
    """
    url = "https://ssr.finanstilsynet.no/api/v2/instruments/export-json"
    print(f"Henter data fra {url} ...")

    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                data = r.json()
                return data
        except Exception as e:
            print(f"Forsøk {attempt+1}/{max_retries} feilet: {e}")
            time.sleep(5)
    return []


# ---------- 2. KALL SOM STYRER UI OG FREMGANG ----------
def hent_fullt_register(max_retries=3, _progress_callback=None):
    """
    Henter shortregisteret fra Finanstilsynet.
    Viser fremdrift og bruker cache for selve nedlastingen.
    """
    # Simulert progressbar under lasting
    if _progress_callback:
        _progress_callback(0.05, 0, 1)

    data = _last_ned_data(max_retries=max_retries)

    if not data:
        st.error("Klarte ikke hente data fra Finanstilsynet.")
        return pd.DataFrame()

    # Simulert “laster ferdig”
    if _progress_callback:
        _progress_callback(0.95, 1, 1)

    # Konverter JSON → DataFrame
    rows = []
    for instrument in data:
        isin = instrument.get("isin")
        issuer = instrument.get("issuerName")
        for event in instrument.get("events", []):
            rows.append({
                "isin": isin,
                "issuerName": issuer,
                "date": event.get("date"),
                "shortPercent": event.get("shortPercent"),
                "shares": event.get("shares")
            })

    df = pd.DataFrame(rows)

    if _progress_callback:
        _progress_callback(1.0, 1, 1)

    print(f"Totalt {len(df):,} rader hentet.")
    return df


# ---------- 3. DATABASEFUNKSJONER ----------
def lagre_i_database(df, db_path="shortsalg.db"):
    """Lagrer DataFrame i SQLite-database"""
    if df.empty:
        print("Ingen data å lagre.")
        return

    conn = sqlite3.connect(db_path)
    df.to_sql("short_positions", conn, if_exists="append", index=False)
    conn.close()
    print(f"Lagret {len(df)} rader i databasen {db_path}.")


def oppdater_database_automatisk(db_path="shortsalg.db"):
    """Oppdaterer databasen og logger tidspunkt + antall nye rader."""
    print("🔄 Starter automatisk oppdatering av shortregisteret...")
    df_ny = hent_fullt_register()

    if df_ny.empty:
        print("Ingen nye data tilgjengelig.")
        return

    conn = sqlite3.connect(db_path)
    try:
        df_gammel = pd.read_sql("SELECT * FROM short_positions", conn)
    except Exception:
        df_gammel = pd.DataFrame()

    # Finn nye rader basert på ISIN + Dato
    if not df_gammel.empty:
        kombinasjon_gammel = set(zip(df_gammel["isin"], df_gammel["date"]))
        df_ny = df_ny[~df_ny.apply(lambda r: (r["isin"], r["date"]) in kombinasjon_gammel, axis=1)]

    antall_nye = len(df_ny)
    if antall_nye > 0:
        df_ny.to_sql("short_positions", conn, if_exists="append", index=False)
        print(f"✅ Lagret {antall_nye} nye rader ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    else:
        print("🟡 Ingen nye rader å lagre.")

    # Loggfør oppdateringen
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

        # Hent siste oppdatering fra log-tabellen
        siste_tid = None
        try:
            df_log = pd.read_sql("SELECT * FROM updates_log ORDER BY timestamp DESC LIMIT 1", conn)
            if not df_log.empty:
                siste_tid = df_log.iloc[0]["timestamp"]
        except Exception:
            pass

        # Tell antall rader totalt
        try:
            total = pd.read_sql("SELECT COUNT(*) as antall FROM short_positions", conn).iloc[0]["antall"]
        except Exception:
            total = 0

        conn.close()
        return siste_tid, total

    except Exception as e:
        print(f"Feil ved henting av oppdateringsinfo: {e}")
        return None, 0
