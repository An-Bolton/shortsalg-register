import pandas as pd
import plotly.express as px
import streamlit as st

from ssr_api import (
    hent_database_data,
    hent_fullt_register,
    hent_siste_oppdatering,
    lagre_i_database,
    tving_ny_nedlasting,
)

# -------------------- DATAHJELPERE --------------------

def _standardiser_shortpercent(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "shortPercent" not in df.columns:
        return df
    out = df.copy()
    out["shortPercent"] = pd.to_numeric(out["shortPercent"], errors="coerce")
    maximum = out["shortPercent"].max(skipna=True)
    if pd.notna(maximum) and maximum > 20:
        out["shortPercent"] = out["shortPercent"] / 100
    return out


def _agg_issuer_date(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = _standardiser_shortpercent(df)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["issuerName", "date", "shortPercent"])
    return (
        out.groupby(["issuerName", "date"], as_index=False)["shortPercent"]
        .sum()
        .sort_values(["issuerName", "date"])
    )


def beregn_storste_endringer(df: pd.DataFrame) -> pd.DataFrame:
    data = _agg_issuer_date(df)
    if data.empty:
        return data
    data["forrige_short"] = data.groupby("issuerName")["shortPercent"].shift(1)
    latest = data.groupby("issuerName").tail(1).copy()
    latest["endring"] = latest["shortPercent"] - latest["forrige_short"]
    latest = latest.dropna(subset=["endring"])
    return latest.reindex(latest["endring"].abs().sort_values(ascending=False).index)


def finn_nye_shortposisjoner(df: pd.DataFrame, terskel: float = 0.5) -> pd.DataFrame:
    data = _agg_issuer_date(df)
    if data.empty:
        return data
    data["forrige_short"] = data.groupby("issuerName")["shortPercent"].shift(1)
    latest = data.groupby("issuerName").tail(1).copy()
    result = latest[
        (latest["shortPercent"] >= terskel)
        & (latest["forrige_short"].isna() | (latest["forrige_short"] < terskel))
    ].copy()
    return result.sort_values(["date", "shortPercent"], ascending=[False, False])


@st.cache_data(ttl=600, max_entries=4, show_spinner=False)
def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def vis_hurtiginnsikt(df: pd.DataFrame, expanded: bool = False) -> None:
    with st.expander("Hurtig-innsikt: største endringer og nye posisjoner", expanded=expanded):
        left, right = st.columns(2)
        with left:
            st.markdown("### Største endringer")
            changes = beregn_storste_endringer(df)
            if changes.empty:
                st.info("Ingen endringer å vise.")
            else:
                st.dataframe(
                    changes[["issuerName", "shortPercent", "endring", "date"]]
                    .rename(columns={
                        "issuerName": "Selskap",
                        "shortPercent": "Short %",
                        "endring": "Endring",
                        "date": "Dato",
                    })
                    .head(20),
                    width="stretch",
                    hide_index=True,
                )
        with right:
            st.markdown("### Nye posisjoner over 0,5 %")
            new_positions = finn_nye_shortposisjoner(df)
            if new_positions.empty:
                st.info("Ingen nye posisjoner å vise.")
            else:
                st.dataframe(
                    new_positions[["issuerName", "shortPercent", "forrige_short", "date"]]
                    .rename(columns={
                        "issuerName": "Selskap",
                        "shortPercent": "Short %",
                        "forrige_short": "Forrige %",
                        "date": "Dato",
                    })
                    .head(20),
                    width="stretch",
                    hide_index=True,
                )


def vis_sok_og_graf(df: pd.DataFrame, key_prefix: str) -> None:
    if df.empty:
        st.info("Ingen data tilgjengelig.")
        return

    required = {"issuerName", "isin", "date", "shortPercent"}
    missing = sorted(required.difference(df.columns))
    if missing:
        st.error("Dataene mangler kolonnene: " + ", ".join(missing))
        return

    search = st.text_input(
        "Søk etter selskap eller ISIN",
        placeholder="F.eks. EQUINOR, MPC eller NO0010096985",
        key=f"{key_prefix}_search",
    ).strip()

    filtered = df
    if search:
        issuer_mask = (
            filtered["issuerName"].fillna("").astype(str)
            .str.contains(search, case=False, na=False, regex=False)
        )
        isin_mask = (
            filtered["isin"].fillna("").astype(str)
            .str.contains(search, case=False, na=False, regex=False)
        )
        filtered = filtered.loc[issuer_mask | isin_mask]

    issuers = sorted(filtered["issuerName"].dropna().astype(str).unique().tolist())
    selected = st.multiselect(
        "Velg ett eller flere selskaper",
        options=issuers,
        default=issuers[:1] if search and issuers else [],
        key=f"{key_prefix}_issuers",
    )

    shown = filtered.loc[filtered["issuerName"].astype(str).isin(selected)] if selected else filtered
    if shown.empty:
        st.info("Ingen treff for søket eller filteret.")
        return

    st.caption(f"Viser {len(shown):,} av {len(df):,} rader.")
    st.dataframe(_standardiser_shortpercent(shown).head(1000), width="stretch", hide_index=True)

    plot_data = _agg_issuer_date(shown)
    if not plot_data.empty:
        fig = px.line(
            plot_data,
            x="date",
            y="shortPercent",
            color="issuerName",
            markers=True,
            title="Utvikling i shortposisjon",
            labels={"date": "Dato", "shortPercent": "Shortandel (%)", "issuerName": "Selskap"},
        )
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            height=600,
            legend_title_text="Utsteder",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(color="#0f172a"),
            margin=dict(l=20, r=20, t=70, b=20),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_short_chart")


# -------------------- APP --------------------
st.set_page_config(page_title="Shortsalg-register", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #050914;
        --panel: rgba(11, 18, 32, 0.88);
        --panel-2: rgba(15, 25, 45, 0.92);
        --border: rgba(95, 140, 255, 0.22);
        --blue: #4f7cff;
        --cyan: #22d3ee;
        --purple: #8b5cf6;
        --green: #22c55e;
        --red: #fb7185;
        --text: #f8fafc;
        --muted: #94a3b8;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 84% 3%, rgba(79, 124, 255, 0.10), transparent 24%),
            linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%);
        color: #0f172a;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 249, 252, 0.88);
        backdrop-filter: blur(16px);
        border-bottom: 1px solid rgba(15, 23, 42, 0.08);
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    section.main > div.block-container, .block-container {
        padding-top: 1.25rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1480px !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--border);
        border-radius: 26px;
        padding: 34px 38px;
        margin: 6px 0 22px 0;
        background:
            linear-gradient(130deg, rgba(9, 18, 36, 0.96), rgba(10, 18, 34, 0.84)),
            radial-gradient(circle at 86% 20%, rgba(34, 211, 238, 0.22), transparent 34%);
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.38);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 360px;
        height: 360px;
        right: -100px;
        top: -180px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(79,124,255,.32), rgba(79,124,255,0));
        filter: blur(3px);
    }

    .hero-kicker {
        color: var(--cyan);
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.17em;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: clamp(2.15rem, 5vw, 4.25rem);
        line-height: 0.98;
        margin: 0;
        font-weight: 900;
        letter-spacing: -0.045em;
        background: linear-gradient(90deg, #ffffff 12%, #b7d5ff 55%, #67e8f9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-copy {
        color: #b7c3d5;
        font-size: 1.02rem;
        max-width: 760px;
        margin: 16px 0 0 0;
        line-height: 1.65;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 20px;
    }

    .hero-badge {
        padding: 8px 12px;
        border-radius: 999px;
        border: 1px solid rgba(103, 232, 249, 0.18);
        background: rgba(15, 35, 58, 0.72);
        color: #dbeafe;
        font-size: 0.82rem;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.94);
        border: 1px solid rgba(79, 124, 255, 0.16);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 12px 30px rgba(15,23,42,.08);
    }

    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 1.85rem !important;
        font-weight: 900 !important;
    }

    div[data-testid="stMetricDelta"] {
        font-weight: 800 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background: rgba(255,255,255,0.88) !important;
        border: 1px solid rgba(79, 124, 255, 0.14);
        border-radius: 16px;
        padding: 7px !important;
        margin-bottom: 18px;
        backdrop-filter: blur(14px);
    }

    .stTabs [data-baseweb="tab"] {
        min-height: 46px !important;
        padding: 0 18px !important;
        border-radius: 11px !important;
        color: #475569 !important;
        font-weight: 800 !important;
        transition: all .2s ease;
    }

    .stTabs [data-baseweb="tab"] p {
        color: inherit !important;
        font-size: 0.9rem !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(79, 124, 255, 0.13) !important;
        color: white !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #315cff, #7048f5) !important;
        color: white !important;
        box-shadow: 0 10px 28px rgba(72, 92, 255, 0.28);
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(95, 140, 255, 0.15) !important;
        border-radius: 18px !important;
    }

    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(79, 124, 255, 0.14);
        border-radius: 16px;
        overflow: hidden;
    }

    [data-testid="stDataFrame"] {
        background: white;
        border: 1px solid rgba(79, 124, 255, 0.14);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 14px 36px rgba(0,0,0,.16);
    }

    [data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid rgba(79, 124, 255, 0.14);
        border-radius: 18px;
        padding: 10px;
        box-shadow: 0 14px 32px rgba(15,23,42,.08);
        overflow: hidden;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    [data-testid="stTextInput"] input {
        background: white !important;
        border-color: rgba(79, 124, 255, 0.20) !important;
        color: #0f172a !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        border: 1px solid rgba(103, 232, 249, 0.2) !important;
        background: linear-gradient(135deg,#0f172a,#1e3a8a); !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.05rem !important;
        font-weight: 850 !important;
        box-shadow: 0 10px 25px rgba(61, 82, 255, 0.23);
        transition: transform .2s ease, box-shadow .2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(61, 82, 255, 0.34);
        border-color: rgba(103, 232, 249, 0.52) !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid rgba(79, 124, 255, 0.14);
        background: rgba(255,255,255,0.92);
    }

    h1, h2, h3 {
        color: #0f172a !important;
        letter-spacing: -0.025em;
    }

    p, label, .stCaption, [data-testid="stCaptionContainer"] {
        color: #475569;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.12) !important;
    }

    @media (max-width: 760px) {
        section.main > div.block-container, .block-container {
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }
        .hero {
            padding: 25px 22px;
            border-radius: 20px;
        }
        .hero-title {
            font-size: 2.45rem;
        }
    }
    </style>

    <div class="hero">
        <div class="hero-kicker">Velkommen!</div>
        <h1 class="hero-title">Shortsalg-register fra Finanstilsynet</h1>
        <p class="hero-copy">
            Dette er et analyseverktøy for offentlig rapporterte shortposisjoner i norske børsnoterte selskaper som jeg lagde ved University of Oxford - Säid Business School (i ettertid har jeg bare lagt på et enkelt design).
            Følg utvikling, oppdag nye posisjoner og analyser markedets mest shortede aksjer. Shortregisteret fra Finanstilsynet oppdateres fra dem hver handelsdag kl. 15:30 CET.
        </p>
        <div class="hero-badges">
            <span class="hero-badge"> Live Finanstilsynet-data</span>
            <span class="hero-badge"> Interaktive analyser</span>
            <span class="hero-badge"> Historisk SQLite-register</span>
            <span class="hero-badge"> Søk på selskap og ISIN</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Registeret ligger i en delt ressurs-cache. Ingen kopier lagres i brukernes session_state.
with st.spinner("Laster delt datagrunnlag …"):
    df_live = hent_fullt_register()

# SQLite-data leses også fra en delt cache og blir ikke lagret per bruker.
df_db = hent_database_data()

tab_live, tab_db, tab_top10, tab_about = st.tabs(
    ["Live oversikt", "Søk i selskaper", "Topp 10", "Om plattformen"]
)

with tab_live:
    st.header("Live markedsoversikt")
    st.info(
        "Dette registeret hentes automatisk og deles mellom alle besøkende."
        "Dermed så slipper hver bruker å laste ned og lagre sin egen kopi i minnet."
    )

    if df_live.empty:
        st.error("Klarte ikke hente data fra Finanstilsynet akkurat nå.")
    else:
        latest_date = pd.to_datetime(df_live["date"], errors="coerce").max()
        total_short = _standardiser_shortpercent(df_live)["shortPercent"].sum()
        max_short = _standardiser_shortpercent(df_live)["shortPercent"].max()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Live-posisjoner", f"{len(df_live):,}")
        col2.metric("Unike selskaper", f"{df_live['issuerName'].nunique():,}")
        col3.metric("Aggregert short", f"{total_short:,.2f} %")
        col4.metric("Største enkeltposisjon", f"{max_short:,.2f} %")

        st.caption(
            "Siste registrerte dato: "
            + (latest_date.strftime("%d.%m.%Y") if pd.notna(latest_date) else "ukjent")
        )

        action_left, action_right = st.columns([1, 2])
        with action_left:
            if st.button("Lagre nye rader i SQLite", key="save_live"):
                with st.spinner("Sammenligner og lagrer nye rader …"):
                    new_rows = lagre_i_database(df_live)
                st.success(f"Ferdig. {new_rows:,} nye rader ble lagret.")
                st.rerun()

        with action_right:
            st.info("Den delte cachen reduserer belastning og gjør appen mer stabil ved høy trafikk.")

        with st.expander("Administrativ oppdatering (man må inn her for å laste ned short-registeret)", expanded=False):
            st.warning(
                "Denne knappen tømmer den delte én-timescachen. Bruk den bare når man faktisk trenger helt nye data."
            )
            if st.button("Nedlasting fra Finanstilsynet", key="force_refresh"):
                tving_ny_nedlasting()
                st.rerun()

        st.download_button(
            "Last ned live-registeret som CSV",
            data=dataframe_to_csv(df_live),
            file_name="shortregister.csv",
            mime="text/csv",
        )

        vis_hurtiginnsikt(df_live, expanded=True)
        st.subheader("Søk og filtrering")
        vis_sok_og_graf(df_live, "live")

    st.divider()
    st.subheader("Status for SQLite-registeret")
    latest_time, total_rows = hent_siste_oppdatering()
    if latest_time:
        st.markdown(f"**Sist lagret:** {latest_time}  \n**Totalt antall rader:** {total_rows:,}")
    else:
        st.info("Ingen lagringshistorikk er registrert ennå.")


with tab_db:
    st.header("Søk i historiske shortposisjoner")
    if df_db.empty:
        st.info("SQLite-databasen er tom. Lagre live-registeret først.")
    else:
        st.success(f"Databasen inneholder {len(df_db):,} rader.")
        vis_hurtiginnsikt(df_db)
        vis_sok_og_graf(df_db, "db")


with tab_top10:
    st.header("Markedets mest shortede selskaper")
    if df_db.empty:
        st.info("SQLite-databasen er tom. Lagre live-registeret først.")
    else:
        data = _standardiser_shortpercent(df_db)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["issuerName", "date", "shortPercent"])

        period = st.selectbox("Velg tidsperiode", ["30 dager", "90 dager", "180 dager", "365 dager"])
        days = int(period.split()[0])
        start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
        recent = data.loc[data["date"] >= start_date]

        if recent.empty:
            st.warning("Ingen data for valgt periode.")
        else:
            top10 = (
                recent.groupby("issuerName", as_index=False)["shortPercent"]
                .mean()
                .sort_values("shortPercent", ascending=False)
                .head(10)
            )
            st.download_button(
                "Last ned Topp 10 som CSV",
                dataframe_to_csv(top10),
                f"topp10_shorts_{days}d.csv",
                "text/csv",
            )

            fig_bar = px.bar(
                top10,
                x="issuerName",
                y="shortPercent",
                text_auto=".2f",
                title=f"Topp 10 – gjennomsnittlig shortandel siste {days} dager",
                labels={"issuerName": "Selskap", "shortPercent": "Shortandel (%)"},
            )
            fig_bar.update_layout(
                template="plotly_white",
                xaxis_tickangle=-35,
                height=500,
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font=dict(color="#0f172a"),
                margin=dict(l=20, r=20, t=70, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="top10_bar_chart")
            st.dataframe(top10, width="stretch", hide_index=True)

            names = top10["issuerName"].tolist()
            development = (
                recent.loc[recent["issuerName"].isin(names)]
                .groupby(["issuerName", "date"], as_index=False)["shortPercent"]
                .mean()
            )
            if not development.empty:
                fig_line = px.line(
                    development,
                    x="date",
                    y="shortPercent",
                    color="issuerName",
                    title="Utvikling over tid for Topp 10",
                    labels={"date": "Dato", "shortPercent": "Shortandel (%)", "issuerName": "Selskap"},
                )
                fig_line.update_layout(
                    template="plotly_white",
                    hovermode="x unified",
                    height=600,
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#0f172a"),
                    margin=dict(l=20, r=20, t=70, b=20),
                )
                st.plotly_chart(fig_line, use_container_width=True, key="top10_line_chart")

                heat = (
                    development.pivot_table(index="issuerName", columns="date", values="shortPercent")
                    .diff(axis=1)
                    .fillna(0)
                )
                if not heat.empty:
                    fig_heat = px.imshow(
                        heat,
                        aspect="auto",
                        title="Daglige endringer i shortandel",
                        labels={"x": "Dato", "y": "Selskap", "color": "Endring (%)"},
                    )
                    fig_heat.update_layout(
                        template="plotly_white",
                        height=600,
                        paper_bgcolor="#ffffff",
                        font=dict(color="#0f172a"),
                        margin=dict(l=20, r=20, t=70, b=20),
                    )
                    st.plotly_chart(fig_heat, use_container_width=True, key="top10_heatmap")


with tab_about:
    st.header("Om plattformen")
    st.markdown(
        """
        Plattformen visualiserer offentlig tilgjengelige shortposisjoner fra Finanstilsynets
        Short Sale Register. Den er utviklet for læring, markedsinnsikt og enklere tilgang til
        historiske data.

        **Teknologi:** Python, Streamlit, Pandas, Plotly og SQLite.

        **Utvikler:** Andreas Bolton Seielstad.
        """
    )
