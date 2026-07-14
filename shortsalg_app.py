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
        )
        st.plotly_chart(fig, width="stretch")


# -------------------- APP --------------------
st.set_page_config(page_title="Shortsalg-register", layout="wide")

st.markdown(
    """
    <style>
    section.main > div.block-container, .block-container {
        padding-top: 0.6rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 100% !important;
    }
    .euronext-title {
        background-color: #7f95a8;
        color: white !important;
        padding: 60px;
        font-size: 34px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 20px 0 0 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #4169E1, #1E3FAF) !important;
        border-bottom: 2px solid #0f2f8f !important;
        gap: 0 !important;
        min-height: 58px !important;
        overflow-x: auto !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: white !important;
        min-height: 58px !important;
        padding: 0 22px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
    }
    .stTabs [data-baseweb="tab"] p { color: white !important; font-weight: 700 !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #eeece6 !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] p { color: black !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: transparent !important; height: 0 !important; }
    @media (max-width: 700px) {
        .euronext-title { font-size: 24px; padding: 20px; }
        .stTabs [data-baseweb="tab"] { padding: 0 14px !important; }
    }
    </style>
    <div class="euronext-title">Shortsalg-register fra Finanstilsynet</div>
    """,
    unsafe_allow_html=True,
)

# Registeret ligger i en delt ressurs-cache. Ingen kopier lagres i brukernes session_state.
with st.spinner("Laster delt datagrunnlag …"):
    df_live = hent_fullt_register()

# SQLite-data leses også fra en delt cache og blir ikke lagret per bruker.
df_db = hent_database_data()

tab_live, tab_db, tab_top10, tab_about = st.tabs(
    ["Live data", "Søk i selskaper", "Topp 10 mest shortede", "Om plattformen"]
)

with tab_live:
    st.header("Live data fra Finanstilsynet")
    st.info(
        "Registeret hentes automatisk og deles mellom alle besøkende. "
        "Dermed slipper hver bruker å laste ned og lagre sin egen kopi i minnet."
    )

    if df_live.empty:
        st.error("Klarte ikke hente data fra Finanstilsynet akkurat nå.")
    else:
        col1, col2, col3 = st.columns([1, 1, 2])
        col1.metric("Rader i live-registeret", f"{len(df_live):,}")
        col2.metric("Unike selskaper", f"{df_live['issuerName'].nunique():,}")

        with col3:
            if st.button("Lagre nye rader i SQLite", key="save_live"):
                with st.spinner("Sammenligner og lagrer nye rader …"):
                    new_rows = lagre_i_database(df_live)
                st.success(f"Ferdig. {new_rows:,} nye rader ble lagret.")
                st.rerun()

        with st.expander("Administrativ oppdatering", expanded=False):
            st.warning(
                "Ja, denne knappen tømmer den delte én-timescachen. Denne bruker jeg når jeg faktisk trenger helt nye data."
            )
            if st.button("Last ned data fra Finanstilsynet", key="force_refresh"):
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
    st.header("Søk i lagret historikk")
    if df_db.empty:
        st.info("SQLite-databasen er tom. Lagre live-registeret først.")
    else:
        st.success(f"Databasen inneholder {len(df_db):,} rader.")
        vis_hurtiginnsikt(df_db)
        vis_sok_og_graf(df_db, "db")


with tab_top10:
    st.header("Topp 10 mest shortede selskaper")
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
            fig_bar.update_layout(template="plotly_white", xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig_bar, width="stretch")
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
                fig_line.update_layout(template="plotly_white", hovermode="x unified", height=600)
                st.plotly_chart(fig_line, width="stretch")

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
                    fig_heat.update_layout(template="plotly_white", height=600)
                    st.plotly_chart(fig_heat, width="stretch")


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
