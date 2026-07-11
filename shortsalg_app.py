import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

from ssr_api import hent_fullt_register, lagre_i_database, hent_siste_oppdatering


# -------------------- HJELPEFUNKSJONER I APPEN --------------------

def _standardiser_shortpercent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardiserer shortPercent til prosentpoeng.

    Finanstilsynets data kan komme som hundredeler av et prosentpoeng:
    58 betyr da 0,58 %, 126 betyr 1,26 %, osv.

    Funksjonen er idempotent: verdier som allerede er standardisert
    (for eksempel 0,58 eller 1,26) blir ikke endret på nytt.
    """
    if df.empty or "shortPercent" not in df.columns:
        return df

    out = df.copy()
    out["shortPercent"] = pd.to_numeric(out["shortPercent"], errors="coerce")

    mx = out["shortPercent"].max(skipna=True)
    if pd.isna(mx):
        return out

    if mx > 20:
        out["shortPercent"] = out["shortPercent"] / 100

    return out

def _agg_issuer_date(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregerer til én rad per selskap per dato (summerer shortPercent)."""
    if df.empty:
        return df.copy()

    out = df.copy()
    out = _standardiser_shortpercent(out)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["issuerName", "date", "shortPercent"])
    out = (
        out.groupby(["issuerName", "date"], as_index=False)["shortPercent"]
        .sum()
        .sort_values(["issuerName", "date"])
    )
    return out


def beregn_storste_endringer(df: pd.DataFrame) -> pd.DataFrame:
    """Returnerer siste verdi per selskap + endring fra forrige tilgjengelige dato."""
    d = _agg_issuer_date(df)
    if d.empty:
        return d

    d["forrige_short"] = d.groupby("issuerName")["shortPercent"].shift(1)
    siste = d.groupby("issuerName").tail(1).copy()
    siste["endring"] = siste["shortPercent"] - siste["forrige_short"]
    siste = siste.dropna(subset=["endring"])

    # sorter etter absolutt endring
    siste = siste.reindex(siste["endring"].abs().sort_values(ascending=False).index)
    return siste


def finn_nye_shortposisjoner(df: pd.DataFrame, terskel: float = 0.5) -> pd.DataFrame:
    """Flagger selskaper som 'dukker opp' over terskel på siste tilgjengelige dato."""
    d = _agg_issuer_date(df)
    if d.empty:
        return d

    d["forrige_short"] = d.groupby("issuerName")["shortPercent"].shift(1)
    siste = d.groupby("issuerName").tail(1).copy()

    nye = siste[
        (siste["shortPercent"] >= terskel)
        & ((siste["forrige_short"].isna()) | (siste["forrige_short"] < terskel))
    ].copy()

    return nye.sort_values(["date", "shortPercent"], ascending=[False, False])


def _fmt_pct(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return x


# -------------------- APPEN --------------------
st.set_page_config(page_title="Shortsalg-register", layout="wide")

# Euronext-inspirert toppmeny / header
st.markdown("""
<style>

/* Gjør siden bred og fjerner litt Streamlit-luft */
section.main > div.block-container, .block-container {
    padding-top: 0.6rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
    max-width: 100% !important;
}

/* Tittelbanner */
.euronext-title {
    background-color: #7f95a8;
    color: white !important;
    padding: 60px;
    font-size: 34px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0;
    margin-top: 20px;
}

/* Ekstra royal-blue menylinje-look rundt Streamlit-tabs */
.stTabs, div[data-testid="stTabs"] {
    width: 100% !important;
}

/* Selve faneraden */
.stTabs [data-baseweb="tab-list"],
div[data-testid="stTabs"] [data-baseweb="tab-list"],
div[data-testid="stTabs"] > div > div[role="tablist"] {
    background: linear-gradient(90deg, #4169E1, #1E3FAF) !important;
    border-bottom: 2px solid #0f2f8f !important;
    padding: 0 !important;
    gap: 0 !important;
    min-height: 58px !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    white-space: nowrap !important;
}

/* Alle faneknapper */
.stTabs [data-baseweb="tab"],
div[data-testid="stTabs"] [data-baseweb="tab"],
button[role="tab"] {
    background: transparent !important;
    color: white !important;
    border: none !important;
    border-radius: 0 !important;
    min-height: 58px !important;
    height: 58px !important;
    padding: 0 22px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}

/* Streamlit legger ofte tekstfargen på p/span inni knappen, derfor må disse også tvinges */
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span,
div[data-testid="stTabs"] button[role="tab"] p,
div[data-testid="stTabs"] button[role="tab"] span,
button[role="tab"] p,
button[role="tab"] span {
    color: white !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

/* Hover */
.stTabs [data-baseweb="tab"]:hover,
div[data-testid="stTabs"] button[role="tab"]:hover,
button[role="tab"]:hover {
    background-color: #2EB57A !important;
    color: black !important;
}

.stTabs [data-baseweb="tab"]:hover p,
.stTabs [data-baseweb="tab"]:hover span,
div[data-testid="stTabs"] button[role="tab"]:hover p,
div[data-testid="stTabs"] button[role="tab"]:hover span {
    color: black !important;
}

/* Aktiv fane */
.stTabs [data-baseweb="tab"][aria-selected="true"],
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
button[role="tab"][aria-selected="true"] {
    background-color: #eeece6 !important;
    color: black !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] p,
.stTabs [data-baseweb="tab"][aria-selected="true"] span,
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span,
button[role="tab"][aria-selected="true"] p,
button[role="tab"][aria-selected="true"] span {
    color: black !important;
}

/* Fjerner Streamlit sin røde/standard underline */
.stTabs [data-baseweb="tab-highlight"],
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background-color: transparent !important;
    height: 0 !important;
}

@media (max-width: 700px) {
    .euronext-title {
        font-size: 24px;
        padding: 20px;
    }
    .stTabs [data-baseweb="tab"],
    div[data-testid="stTabs"] [data-baseweb="tab"],
    button[role="tab"] {
        padding: 0 14px !important;
    }
}

</style>

<div class="euronext-title">
Shortsalg-register fra Finanstilsynet
</div>

""", unsafe_allow_html=True)

tab_live, tab_db, tab_top10, tab_about = st.tabs(
    ["Live data", "Søk i selskaper", "Topp 10 mest shortede", "Om plattformen"]
)

# ---------- FANEN FOR LIVE-DATA ----------
with tab_live:
    st.header("Hent hele shortregisteret")
    st.info("Etter at man har hentet short-informasjonen, kan man søke i de ulike selskapene lenger ned på nettsiden her, eller så kan man gå via fanen *søk i selskaper* i menyen :).")

    # Sidebar-status
    st.sidebar.markdown("### Status for live-nedlasting")
    sidebar_status = st.sidebar.empty()

    if st.button("Hent full data fra Finanstilsynet", key="live_download"):
        st.info("Starter nedlasting fra Finanstilsynet...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(percent, downloaded, total):
            if total > 0:
                msg = f" Laster {downloaded/1_000_000:.1f} MB av {total/1_000_000:.1f} MB ({percent*100:.1f}%)"
            else:
                msg = f" Laster {downloaded/1_000_000:.1f} MB ..."
            status_text.text(msg)
            sidebar_status.info(msg)
            progress_bar.progress(min(percent, 1.0))

        with st.spinner("Henter data... (kan ta litt tid)"):
            df = hent_fullt_register(_progress_callback=update_progress)

        progress_bar.progress(1.0)
        status_text.text("Jaujau!! Detta funka,og alle tilgjengelig data fra Finanstilsynet ble lastet ned.")
        sidebar_status.success("Nedlastingen er fullført!")

        if df.empty:
            st.warning("Fant ingen data fra Finanstilsynet.")
        else:
            st.success(f"Hentet {len(df):,} rader ")
            lagre_i_database(df)
            st.session_state["live_df"] = df.copy()

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(" Last ned som CSV", csv, "shortregister.csv", "text/csv")

    st.divider()

    df_live = st.session_state.get("live_df", pd.DataFrame())
    if df_live.empty:
        st.info("Ingen live-data lastet ennå. Trykk på knappen «Hent full data fra Finanstilsynet».")
    else:
        st.success(f"Live-data i minne: {len(df_live):,} rader")

        required_columns = {"issuerName", "isin", "date", "shortPercent"}
        missing_columns = sorted(required_columns.difference(df_live.columns))

        if missing_columns:
            st.error(
                "Dataene mangler forventede kolonner: "
                + ", ".join(missing_columns)
                + ". Hent data på nytt."
            )
            st.session_state["live_df"] = pd.DataFrame()
            st.rerun()

        # --- Hurtig-innsikt: endringer & nye posisjoner ---
        with st.expander("Hurtig-innsikt: Største endringer / nye posisjoner", expanded=True):
            colA, colB = st.columns(2)

            endringer = beregn_storste_endringer(df_live)
            nye = finn_nye_shortposisjoner(df_live, terskel=0.5)

            with colA:
                st.markdown("### Største endringer (siste vs forrige dato)")
                if endringer.empty:
                    st.info("Ingen endringer å vise (mangler historikk).")
                else:
                    vis = endringer[["issuerName", "shortPercent", "endring", "date"]].rename(
                        columns={"issuerName": "Selskap", "shortPercent": "Short %", "endring": "Endring", "date": "Dato"}
                    )
                    st.dataframe(vis.head(15), width="stretch")

            with colB:
                st.markdown("### Nye shortposisjoner (>= 0,5%)")
                if nye.empty:
                    st.info("Ingen nye posisjoner over 0,5% på siste dato.")
                else:
                    vis2 = nye[["issuerName", "shortPercent", "forrige_short", "date"]].rename(
                        columns={
                            "issuerName": "Selskap",
                            "shortPercent": "Short %",
                            "forrige_short": "Forrige %",
                            "date": "Dato",
                        }
                    )
                    st.dataframe(vis2.head(15), width="stretch")

        # --- Søk + filter ---
        st.subheader("Søk og filtrering")

        def nullstill_live_filter():
            st.session_state["live_sokefelt"] = ""
            st.session_state["live_utsteder_filter"] = ["(Alle)"]

        if "live_sokefelt" not in st.session_state:
            st.session_state["live_sokefelt"] = ""

        if "live_utsteder_filter" not in st.session_state:
            st.session_state["live_utsteder_filter"] = ["(Alle)"]

        col1, col2 = st.columns([4, 1])

        with col1:
            sok = st.text_input(
                "Søk etter selskap eller ISIN",
                placeholder="F.eks. 'Hoegh', 'MPC', 'AKER BP'...",
                key="live_sokefelt",
            )

        with col2:
            st.button(
                "Nullstill filter",
                key="live_nullstill",
                on_click=nullstill_live_filter,
            )

        df_filtered = df_live.copy()

        if sok.strip():
            s = sok.strip()

            issuer_mask = (
                df_filtered["issuerName"]
                .fillna("")
                .astype(str)
                .str.contains(s, case=False, na=False, regex=False)
            )

            isin_mask = (
                df_filtered["isin"]
                .fillna("")
                .astype(str)
                .str.contains(s, case=False, na=False, regex=False)
            )

            df_filtered = df_filtered[issuer_mask | isin_mask]

        utstedere = sorted(
            df_filtered["issuerName"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        alle_valg = ["(Alle)"] + utstedere

        tidligere_valg = st.session_state.get(
            "live_utsteder_filter",
            ["(Alle)"],
        )

        gyldige_valg = [
            verdi
            for verdi in tidligere_valg
            if verdi in alle_valg
        ]

        if not gyldige_valg:
            gyldige_valg = ["(Alle)"]

        st.session_state["live_utsteder_filter"] = gyldige_valg

        valgte = st.multiselect(
            "Velg ett eller flere selskaper",
            options=alle_valg,
            key="live_utsteder_filter",
        )

        if "(Alle)" in valgte or not valgte:
            df_plot = df_filtered.copy()
        else:
            df_plot = df_filtered[
                df_filtered["issuerName"].fillna("").astype(str).isin(valgte)
            ].copy()

        if df_plot.empty:
            st.info("Ingen treff for søket eller filteret.")
        else:
            df_visning = _standardiser_shortpercent(df_plot)
            st.dataframe(df_visning.head(1000), width="stretch")

        if not df_plot.empty:
            df_plot = _agg_issuer_date(df_plot)
            fig = px.line(
                df_plot,
                x="date",
                y="shortPercent",
                color="issuerName",
                markers=True,
                title="Utvikling i shortposisjon",
                labels={"issuerName": "Selskap", "shortPercent": "Shortandel (%)"},
            )
            fig.update_layout(
                template="plotly_white",
                hovermode="x unified",
                title_font_size=20,
                xaxis_title="Dato",
                yaxis_title="Shortandel (%)",
                legend_title_text="Utsteder",
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ingen data tilgjengelig for valgt søk eller filter.")

    # Status nederst
    st.divider()
    st.subheader("Status for shortregisteret")
    siste_tid, total_rader = hent_siste_oppdatering()
    if siste_tid:
        st.markdown(f"**Sist oppdatert (lokalt):** {siste_tid}  \n**Totalt antall rader:** {total_rader:,}")
        st.caption("NB: 'Sist oppdatert' er tidspunktet siste data ble lagret lokalt i databasen, ikke nødvendigvis Finanstilsynets publiseringstidspunkt.")
    else:
        st.info("Ingen oppdateringsinformasjon funnet ennå.")


# ---------- FANEN FOR DATABASE-DATA ----------
with tab_db:
    st.header("Lokalt lagret shortdata (fra SQLite)")

    if "db_data" not in st.session_state:
        st.session_state["db_data"] = pd.DataFrame()

    if st.button(" Hent data fra database"):
        try:
            conn = sqlite3.connect("shortsalg.db")
            df_hist = pd.read_sql("SELECT * FROM short_positions", conn)
            conn.close()
            if df_hist.empty:
                st.info("Databasen er tom. Hent og lagre data først.")
            else:
                st.session_state["db_data"] = df_hist
                st.success(f"Fant {len(df_hist):,} rader ")
        except Exception as e:
            st.error(f"Feil ved lesing av database: {e}")

    if not st.session_state["db_data"].empty:
        df_hist = st.session_state["db_data"]

        with st.expander(" Hurtig-innsikt: Største endringer / nye posisjoner", expanded=False):
            colA, colB = st.columns(2)
            with colA:
                endringer = beregn_storste_endringer(df_hist)
                st.markdown("### Største endringer (siste vs forrige dato)")
                st.dataframe(
                    endringer[["issuerName", "shortPercent", "endring", "date"]].rename(
                        columns={"issuerName": "Selskap", "shortPercent": "Short %", "endring": "Endring", "date": "Dato"}
                    ).head(20),
                    width="stretch",
                )
            with colB:
                nye = finn_nye_shortposisjoner(df_hist, terskel=0.5)
                st.markdown("### Nye shortposisjoner (>= 0,5%)")
                st.dataframe(
                    nye[["issuerName", "shortPercent", "forrige_short", "date"]].rename(
                        columns={"issuerName": "Selskap", "shortPercent": "Short %", "forrige_short": "Forrige %", "date": "Dato"}
                    ).head(20),
                    width="stretch",
                )

        st.markdown("### Søk og filtrering")
        søkbare = sorted(set(df_hist["issuerName"].dropna().tolist() + df_hist["isin"].dropna().tolist()))
        valgt_søk = st.selectbox("Velg eller søk (autocomplete)", ["(Alle)"] + søkbare, index=0, key="db_autocomplete")

        df_søk = df_hist.copy()
        if valgt_søk != "(Alle)":
            issuer_text = df_hist["issuerName"].fillna("").astype(str)
            isin_text = df_hist["isin"].fillna("").astype(str)

            df_søk = df_hist[
                issuer_text.str.contains(
                    str(valgt_søk),
                    case=False,
                    na=False,
                    regex=False,
                )
                | isin_text.str.contains(
                    str(valgt_søk),
                    case=False,
                    na=False,
                    regex=False,
                )
            ].copy()

        utstedere = sorted(df_søk["issuerName"].dropna().astype(str).unique().tolist())
        valgte = st.multiselect(
            "Velg ett eller flere selskaper (kan kombineres med søk over)",
            utstedere,
            default=utstedere[:1] if utstedere else None,
            key="db_multiselect",
        )

        df_vis = (
            df_søk[df_søk["issuerName"].fillna("").astype(str).isin(valgte)].copy()
            if valgte
            else df_søk.copy()
        )
        df_visning = _standardiser_shortpercent(df_vis)
        st.dataframe(df_visning.head(1000), width="stretch")

        if not df_vis.empty:
            df_plotly = _agg_issuer_date(df_vis)
            fig_plotly = px.line(
                df_plotly,
                x="date",
                y="shortPercent",
                color="issuerName",
                title="Interaktiv utvikling i shortandel",
                markers=True,
                labels={"date": "Dato", "shortPercent": "Shortandel (%)", "issuerName": "Selskap"},
            )
            fig_plotly.update_layout(template="plotly_white", hovermode="x unified", height=600)
            st.plotly_chart(fig_plotly, use_container_width=True)


# ---------- FANEN FOR TOPP 10 ----------
with tab_top10:
    st.header("Topp 10 shortede selskaper fra SQLite")
    st.info(
        "Man må laste ned dataene fra Finanstilsynet under Live Data oppe i menyen, "
        "før man kan ta i bruk dette."
    )

    try:
        conn = sqlite3.connect("shortsalg.db")
        df_all = pd.read_sql("SELECT * FROM short_positions", conn)
        conn.close()
    except Exception as e:
        df_all = pd.DataFrame()
        st.warning(f"Databasen er ikke klar ennå: {e}")

    if df_all.empty:
        st.info("Det er ingen data i databasen. Gå til «Live-data» og hent det først.")
    else:
        df_all = _standardiser_shortpercent(df_all)
        df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
        df_all = df_all.dropna(subset=["issuerName", "date", "shortPercent"])

        with st.expander("Hurtig-innsikt på hele markedet", expanded=False):
            colA, colB = st.columns(2)

            with colA:
                endr = beregn_storste_endringer(df_all)
                st.markdown("### Største endringer (siste vs forrige dato)")
                if endr.empty:
                    st.info("Ingen endringer å vise.")
                else:
                    st.dataframe(
                        endr[["issuerName", "shortPercent", "endring", "date"]]
                        .rename(
                            columns={
                                "issuerName": "Selskap",
                                "shortPercent": "Short %",
                                "endring": "Endring",
                                "date": "Dato",
                            }
                        )
                        .head(20),
                        width="stretch",
                    )

            with colB:
                nye = finn_nye_shortposisjoner(df_all, terskel=0.5)
                st.markdown("### Nye shortposisjoner (>= 0,5%)")
                if nye.empty:
                    st.info("Ingen nye posisjoner å vise.")
                else:
                    st.dataframe(
                        nye[["issuerName", "shortPercent", "forrige_short", "date"]]
                        .rename(
                            columns={
                                "issuerName": "Selskap",
                                "shortPercent": "Short %",
                                "forrige_short": "Forrige %",
                                "date": "Dato",
                            }
                        )
                        .head(20),
                        width="stretch",
                    )

        periodevalg = st.selectbox(
            "Velg tidsperiode",
            ["30 dager", "90 dager", "180 dager", "365 dager"],
            index=0,
        )
        antall_dager = int(periodevalg.split()[0])
        start_dato = pd.Timestamp.today().normalize() - pd.Timedelta(days=int(antall_dager))
        df_recent = df_all[df_all["date"] >= start_dato].copy()

        if df_recent.empty:
            st.warning("Ingen data for valgt periode.")
        else:
            df_top10 = (
                df_recent.groupby("issuerName", as_index=False)["shortPercent"]
                .mean()
                .sort_values("shortPercent", ascending=False)
                .head(10)
            )

            st.markdown(f"### Topp 10 shortede selskaper (siste {antall_dager} dager)")

            csv_top10 = df_top10.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Last ned Topp 10 som CSV",
                data=csv_top10,
                file_name=f"topp10_shorts_{antall_dager}d.csv",
                mime="text/csv",
            )

            fig_bar = px.bar(
                df_top10,
                x="issuerName",
                y="shortPercent",
                title=f"Topp 10 shortede selskaper (siste {antall_dager} dager)",
                labels={"issuerName": "Selskap", "shortPercent": "Shortandel (%)"},
                text_auto=".2f",
                color="shortPercent",
                color_continuous_scale="Reds",
            )
            fig_bar.update_layout(template="plotly_white", xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(df_top10, width="stretch")

            topp10_liste = df_top10["issuerName"].tolist()
            df_utv = (
                df_recent[df_recent["issuerName"].isin(topp10_liste)]
                .groupby(["issuerName", "date"], as_index=False)["shortPercent"]
                .mean()
            )

            if df_utv.empty:
                st.info("Ingen utviklingsdata tilgjengelig for Topp 10.")
            else:
                siste_dato = df_utv["date"].max()
                forste_dato = df_utv["date"].min()

                siste = df_utv[df_utv["date"] == siste_dato]
                forste = df_utv[df_utv["date"] == forste_dato]

                diff = (
                    siste.set_index("issuerName")["shortPercent"]
                    .subtract(forste.set_index("issuerName")["shortPercent"], fill_value=0)
                    .sort_values(ascending=False)
                )

                st.markdown("### Utvikling over tid for Topp 10")
                fig_utv = px.line(
                    df_utv,
                    x="date",
                    y="shortPercent",
                    color="issuerName",
                    labels={
                        "date": "Dato",
                        "shortPercent": "Shortandel (%)",
                        "issuerName": "Selskap",
                    },
                    title=f"Utvikling i shortandel for Topp 10 (siste {antall_dager} dager)",
                )
                fig_utv.update_layout(
                    template="plotly_white",
                    hovermode="x unified",
                    height=600,
                )
                st.plotly_chart(fig_utv, use_container_width=True)

                st.markdown("### Endring siste periode")
                df_diff = pd.DataFrame(
                    {
                        "issuerName": diff.index,
                        "Endring siste periode (%)": diff.values,
                    }
                )

                df_spi = df_top10.merge(df_diff, on="issuerName", how="left")
                df_spi["Endring siste periode (%)"] = (
                    df_spi["Endring siste periode (%)"].fillna(0)
                )
                df_spi["Short Pressure Index (0–100)"] = (
                    (df_spi["shortPercent"] * 0.7)
                    + (df_spi["Endring siste periode (%)"] * 3.0)
                ).clip(0, 100)

                df_spi["Retning"] = df_spi["Endring siste periode (%)"].apply(
                    lambda x: "🔺 Økende" if x > 0 else "🔻 Fallende"
                )

                st.dataframe(
                    df_spi.sort_values(
                        "Short Pressure Index (0–100)",
                        ascending=False,
                    ),
                    width="stretch",
                )

                st.markdown("### Short Pressure Index (SPI)")
                fig_spi = px.bar(
                    df_spi,
                    x="issuerName",
                    y="Short Pressure Index (0–100)",
                    color="Short Pressure Index (0–100)",
                    color_continuous_scale="RdYlGn_r",
                    text_auto=".1f",
                    labels={
                        "issuerName": "Selskap",
                        "Short Pressure Index (0–100)": "Shortpress",
                    },
                    title="Short Pressure Index – kombinasjon av shortandel og endring",
                )
                fig_spi.update_layout(
                    template="plotly_white",
                    xaxis_tickangle=-45,
                    height=500,
                )
                st.plotly_chart(fig_spi, use_container_width=True)

                st.markdown("### Short Heatmap – daglige endringer for Topp 10")
                df_heat = (
                    df_utv.pivot_table(
                        index="issuerName",
                        columns="date",
                        values="shortPercent",
                    )
                    .diff(axis=1)
                    .fillna(0)
                )

                if df_heat.empty:
                    st.info("Ikke nok historikk til å lage heatmap.")
                else:
                    fig_heat = px.imshow(
                        df_heat,
                        color_continuous_scale=["green", "black", "red"],
                        aspect="auto",
                        title=(
                            "Daglige endringer i shortandel "
                            "(grønn = dekker inn, rød = økende short)"
                        ),
                        labels={
                            "x": "Dato",
                            "y": "Selskap",
                            "color": "Endring (%)",
                        },
                    )
                    fig_heat.update_layout(
                        template="plotly_white",
                        height=600,
                        xaxis_title="Dato",
                        yaxis_title="Selskap",
                        xaxis_tickangle=-45,
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)


# ---------- ℹ️ FANEN: OM / ABOUT ----------
with tab_about:
    st.markdown(
        """
        <style>
        .about-card {
            background-color: #f9f9f9;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 25px;
        }
        .about-header {
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .about-section-title {
            font-size: 20px;
            font-weight: 600;
            color: #444;
            margin-top: 20px;
        }
        .emoji {
            font-size: 22px;
            margin-right: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='about-header'> Om denne plattformen</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class='about-card'>
        Denne apppen her visualiserer shortposisjoner i norske børsnoterte selskaper, basert på åpne data fra
        <a href='https://ssr.finanstilsynet.no/' target='_blank'>Finanstilsynets Short Sale Register (SSR)</a>.
        <br><br>
        Målet med dette her er å gjøre shortinformasjon lettere tilgjengelig og mer oversiktlig for investorer, analytikere og studenter.
        </div>

        <div class='about-section-title'>Hovedfunksjoner</div>
        <div class='about-card'>
        <ul>
            <li>Her kan man søke, filtrere og sammenligne shortposisjoner per selskap</li>
            <li>Se topp 10 shortede selskaper med historikk og SPI (det står for Short Pressure Index)</li>
            <li>Se største endringer / nye posisjoner på siste oppdaterte dato</li>
            <li>Man kan lagre historikk lokalt i denne SQLite-databasen på nettsiden her</li>
            <li>Det er også en rekke muligheter med visualisering med Plotly, interaktive grafer og heatmaps</li>
        </ul>
        </div>

        <div class='about-section-title'>Teknisk stack</div>
        <div class='about-card'>
        <ul>
            <li><b>Python</b> + <b>Streamlit</b> – frontend og logikk i Python</li>
            <li><b>Pandas</b> – Det er for databehandling og analyse</li>
            <li><b>Plotly</b> – Det er interaktive grafer og visualisering som man ser på nettsiden</li>
            <li><b>SQLite</b> – lokal database for lagring</li>
            <li><b>Finanstilsynet SSR</b> – Det er datakilden jeg har brukt her</li>
        </ul>
        </div>

        <div class='about-section-title'>Om utvikleren</div>
        <div class='about-card'>
        <p>Utviklet av <b>Andreas Bolton Seielstad</b></p>
        <p>Prosjektet er laget for læring, innsikt og åpenhet i finansmarkedet.</p>
        <a href='https://github.com/An-Bolton' target='_blank'>GitHub: An-Bolton</a>
        </div>
        """,
        unsafe_allow_html=True,
    )