import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

from ssr_api import hent_fullt_register, lagre_i_database, hent_siste_oppdatering


# -------------------- HJELPEFUNKSJONER --------------------
def _agg_issuer_date(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregerer til én rad per selskap per dato (summerer shortPercent)."""
    if df.empty:
        return df.copy()

    out = df.copy()
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


# -------------------- APP --------------------
st.set_page_config(page_title="Shortsalg-register", layout="wide")
st.title("Shortsalg-register fra Finanstilsynet")

tab_live, tab_db, tab_top10, tab_about = st.tabs(
    ["Live-data", "Søk i selskaper på Oslo Børs", "Oversikt over de top 10 mest shortede", "Om plattformen"]
)

# ---------- 📈 FANEN FOR LIVE-DATA ----------
with tab_live:
    st.header("Hent hele shortregisteret")

    # Sidebar-status
    st.sidebar.markdown("### 🌀 Status for live-nedlasting")
    sidebar_status = st.sidebar.empty()

    if st.button("🔄 Hent full data fra Finanstilsynet", key="live_download"):
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
        status_text.text("✅ Ferdig! Data mottatt.")
        sidebar_status.success("✅ Nedlasting fullført!")

        if df.empty:
            st.warning("Fant ingen data fra Finanstilsynet.")
        else:
            st.success(f"Hentet {len(df):,} rader ✅")
            lagre_i_database(df)
            st.session_state["live_df"] = df.copy()

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Last ned som CSV", csv, "shortregister.csv", "text/csv")

    st.divider()

    df_live = st.session_state.get("live_df", pd.DataFrame())
    if df_live.empty:
        st.info("Ingen live-data lastet ennå. Trykk «Hent full data fra Finanstilsynet».")
    else:
        st.success(f"Live-data i minne: {len(df_live):,} rader")

        # --- Hurtig-innsikt: endringer & nye posisjoner ---
        with st.expander("⚡ Hurtig-innsikt: Største endringer / nye posisjoner", expanded=True):
            colA, colB = st.columns(2)

            endringer = beregn_storste_endringer(df_live)
            nye = finn_nye_shortposisjoner(df_live, terskel=0.5)

            with colA:
                st.markdown("### 📊 Største endringer (siste vs forrige dato)")
                if endringer.empty:
                    st.info("Ingen endringer å vise (mangler historikk).")
                else:
                    vis = endringer[["issuerName", "shortPercent", "endring", "date"]].rename(
                        columns={"issuerName": "Selskap", "shortPercent": "Short %", "endring": "Endring", "date": "Dato"}
                    )
                    st.dataframe(vis.head(15), use_container_width=True)

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
                    st.dataframe(vis2.head(15), use_container_width=True)

        # — Søk + filter —
        st.subheader("🔍 Søk og filtrering")

        if "live_sokeord" not in st.session_state:
            st.session_state.live_sokeord = ""
        if "live_valgte_utstedere" not in st.session_state:
            st.session_state.live_valgte_utstedere = ["(Alle)"]

        col1, col2 = st.columns([4, 1])
        with col1:
            sok = st.text_input(
                "Søk etter selskap eller ISIN",
                value=st.session_state.live_sokeord,
                placeholder="F.eks. 'Hoegh', 'MPC', 'AKER BP'...",
                key="live_sokefelt",
            )
        with col2:
            if st.button("🔄 Nullstill filter", key="live_nullstill"):
                st.session_state.live_sokeord = ""
                st.session_state.live_valgte_utstedere = ["(Alle)"]
                sok = ""

        if sok != st.session_state.live_sokeord:
            st.session_state.live_sokeord = sok

        df_filtered = df_live.copy()
        if st.session_state.live_sokeord.strip():
            s = st.session_state.live_sokeord.strip().lower()
            df_filtered = df_filtered[
                df_filtered["issuerName"].str.lower().str.contains(s, na=False)
                | df_filtered["isin"].str.lower().str.contains(s, na=False)
            ]

        utstedere = sorted(df_filtered["issuerName"].dropna().unique().tolist())
        alle_valg = ["(Alle)"] + utstedere

        st.session_state.live_valgte_utstedere = [
            v for v in st.session_state.live_valgte_utstedere if (v == "(Alle)" or v in utstedere)
        ] or ["(Alle)"]

        valgte = st.multiselect(
            "Velg ett eller flere selskaper",
            options=alle_valg,
            key="live_utsteder_filter",
        )

        if valgte:
            st.session_state.live_valgte_utstedere = valgte

        valgte = st.session_state.live_valgte_utstedere

        if "(Alle)" in valgte or not valgte:
            df_plot = df_filtered.copy()
        else:
            df_plot = df_filtered[df_filtered["issuerName"].isin(valgte)]

        st.dataframe(df_plot.head(1000), use_container_width=True)

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
        st.caption("NB: 'Sist oppdatert' er tidspunktet siste data ble lagret i din lokale database, ikke nødvendigvis Finanstilsynets publiseringstidspunkt.")
    else:
        st.info("Ingen oppdateringsinformasjon funnet ennå.")


# ---------- FANEN FOR DATABASE-DATA ----------
with tab_db:
    st.header("Lokalt lagret shortdata (fra SQLite)")

    if "db_data" not in st.session_state:
        st.session_state["db_data"] = pd.DataFrame()

    if st.button("📂 Hent data fra database"):
        try:
            conn = sqlite3.connect("shortsalg.db")
            df_hist = pd.read_sql("SELECT * FROM short_positions", conn)
            conn.close()
            if df_hist.empty:
                st.info("Databasen er tom. Hent og lagre data først.")
            else:
                st.session_state["db_data"] = df_hist
                st.success(f"Fant {len(df_hist):,} rader ✅")
        except Exception as e:
            st.error(f"Feil ved lesing av database: {e}")

    if not st.session_state["db_data"].empty:
        df_hist = st.session_state["db_data"]

        with st.expander("⚡ Hurtig-innsikt: Største endringer / nye posisjoner", expanded=False):
            colA, colB = st.columns(2)
            with colA:
                endringer = beregn_storste_endringer(df_hist)
                st.markdown("### Største endringer (siste vs forrige dato)")
                st.dataframe(
                    endringer[["issuerName", "shortPercent", "endring", "date"]].rename(
                        columns={"issuerName": "Selskap", "shortPercent": "Short %", "endring": "Endring", "date": "Dato"}
                    ).head(20),
                    use_container_width=True,
                )
            with colB:
                nye = finn_nye_shortposisjoner(df_hist, terskel=0.5)
                st.markdown("### Nye shortposisjoner (>= 0,5%)")
                st.dataframe(
                    nye[["issuerName", "shortPercent", "forrige_short", "date"]].rename(
                        columns={"issuerName": "Selskap", "shortPercent": "Short %", "forrige_short": "Forrige %", "date": "Dato"}
                    ).head(20),
                    use_container_width=True,
                )

        st.markdown("### Søk og filtrering")
        søkbare = sorted(set(df_hist["issuerName"].dropna().tolist() + df_hist["isin"].dropna().tolist()))
        valgt_søk = st.selectbox("Velg eller søk (autocomplete)", ["(Alle)"] + søkbare, index=0, key="db_autocomplete")

        df_søk = df_hist.copy()
        if valgt_søk != "(Alle)":
            df_søk = df_hist[
                df_hist["issuerName"].str.contains(valgt_søk, case=False, na=False)
                | df_hist["isin"].str.contains(valgt_søk, case=False, na=False)
            ]

        utstedere = sorted(df_søk["issuerName"].dropna().unique().tolist())
        valgte = st.multiselect(
            "Velg ett eller flere selskaper (kan kombineres med søk over)",
            utstedere,
            default=utstedere[:1] if utstedere else None,
            key="db_multiselect",
        )

        df_vis = df_søk[df_søk["issuerName"].isin(valgte)] if valgte else df_søk
        st.dataframe(df_vis.head(1000), use_container_width=True)

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

    try:
        conn = sqlite3.connect("shortsalg.db")
        df_all = pd.read_sql("SELECT * FROM short_positions", conn)
        conn.close()
    except Exception as e:
        st.error(f"Kunne ikke lese fra database: {e}")
        st.stop()

    if df_all.empty:
        st.info("Ingen data i databasen. Gå til «Live-data» og hent først.")
        st.stop()

    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")

    # Hurtig innsikt på hele databasen
    with st.expander("⚡ Hurtig-innsikt på hele markedet", expanded=False):
        colA, colB = st.columns(2)
        with colA:
            endr = beregn_storste_endringer(df_all)
            st.markdown("### Største endringer (siste vs forrige dato)")
            st.dataframe(
                endr[["issuerName", "shortPercent", "endring", "date"]].rename(
                    columns={"issuerName": "Selskap", "shortPercent": "Short %", "endring": "Endring", "date": "Dato"}
                ).head(20),
                use_container_width=True,
            )
        with colB:
            nye = finn_nye_shortposisjoner(df_all, terskel=0.5)
            st.markdown("### Nye shortposisjoner (>= 0,5%)")
            st.dataframe(
                nye[["issuerName", "shortPercent", "forrige_short", "date"]].rename(
                    columns={"issuerName": "Selskap", "shortPercent": "Short %", "forrige_short": "Forrige %", "date": "Dato"}
                ).head(20),
                use_container_width=True,
            )

    # Velg periode
    periodevalg = st.selectbox("Velg tidsperiode", ["30 dager", "90 dager", "180 dager", "365 dager"], index=0)
    antall_dager = int(periodevalg.split()[0])
    start_dato = pd.Timestamp.today() - pd.Timedelta(days=antall_dager)
    df_recent = df_all[df_all["date"] >= start_dato]

    if df_recent.empty:
        st.warning("Ingen data for valgt periode.")
        st.stop()

    # --- Topp 10 shortede selskaper ---
    df_top10 = (
        df_recent.groupby("issuerName")["shortPercent"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    st.markdown(f"### Topp 10 shortede selskaper (siste {antall_dager} dager)")

    csv_top10 = df_top10.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Last ned Topp 10 som CSV",
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
    st.dataframe(df_top10, use_container_width=True)

    topp10_liste = df_top10["issuerName"].tolist()
    df_utv = (
        df_recent[df_recent["issuerName"].isin(topp10_liste)]
        .groupby(["issuerName", "date"])["shortPercent"]
        .mean()
        .reset_index()
    )

    siste = df_utv[df_utv["date"] == df_utv["date"].max()]
    første = df_utv[df_utv["date"] == df_utv["date"].min()]
    diff = (siste.set_index("issuerName")["shortPercent"] - første.set_index("issuerName")["shortPercent"]).sort_values(
        ascending=False
    )

    st.markdown("### Utvikling over tid for Topp 10")
    fig_utv = px.line(
        df_utv,
        x="date",
        y="shortPercent",
        color="issuerName",
        labels={"date": "Dato", "shortPercent": "Shortandel (%)", "issuerName": "Selskap"},
        title=f"Utvikling i shortandel for Topp 10 (siste {antall_dager} dager)",
    )
    fig_utv.update_layout(template="plotly_white", hovermode="x unified", height=600)
    st.plotly_chart(fig_utv, use_container_width=True)

    st.markdown("### 🔺 Endring siste periode (mest økende / fallende)")
    df_diff = pd.DataFrame({"issuerName": diff.index, "Endring siste periode (%)": diff.values})

    df_spi = df_top10.merge(df_diff, on="issuerName", how="left")
    df_spi["Short Pressure Index (0–100)"] = ((df_spi["shortPercent"] * 0.7) + (df_spi["Endring siste periode (%)"] * 3.0)).clip(
        0, 100
    )

    df_spi["Retning"] = df_spi["Endring siste periode (%)"].apply(lambda x: "🔺 Økende" if x > 0 else "🔻 Fallende")
    st.dataframe(df_spi.sort_values("Short Pressure Index (0–100)", ascending=False), use_container_width=True)

    st.markdown("### Short Pressure Index (SPI)")
    fig_spi = px.bar(
        df_spi,
        x="issuerName",
        y="Short Pressure Index (0–100)",
        color="Short Pressure Index (0–100)",
        color_continuous_scale="RdYlGn_r",
        text_auto=".1f",
        labels={"issuerName": "Selskap", "Short Pressure Index (0–100)": "Shortpress"},
        title="Short Pressure Index – kombinasjon av shortandel og endring",
    )
    fig_spi.update_layout(template="plotly_white", xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig_spi, use_container_width=True)

    st.markdown("### Short Heatmap – daglige endringer for Topp 10")
    df_heat = (
        df_utv.pivot_table(index="issuerName", columns="date", values="shortPercent")
        .diff(axis=1)
        .fillna(0)
    )

    fig_heat = px.imshow(
        df_heat,
        color_continuous_scale=["green", "black", "red"],
        aspect="auto",
        title="Daglige endringer i shortandel (grønn = dekker inn, rød = økende short)",
        labels={"x": "Dato", "y": "Selskap", "color": "Endring (%)"},
    )
    fig_heat.update_layout(template="plotly_white", height=600, xaxis_title="Dato", yaxis_title="Selskap", xaxis_tickangle=-45)
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
        Denne applikasjonen visualiserer shortposisjoner i norske børsnoterte selskaper, basert på åpne data fra
        <a href='https://ssr.finanstilsynet.no/' target='_blank'>Finanstilsynets Short Sale Register (SSR)</a>.
        <br><br>
        Målet er å gjøre shortinformasjon lettere tilgjengelig og mer oversiktlig for investorer, analytikere og studenter.
        </div>

        <div class='about-section-title'>Hovedfunksjoner</div>
        <div class='about-card'>
        <ul>
            <li>Søk, filtrer og sammenlign shortposisjoner per selskap</li>
            <li>Se topp 10 shortede selskaper med historikk og SPI</li>
            <li>Se største endringer / nye posisjoner på siste oppdaterte dato</li>
            <li>Lagre historikk lokalt i SQLite-database</li>
            <li>Visualisering med Plotly, interaktive grafer og heatmaps</li>
        </ul>
        </div>

        <div class='about-section-title'>Teknisk stack</div>
        <div class='about-card'>
        <ul>
            <li><b>Python</b> + <b>Streamlit</b> – frontend og logikk</li>
            <li><b>Pandas</b> – databehandling og analyse</li>
            <li><b>Plotly</b> – interaktive grafer og visualisering</li>
            <li><b>SQLite</b> – lokal database for lagring</li>
            <li><b>Finanstilsynet SSR</b> – datakilde</li>
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
