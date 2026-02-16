import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import plotly.express as px
from ssr_api import hent_fullt_register, lagre_i_database, hent_siste_oppdatering

st.set_page_config(page_title="Shortsalg-register", layout="wide")

st.title("Shortsalg-register fra Finanstilsynet")

# Tabs i hovedmenyen
tab_live, tab_db, tab_top10, tab_about = st.tabs(["Live-data", "Søk i selskaper på Oslo Børs", "Oversikt over de top 10 mest shortede", "Om plattformen"])


# ---------- 📈 FANEN FOR LIVE-DATA ----------
with tab_live:
    st.header("Hent hele shortregisteret")

    # Sidebar-status
    st.sidebar.markdown("### 🌀 Status for live-nedlasting")
    sidebar_status = st.sidebar.empty()

    # 1) Hent og lagre data EN gang til session_state
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
            # LAGRE TIL SESSION STATE
            st.session_state["live_df"] = df.copy()

            # Mulighet til å laste ned CSV med en gang
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Last ned som CSV", csv, "shortregister.csv", "text/csv")

    st.divider()

    # 2) VISNING / FILTRERING (ALLTID utenfor knappeblokken)
    df_live = st.session_state.get("live_df", pd.DataFrame())
    if df_live.empty:
        st.info("Ingen live-data lastet ennå. Trykk «Hent full data fra Finanstilsynet».")
    else:
        st.success(f"Live-data i minne: {len(df_live):,} rader")

        # — Søk + filter — stabil og vedvarende
        st.subheader("🔍 Søk og filtrering")

        # init session state kun én gang
        if "live_sokeord" not in st.session_state:
            st.session_state.live_sokeord = ""
        if "live_valgte_utstedere" not in st.session_state:
            st.session_state.live_valgte_utstedere = ["(Alle)"]

        # søkefelt + nullstill
        col1, col2 = st.columns([4, 1])
        with col1:
            sok = st.text_input(
                "Søk etter selskap eller ISIN",
                value=st.session_state.live_sokeord,
                placeholder="F.eks. 'Hoegh', 'MPC', 'AKER BP'...",
                key="live_sokefelt"
            )
        with col2:
            if st.button("🔄 Nullstill filter", key="live_nullstill"):
                st.session_state.live_sokeord = ""
                st.session_state.live_valgte_utstedere = ["(Alle)"]
                sok = ""

        if sok != st.session_state.live_sokeord:
            st.session_state.live_sokeord = sok

        # filtrer på søk (men bygg VALG-listen fra det filtrerte)
        df_filtered = df_live.copy()
        if st.session_state.live_sokeord.strip():
            s = st.session_state.live_sokeord.strip().lower()
            df_filtered = df_filtered[
                df_filtered["issuerName"].str.lower().str.contains(s, na=False)
                | df_filtered["isin"].str.lower().str.contains(s, na=False)
            ]

        # valg-liste
        utstedere = sorted(df_filtered["issuerName"].dropna().unique().tolist())
        alle_valg = ["(Alle)"] + utstedere

        # trim eksisterte valg hvis de ikke finnes i nye opsjoner
        st.session_state.live_valgte_utstedere = [
            v for v in st.session_state.live_valgte_utstedere if (v == "(Alle)" or v in utstedere)
        ] or ["(Alle)"]

        # multiselect med fast key, ingen default (bruker state)
        valgte = st.multiselect(
            "Velg ett eller flere selskaper",
            options=alle_valg,
            key="live_utsteder_filter"
        )

        # oppdater state hvis bruker faktisk valgte noe
        if valgte:
            st.session_state.live_valgte_utstedere = valgte

        valgte = st.session_state.live_valgte_utstedere

        # filtrer datasettet for plotting
        if "(Alle)" in valgte or not valgte:
            df_plot = df_filtered.copy()
        else:
            df_plot = df_filtered[df_filtered["issuerName"].isin(valgte)]

        # tabell (valgfritt)
        st.dataframe(df_plot.head(1000))

        # interaktiv graf
        if not df_plot.empty:
            df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")
            df_plot = (
                df_plot.groupby(["issuerName", "date"])["shortPercent"]
                .sum()
                .reset_index()
            )
            import plotly.express as px
            fig = px.line(
                df_plot, x="date", y="shortPercent", color="issuerName",
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
        st.markdown(f"**Sist oppdatert:** {siste_tid}  \n**Totalt antall rader:** {total_rader:,}")
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

        st.markdown("### 🔍 Søk og filtrering")
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
            key="db_multiselect"
        )

        df_vis = df_søk[df_søk["issuerName"].isin(valgte)] if valgte else df_søk
        st.dataframe(df_vis.head(1000))

        if not df_vis.empty:
            df_vis["date"] = pd.to_datetime(df_vis["date"], errors="coerce")
            df_plotly = (
                df_vis.groupby(["issuerName", "date"])["shortPercent"]
                .sum()
                .reset_index()
            )
            fig_plotly = px.line(
                df_plotly,
                x="date", y="shortPercent", color="issuerName",
                title=" Interaktiv utvikling i shortandel",
                markers=True,
                labels={"date":"Dato","shortPercent":"Shortandel (%)","issuerName":"Selskap"},
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

    # Eksportknapp
    csv_top10 = df_top10.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Last ned Topp 10 som CSV",
        data=csv_top10,
        file_name=f"topp10_shorts_{antall_dager}d.csv",
        mime="text/csv"
    )

    # Stolpediagram
    fig_bar = px.bar(
        df_top10,
        x="issuerName",
        y="shortPercent",
        title=f"Topp 10 shortede selskaper (siste {antall_dager} dager)",
        labels={"issuerName": "Selskap", "shortPercent": "Shortandel (%)"},
        text_auto=".2f",
        color="shortPercent",
        color_continuous_scale="Reds"
    )
    fig_bar.update_layout(template="plotly_white", xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.dataframe(df_top10, use_container_width=True)

    # ---  Utvikling over tid for Topp 10 ---
    topp10_liste = df_top10["issuerName"].tolist()
    df_utv = (
        df_recent[df_recent["issuerName"].isin(topp10_liste)]
        .groupby(["issuerName", "date"])["shortPercent"]
        .mean()
        .reset_index()
    )

    # Beregn endring siste 14 dager
    siste = df_utv[df_utv["date"] == df_utv["date"].max()]
    første = df_utv[df_utv["date"] == df_utv["date"].min()]
    diff = (
        siste.set_index("issuerName")["shortPercent"]
        - første.set_index("issuerName")["shortPercent"]
    ).sort_values(ascending=False)

    st.markdown("### Utvikling over tid for Topp 10")
    fig_utv = px.line(
        df_utv,
        x="date",
        y="shortPercent",
        color="issuerName",
        labels={"date": "Dato", "shortPercent": "Shortandel (%)", "issuerName": "Selskap"},
        title=f"Utvikling i shortandel for Topp 10 (siste {antall_dager} dager)"
    )
    fig_utv.update_layout(template="plotly_white", hovermode="x unified", height=600)
    st.plotly_chart(fig_utv, use_container_width=True)

    # Tabell for "stiger/faller mest"
    st.markdown("### 🔺 Endring siste periode (mest økende / fallende)")
    df_diff = pd.DataFrame({
        "issuerName": diff.index,
        "Endring siste periode (%)": diff.values
    })

    # --- Beregn Short Pressure Index (SPI) ---
    df_spi = df_top10.merge(df_diff, on="issuerName", how="left")
    df_spi["Short Pressure Index (0–100)"] = (
        (df_spi["shortPercent"] * 0.7) + (df_spi["Endring siste periode (%)"] * 3.0)
    ).clip(0, 100)

    df_spi["Retning"] = df_spi["Endring siste periode (%)"].apply(lambda x: "🔺 Økende" if x > 0 else "🔻 Fallende")
    st.dataframe(df_spi.sort_values("Short Pressure Index (0–100)", ascending=False), use_container_width=True)

    # Visualiser SPI som stolpediagram
    st.markdown("### Short Pressure Index (SPI)")
    fig_spi = px.bar(
        df_spi,
        x="issuerName",
        y="Short Pressure Index (0–100)",
        color="Short Pressure Index (0–100)",
        color_continuous_scale="RdYlGn_r",
        text_auto=".1f",
        labels={"issuerName": "Selskap", "Short Pressure Index (0–100)": "Shortpress"},
        title="Short Pressure Index – kombinasjon av shortandel og endring"
    )
    fig_spi.update_layout(template="plotly_white", xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig_spi, use_container_width=True)

    # ---------- SHORT HEATMAP ----------
    st.markdown("### Short Heatmap – daglige endringer for Topp 10")
    df_heat = (
        df_utv.pivot_table(
            index="issuerName",
            columns="date",
            values="shortPercent"
        )
        .diff(axis=1)
        .fillna(0)
    )

    fig_heat = px.imshow(
        df_heat,
        color_continuous_scale=["green", "black", "red"],
        aspect="auto",
        title="Daglige endringer i shortandel (grønn = dekker inn, rød = økende short)",
        labels={"x": "Dato", "y": "Selskap", "color": "Endring (%)"}
    )
    fig_heat.update_layout(
        template="plotly_white",
        height=600,
        xaxis_title="Dato",
        yaxis_title="Selskap",
        xaxis_tickangle=-45
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
            unsafe_allow_html=True
        )

        st.markdown("<div class='about-header'> Om denne plattformen</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class='about-card'>
           
            Denne applikasjonen visualiserer shortposisjoner i norske børsnoterte selskaper, basert på åpne data fra 
            <a href='https://ssr.finanstilsynet.no/' target='_blank'>Finanstilsynets Short Sale Register (SSR)</a>.</p>

            Målet er å gjøre shortinformasjon lettere tilgjengelig og mer oversiktlig for investorer, analytikere og studenter.</p>
            </div>

            <div class='about-section-title'>Hovedfunksjoner</div>
            <div class='about-card'>
            <ul>
                <li>Søk, filtrer og sammenlign shortposisjoner per selskap</li>
                <li>Se topp 10 shortede selskaper med historikk og SPI</li>
                <li>Lagre historikk lokalt i SQLite-database</li>
                <li>Visualisering med Plotly, interaktive grafer og heatmaps</li>
            </ul>
            </div>

            <div class='about-section-title'> Teknisk stack</div>
            <div class='about-card'>
            <ul>
                <li><b>Python</b> + <b>Streamlit</b> – frontend og logikk</li>
                <li><b>Pandas</b> – databehandling og analyse</li>
                <li><b>Plotly</b> – interaktive grafer og visualisering</li>
                <li><b>SQLite</b> – lokal database for lagring</li>
                <li><b>Finanstilsynet SSR API</b> – datakilde</li>
            </ul>
            </div> 

            <div class='about-section-title'> Om utvikleren</div>
            <div class='about-card'>
            <p>Utviklet av <b>Andreas Bolton Seielstad</b> </p>
            <p>Prosjektet er laget for læring, innsikt og åpenhet i finansmarkedet.<br>
            <a href='https://github.com/An-Bolton' target='_blank'> 🌐 https://github.com/An-Bolton/shortsalg-register</a>
            </div>

            <div class='about-card'>
            <p><b>Kilde:</b> <a href='https://ssr.finanstilsynet.no/api/v2/' target='_blank'>Finanstilsynet SSR API</a></p>
            </div>
            """,
            unsafe_allow_html=True
        )


