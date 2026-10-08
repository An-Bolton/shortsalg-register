import html
import pandas as pd
import plotly.express as px
import streamlit as st

from ssr_api import (
    hent_database_data,
    hent_fullt_register,
    hent_posisjonsholdere,
    hent_siste_oppdatering,
    hent_unntatte_instrumenter,
    lagre_i_database,
    tving_ny_nedlasting,
)

# -------------------- DATAHJELPERE --------------------

def _standardiser_shortpercent(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "shortPercent" not in df.columns:
        return df
    out = df.copy()
    out["shortPercent"] = pd.to_numeric(out["shortPercent"], errors="coerce")
    # Eldre SQLite-rader kan ligge i hundredels prosent (f.eks. 792 = 7,92 %),
    # mens nyere API-rader allerede er normalisert (7,92). Normaliser derfor
    # rad for rad. En global sjekk på maksimum delte tidligere også korrekte
    # rader på 100 når de to formatene forekom i samme datasett.
    raw_scale = out["shortPercent"] > 20
    out.loc[raw_scale, "shortPercent"] = (
        out.loc[raw_scale, "shortPercent"] / 100
    )
    return out


def _agg_issuer_date(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = _standardiser_shortpercent(df)
    out["date"] = (
        pd.to_datetime(out["date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    out = out.dropna(subset=["issuerName", "date", "shortPercent"])
    out = out.loc[out["shortPercent"].between(0, 100)].copy()
    return (
        out.groupby(["issuerName", "date"], as_index=False)["shortPercent"]
        # API-et inneholder allerede aggregert shortandel per instrument og
        # endringsdato. DB + live kan inneholde samme observasjon i ulik skala;
        # max hindrer at slike overlapp blir summert og dermed dobbelttelt.
        .max()
        .sort_values(["issuerName", "date"])
    )


def bygg_daglig_shortserie(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Bygger en daglig, fremoverfylt serie av rapporterte shortnivåer.

    SSR-historikken er hendelsesbasert: en rad publiseres når nivået endres.
    For et tidsvektet gjennomsnitt må siste kjente nivå derfor gjelde frem til
    neste endring. Selskaper med en uendret posisjon gjennom hele perioden tas
    med ved å hente siste observasjon før startdatoen.
    """
    columns = ["issuerName", "date", "shortPercent"]
    data = _agg_issuer_date(df)
    if data.empty:
        return pd.DataFrame(columns=columns)

    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame(columns=columns)

    start = pd.Timestamp(start).tz_localize(None).normalize()
    end = pd.Timestamp(end).tz_localize(None).normalize()
    if start > end:
        return pd.DataFrame(columns=columns)

    # Fremtidige observasjoner skal ikke påvirke analysevinduet. Alle utstedere
    # som har hatt en observasjon innen sluttdatoen kan derimot ha et nivå som
    # fortsatt gjelder i perioden, selv om siste endring skjedde før start.
    data = data.loc[data["date"] <= end].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)

    day_index = pd.date_range(start=start, end=end, freq="D")
    daily_frames = []

    for issuer, issuer_data in data.groupby("issuerName", sort=False):
        states = (
            issuer_data.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .set_index("date")["shortPercent"]
        )
        expanded_index = states.index.union(day_index).sort_values()
        daily_values = (
            states.reindex(expanded_index)
            .ffill()
            .reindex(day_index)
            .fillna(0.0)
        )
        daily_frames.append(
            pd.DataFrame(
                {
                    "issuerName": issuer,
                    "date": day_index,
                    "shortPercent": daily_values.to_numpy(dtype=float),
                }
            )
        )

    return pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(columns=columns)


def hent_siste_posisjon_per_selskap(df: pd.DataFrame) -> pd.DataFrame:
    """Returnerer siste registrerte, aggregerte shortandel for hvert selskap."""
    data = _agg_issuer_date(df)
    if data.empty:
        return data

    return (
        data.sort_values(["issuerName", "date"])
        .groupby("issuerName", as_index=False)
        .tail(1)
        .sort_values(["shortPercent", "issuerName"], ascending=[False, True])
        .reset_index(drop=True)
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


def vis_posisjonsholdere(df: pd.DataFrame, key_prefix: str = "holders") -> None:
    """Viser individuelle offentlige posisjonsholdere uten å påvirke aggregert historikk."""
    _render_table_header(
        "Aktive posisjonsholdere",
        "Individuelle offentlige posisjoner fra Finanstilsynets activePositions. "
        "Holdes separat fra aggregert historikk for å unngå dobbelttelling.",
        "LIVE REGISTER",
    )

    if df is None or df.empty:
        st.info("Ingen individuelle posisjonsholdere tilgjengelig akkurat nå.")
        return

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["shortPercent"] = pd.to_numeric(data["shortPercent"], errors="coerce")
    data["shares"] = pd.to_numeric(data.get("shares"), errors="coerce")
    data = data.dropna(subset=["issuerName", "positionHolder", "date", "shortPercent"])

    search = st.text_input(
        "Søk etter selskap, ISIN eller posisjonsholder",
        placeholder="F.eks. EQUINOR, NO0010096985 eller Marshall Wace",
        key=f"{key_prefix}_search",
    ).strip()

    if search:
        mask = (
            data["issuerName"].fillna("").astype(str).str.contains(search, case=False, na=False, regex=False)
            | data["isin"].fillna("").astype(str).str.contains(search, case=False, na=False, regex=False)
            | data["positionHolder"].fillna("").astype(str).str.contains(search, case=False, na=False, regex=False)
        )
        data = data.loc[mask]

    newest_only = st.toggle(
        "Kun siste registrerte posisjon per selskap og posisjonsholder",
        value=True,
        key=f"{key_prefix}_latest",
    )
    if newest_only and not data.empty:
        data = (
            data.sort_values("date")
            .groupby(["issuerName", "positionHolder"], as_index=False)
            .tail(1)
        )

    data = data.sort_values(["date", "shortPercent"], ascending=[False, False])
    view = data.rename(
        columns={
            "issuerName": "Selskap",
            "positionHolder": "Posisjonsholder",
            "shortPercent": "Short %",
            "shares": "Aksjer",
            "isin": "ISIN",
        }
    ).copy()
    view["Dato"] = view["date"].dt.strftime("%d.%m.%Y")
    # Legg de viktigste kolonnene først. På smale skjermer er dermed
    # shortandelen synlig før de brede detaljkolonnene.
    view = view[["Selskap", "Short %", "Posisjonsholder", "Dato", "Aksjer", "ISIN"]]

    st.dataframe(
        view,
        width="stretch",
        hide_index=True,
        column_config={
            "Selskap": st.column_config.TextColumn("Selskap", width="medium"),
            "Posisjonsholder": st.column_config.TextColumn("Posisjonsholder", width="medium"),
            "Dato": st.column_config.TextColumn("Dato", width="small"),
            "Short %": st.column_config.NumberColumn(
                "Short %", format="%.2f %%", width="small"
            ),
            "Aksjer": st.column_config.NumberColumn(
                "Aksjer", format="%d", width="small"
            ),
            "ISIN": st.column_config.TextColumn("ISIN", width="medium"),
        },
    )

    st.download_button(
        "Last ned posisjonsholdere som CSV",
        data=dataframe_to_csv(view),
        file_name="short_posisjonsholdere.csv",
        mime="text/csv",
        key=f"{key_prefix}_download",
    )


def vis_hurtiginnsikt(df: pd.DataFrame, expanded: bool = False) -> None:
    with st.expander("Hurtig-innsikt: største endringer og nye posisjoner", expanded=expanded):
        # Fullbredde tabeller er mer robuste enn to smale sidekolonner. Det hindrer
        # at dato/verdier klippes på små bærbare skjermer og ved nettleser-zoom.
        changes = beregn_storste_endringer(df)
        new_positions = finn_nye_shortposisjoner(df)

        st.markdown(
            f"""
            <div class="quick-summary">
                <div><span>BEVEGELSER</span><strong>{len(changes):,}</strong><small>selskaper med målt endring</small></div>
                <div><span>NYE OVER TERSKEL</span><strong>{len(new_positions):,}</strong><small>siste registrerte kryssing</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _render_table_header(
            "Største endringer",
            "Siste endring mot foregående registrerte nivå. Økning vises rødt, reduksjon grønt.",
            "TOPP 10",
        )

        if changes.empty:
            st.info("Ingen endringer å vise.")
        else:
            changes_view = changes.copy()
            changes_view["Retning"] = changes_view["endring"].apply(
                lambda value: "▲ Økning" if value > 0 else "▼ Reduksjon"
            )
            changes_view["Fra → til"] = changes_view.apply(
                lambda row: f"{row['forrige_short']:.2f} % → {row['shortPercent']:.2f} %",
                axis=1,
            )
            changes_view["date"] = pd.to_datetime(
                changes_view["date"], errors="coerce"
            ).dt.strftime("%d.%m.%Y")
            changes_view = (
                changes_view[
                    ["issuerName", "Retning", "Fra → til", "endring", "date"]
                ]
                .rename(
                    columns={
                        "issuerName": "Selskap",
                        "endring": "Endring (pp)",
                        "date": "Dato",
                    }
                )
                .head(10)
            )

            st.dataframe(
                changes_view,
                width="stretch",
                hide_index=True,
                height=42 + 35 * (len(changes_view) + 1),
                column_config={
                    "Selskap": st.column_config.TextColumn("Selskap", width="large"),
                    "Retning": st.column_config.TextColumn("Retning", width="small"),
                    "Fra → til": st.column_config.TextColumn("Fra → til", width="medium"),
                    "Endring (pp)": st.column_config.NumberColumn(
                        "Endring (pp)", format="%+.2f", width="small"
                    ),
                    "Dato": st.column_config.TextColumn("Dato", width="small"),
                },
            )

        _render_table_header(
            "Nye posisjoner over 0,5 %",
            "Selskaper som sist krysset offentlig rapporteringsterskel.",
            "NYE",
        )

        if new_positions.empty:
            st.info("Ingen nye posisjoner å vise.")
        else:
            new_positions_view = new_positions.copy()
            new_positions_view["forrige_short"] = new_positions_view[
                "forrige_short"
            ].fillna(0.0)
            new_positions_view["Fra → til"] = new_positions_view.apply(
                lambda row: f"{row['forrige_short']:.2f} % → {row['shortPercent']:.2f} %",
                axis=1,
            )
            new_positions_view["date"] = pd.to_datetime(
                new_positions_view["date"], errors="coerce"
            ).dt.strftime("%d.%m.%Y")
            new_positions_view = (
                new_positions_view[
                    ["issuerName", "Fra → til", "shortPercent", "date"]
                ]
                .rename(
                    columns={
                        "issuerName": "Selskap",
                        "shortPercent": "Ny short %",
                        "date": "Dato",
                    }
                )
                .head(10)
            )

            st.dataframe(
                new_positions_view,
                width="stretch",
                hide_index=True,
                height=42 + 35 * (len(new_positions_view) + 1),
                column_config={
                    "Selskap": st.column_config.TextColumn("Selskap", width="large"),
                    "Fra → til": st.column_config.TextColumn("Fra → til", width="medium"),
                    "Ny short %": st.column_config.NumberColumn(
                        "Ny short %", format="%.2f %%", width="small"
                    ),
                    "Dato": st.column_config.TextColumn("Dato", width="small"),
                },
            )


def vis_sok_og_graf(
    df: pd.DataFrame,
    key_prefix: str,
    exempted: pd.DataFrame | None = None,
) -> None:
    if df.empty:
        st.info("Ingen data tilgjengelig.")
        return

    required = {"issuerName", "isin", "date", "shortPercent"}
    missing = sorted(required.difference(df.columns))
    if missing:
        st.error("Dataene mangler kolonnene: " + ", ".join(missing))
        return

    _render_table_header(
        "Søk og filtrering",
        "Avgrens registeret på selskap eller ISIN, velg visning og eksporter resultatet.",
        "REGISTER",
    )

    filter_panel = st.container(key=f"{key_prefix}_filter_panel")
    with filter_panel:
        search = st.text_input(
            "Søk etter selskap eller ISIN",
            placeholder="F.eks. EQUINOR, MPC eller NO0010096985",
            key=f"{key_prefix}_search",
        ).strip()

    filtered = df.copy()
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

    if search and filtered.empty:
        exempt_match = pd.DataFrame()
        if exempted is not None and not exempted.empty:
            exempt_mask = (
                exempted["issuerName"].fillna("").astype(str).str.contains(
                    search, case=False, na=False, regex=False
                )
                | exempted["isin"].fillna("").astype(str).str.contains(
                    search, case=False, na=False, regex=False
                )
            )
            exempt_match = exempted.loc[exempt_mask].copy()

        if not exempt_match.empty:
            names = ", ".join(exempt_match["issuerName"].astype(str).tolist())
            st.warning(
                f"{names} finnes ikke i shortdataene fordi aksjen er unntatt "
                "SSR-rapportering. Manglende tall betyr derfor ikke 0 % short."
            )
        else:
            st.info(
                "Ingen offentlig rapporterbar shortposisjon ble funnet. Registeret er "
                "ikke en komplett selskapsliste for Oslo Børs: posisjoner under 0,5 % "
                "er ikke med, og fravær skal ikke tolkes som 0 % short."
            )
        return

    issuers = sorted(filtered["issuerName"].dropna().astype(str).unique().tolist())
    with filter_panel:
        selected = st.multiselect(
            "Velg ett eller flere selskaper",
            options=issuers,
            default=issuers[:1] if search and issuers else [],
            key=f"{key_prefix}_issuers",
            placeholder="Velg selskaper – tomt valg viser alle",
        )

    shown = (
        filtered.loc[filtered["issuerName"].astype(str).isin(selected)].copy()
        if selected
        else filtered.copy()
    )
    if shown.empty:
        st.info("Ingen treff for søket eller filteret.")
        return

    shown = _standardiser_shortpercent(shown)
    shown["date"] = pd.to_datetime(shown["date"], errors="coerce")
    shown = shown.dropna(subset=["issuerName", "date", "shortPercent"])

    # Beregn endring mot forrige registrerte nivå for hvert selskap.
    shown = shown.sort_values(["issuerName", "date"])
    shown["Endring (pp)"] = shown.groupby("issuerName")["shortPercent"].diff()
    shown["Trend"] = shown["Endring (pp)"].apply(
        lambda value: (
            "▲ Økning" if pd.notna(value) and value > 0
            else "▼ Reduksjon" if pd.notna(value) and value < 0
            else "— Uendret"
        )
    )
    shown = shown.sort_values(["date", "issuerName"], ascending=[False, True])

    with filter_panel:
        controls_left, controls_middle, controls_right = st.columns([1.25, 1, 1])
        with controls_left:
            advanced = st.toggle(
                "Vis avanserte kolonner",
                value=False,
                key=f"{key_prefix}_advanced_columns",
            )
        with controls_middle:
            max_rows = st.selectbox(
                "Rader i tabellen",
                options=[25, 50, 100, 250, 500, 1000],
                index=3,
                key=f"{key_prefix}_max_rows",
            )
        with controls_right:
            newest_only = st.toggle(
                "Kun siste rad per selskap",
                value=False,
                key=f"{key_prefix}_latest_only",
            )

    if newest_only:
        shown = shown.sort_values("date").groupby("issuerName", as_index=False).tail(1)
        shown = shown.sort_values(["shortPercent", "issuerName"], ascending=[False, True])

    shown["Posisjonsholder"] = (
        shown.get("positionHolder", pd.Series(index=shown.index, dtype="object"))
        .fillna("Aggregert")
        .replace({"None": "Aggregert", "": "Aggregert"})
    )
    shown["Dato"] = shown["date"].dt.strftime("%d.%m.%Y")
    shown["Selskap"] = shown["issuerName"].astype(str)
    shown["Short %"] = shown["shortPercent"]
    shown["ISIN"] = shown["isin"].fillna("—").astype(str)

    if "shares" in shown.columns:
        shown["Aksjer"] = pd.to_numeric(shown["shares"], errors="coerce")
    else:
        shown["Aksjer"] = pd.NA

    # Standardvisningen er bevisst kompakt nok for mobil. Trend viser samme
    # retning som Endring (pp), og ligger derfor under avanserte kolonner.
    base_columns = ["Selskap", "Short %", "Endring (pp)", "Dato"]
    advanced_columns = ["Trend", "ISIN", "Posisjonsholder", "Aksjer"]
    display_columns = base_columns + advanced_columns if advanced else base_columns
    table_view = shown[display_columns].head(max_rows)

    info_left, info_right = st.columns([2, 1])
    with info_left:
        st.caption(
            f"Viser {len(table_view):,} av {len(shown):,} filtrerte rader "
            f"({len(df):,} rader totalt)."
        )
    with info_right:
        st.download_button(
            "Eksporter viste data",
            data=dataframe_to_csv(table_view),
            file_name=f"shortposisjoner_{key_prefix}.csv",
            mime="text/csv",
            key=f"{key_prefix}_export_table",
            width="stretch",
        )

    _render_table_header(
        "Shortposisjoner",
        "Klikk på kolonneoverskriftene for å sortere. Tabellen følger filtrene over.",
        f"{len(table_view):,} RADER",
    )

    column_config = {
        "Selskap": st.column_config.TextColumn("Selskap", width="medium"),
        "Dato": st.column_config.TextColumn("Dato", width="small"),
        "Short %": st.column_config.NumberColumn(
            "Short %", format="%.2f %%", width="small"
        ),
        "Endring (pp)": st.column_config.NumberColumn(
            "Endring", format="%+.2f", width="small"
        ),
        "Trend": st.column_config.TextColumn("Trend", width="small"),
        "ISIN": st.column_config.TextColumn("ISIN", width="medium"),
        "Posisjonsholder": st.column_config.TextColumn("Posisjonsholder", width="medium"),
        "Aksjer": st.column_config.NumberColumn(
            "Aksjer", format="%d", width="small"
        ),
    }

    st.dataframe(
        table_view,
        width="stretch",
        hide_index=True,
        column_config=column_config,
        height=min(760, 42 + 35 * min(len(table_view), 20)),
    )

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
            color_discrete_sequence=CHART_PALETTE,
        )
        _style_plotly_chart(fig, height=600, hovermode="x unified")
        fig.update_layout(legend_title_text="Utsteder")
        fig.update_traces(line=dict(width=2.8), marker=dict(size=7, line=dict(width=1, color="#FFFFFF")))
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_short_chart")


CHART_PALETTE = [
    "#1D4ED8",
    "#0F766E",
    "#475569",
    "#7C3AED",
    "#B45309",
    "#BE123C",
    "#0369A1",
    "#4D7C0F",
    "#9D174D",
    "#334155",
]


def _style_plotly_chart(fig, height: int = 560, hovermode: str | None = None):
    """Gir alle diagrammer et lyst, konsistent Shortregister-uttrykk."""
    layout = dict(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        colorway=CHART_PALETTE,
        font=dict(family="Inter, Arial, sans-serif", color="#172033", size=12),
        title=dict(font=dict(size=19, color="#0B1426"), x=0.02, xanchor="left"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.86)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="#0B1426",
            bordercolor="#24344F",
            font=dict(color="#FFFFFF", family="Inter, Arial, sans-serif"),
        ),
        margin=dict(l=34, r=24, t=76, b=42),
    )
    if hovermode:
        layout["hovermode"] = hovermode
    fig.update_layout(**layout)
    fig.update_xaxes(
        showgrid=False,
        linecolor="#CBD5E1",
        tickfont=dict(color="#64748B"),
        title_font=dict(color="#475569"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#EEF2F7",
        zerolinecolor="#CBD5E1",
        linecolor="#CBD5E1",
        tickfont=dict(color="#64748B"),
        title_font=dict(color="#475569"),
    )
    return fig


def _render_section_header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-head">
            <div class="section-head-kicker">{html.escape(kicker)}</div>
            <div class="section-head-title">{html.escape(title)}</div>
            <div class="section-head-copy">{html.escape(description)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_table_header(title: str, description: str, badge: str = "") -> None:
    badge_html = (
        f'<span class="table-head-badge">{html.escape(badge)}</span>'
        if badge
        else ""
    )
    st.markdown(
        f"""
        <div class="table-head">
            <div>
                <div class="table-head-title">{html.escape(title)}</div>
                <div class="table-head-copy">{html.escape(description)}</div>
            </div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------- APP --------------------
st.set_page_config(
    page_title="Shortregister | Markedsdata",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --canvas: #F4F7FB;
        --surface: #FFFFFF;
        --ink: #0B1426;
        --ink-2: #172033;
        --navy: #0D203A;
        --blue: #2563EB;
        --cyan: #06B6D4;
        --mint: #19C99A;
        --amber: #F6C453;
        --red: #EF4444;
        --muted: #64748B;
        --line: #DDE5EF;
        --shadow: 0 18px 48px rgba(15, 35, 65, 0.10);
    }

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    *, *::before, *::after {
        box-sizing: border-box;
    }

    .stApp {
        background:
            radial-gradient(circle at 7% -8%, rgba(37, 99, 235, 0.10), transparent 25rem),
            radial-gradient(circle at 96% 5%, rgba(6, 182, 212, 0.08), transparent 27rem),
            linear-gradient(rgba(15, 35, 65, 0.018) 1px, transparent 1px),
            linear-gradient(90deg, rgba(15, 35, 65, 0.018) 1px, transparent 1px),
            var(--canvas);
        background-size: auto, auto, 32px 32px, 32px 32px, auto;
        color: var(--ink-2);
    }

    [data-testid="stHeader"] {
        background: rgba(244, 247, 251, 0.82);
        backdrop-filter: blur(18px) saturate(145%);
        border-bottom: 1px solid rgba(148, 163, 184, 0.22);
    }

    [data-testid="stToolbar"] { right: 1rem; }
    #MainMenu, footer { visibility: hidden; }

    section.main > div.block-container, .block-container {
        padding-top: 1.35rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        padding-bottom: 4rem !important;
        max-width: 1520px !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        isolation: isolate;
        border: 1px solid rgba(113, 190, 255, 0.20);
        border-radius: 30px;
        padding: 38px 40px 24px;
        margin: 4px 0 18px;
        background:
            radial-gradient(circle at 82% 12%, rgba(6, 182, 212, 0.24), transparent 24rem),
            radial-gradient(circle at 14% 115%, rgba(37, 99, 235, 0.34), transparent 28rem),
            linear-gradient(132deg, #07111F 0%, #0A1E37 52%, #0B3448 100%);
        box-shadow: 0 28px 80px rgba(6, 20, 42, 0.30);
    }

    .hero:before {
        content: "";
        position: absolute;
        inset: 0;
        z-index: -1;
        opacity: 0.38;
        background-image:
            linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
        background-size: 34px 34px;
        mask-image: linear-gradient(to right, black, transparent 78%);
    }

    .hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.55fr) minmax(300px, 0.72fr);
        gap: 44px;
        align-items: center;
    }

    .brand-lockup {
        display: inline-flex;
        align-items: center;
        gap: 11px;
        margin-bottom: 24px;
    }

    .brand-mark {
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        border: 1px solid rgba(103, 232, 249, 0.42);
        border-radius: 11px;
        background: linear-gradient(145deg, rgba(37,99,235,.42), rgba(6,182,212,.12));
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 950;
        letter-spacing: -0.04em;
        box-shadow: inset 0 1px rgba(255,255,255,.18), 0 9px 28px rgba(6,182,212,.12);
    }

    .brand-name {
        color: #FFFFFF;
        font-size: 0.82rem;
        font-weight: 900;
        letter-spacing: 0.14em;
    }

    .brand-sub {
        display: block;
        color: #7DD3FC;
        margin-top: 2px;
        font-size: 0.61rem;
        font-weight: 750;
        letter-spacing: 0.16em;
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        color: #A5F3FC;
        font-size: 0.73rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 13px;
    }

    .hero-kicker:before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--mint);
        box-shadow: 0 0 0 5px rgba(25,201,154,.10), 0 0 18px rgba(25,201,154,.75);
    }

    .hero .hero-title {
        color: #FFFFFF !important;
        font-size: 5.25rem;
        font-size: clamp(3rem, 6vw, 5.25rem);
        line-height: 0.92;
        margin: 0;
        font-weight: 950;
        letter-spacing: -0.065em;
    }

    .hero .hero-title span { color: #67E8F9 !important; }

    .hero-lead {
        color: #D9EAFE !important;
        font-size: 1.55rem;
        font-size: clamp(1.25rem, 2.2vw, 1.72rem);
        line-height: 1.23;
        font-weight: 760;
        letter-spacing: -0.025em;
        margin: 18px 0 0;
    }

    .hero-lead span { color: #67E8F9; }

    .hero-copy {
        color: #9EB1C9 !important;
        font-size: 0.98rem;
        max-width: 690px;
        margin: 15px 0 0;
        line-height: 1.62;
    }

    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 21px;
    }

    .hero-badge {
        padding: 7px 11px;
        border-radius: 999px;
        border: 1px solid rgba(125, 211, 252, 0.18);
        background: rgba(7, 22, 42, 0.48);
        color: #D7E6F8;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.035em;
    }

    .terminal-card {
        position: relative;
        padding: 18px;
        border: 1px solid rgba(125, 211, 252, 0.20);
        border-radius: 21px;
        background: rgba(3, 14, 29, 0.54);
        box-shadow: inset 0 1px rgba(255,255,255,.06), 0 24px 55px rgba(0,0,0,.22);
        backdrop-filter: blur(18px);
    }

    .terminal-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 13px;
        border-bottom: 1px solid rgba(148,163,184,.16);
        color: #7DD3FC;
        font-size: 0.67rem;
        font-weight: 900;
        letter-spacing: 0.15em;
    }

    .terminal-lights { display: inline-flex; gap: 5px; }

    .terminal-lights i {
        display: block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #334155;
    }

    .terminal-lights i:first-child { background: #19C99A; }
    .terminal-lights i:nth-child(2) { background: #F6C453; }
    .terminal-lights i:last-child { background: #2563EB; }

    .pipeline {
        display: grid;
        gap: 0;
        margin: 15px 0 4px;
    }

    .pipeline-step {
        position: relative;
        display: grid;
        grid-template-columns: 28px 1fr;
        gap: 11px;
        align-items: center;
        min-height: 42px;
        color: #DCEBFA;
        font-size: 0.79rem;
        font-weight: 730;
    }

    .pipeline-step:not(:last-child):after {
        content: "";
        position: absolute;
        left: 13px;
        top: 31px;
        width: 1px;
        height: 21px;
        background: linear-gradient(#38BDF8, rgba(56,189,248,.08));
    }

    .pipeline-step b {
        display: grid;
        place-items: center;
        width: 28px;
        height: 28px;
        border-radius: 9px;
        background: rgba(37,99,235,.18);
        border: 1px solid rgba(96,165,250,.24);
        color: #67E8F9;
        font-size: 0.66rem;
        letter-spacing: 0.03em;
    }

    .terminal-meta {
        display: flex;
        justify-content: space-between;
        gap: 14px;
        margin-top: 14px;
        padding-top: 14px;
        border-top: 1px solid rgba(148,163,184,.16);
        color: #8296AF;
        font-size: 0.7rem;
    }

    .terminal-meta strong { color: #DCEBFA; font-weight: 800; }

    .hero-foot {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        margin-top: 28px;
        padding-top: 17px;
        border-top: 1px solid rgba(148,163,184,.14);
        color: #7E91A8;
        font-size: 0.69rem;
        font-weight: 750;
        letter-spacing: 0.045em;
        text-transform: uppercase;
    }

    div[data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        min-height: 126px;
        background: rgba(255,255,255,0.95);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 20px 21px;
        box-shadow: 0 12px 32px rgba(15,35,65,.075);
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
    }

    div[data-testid="stMetric"]:before {
        content: "";
        position: absolute;
        inset: 0 0 auto 0;
        height: 3px;
        background: linear-gradient(90deg, var(--blue), var(--cyan), var(--mint));
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #B9CCEA;
        box-shadow: 0 18px 42px rgba(15,35,65,.12);
    }

    div[data-testid="stMetric"] label {
        color: var(--muted) !important;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.075em;
        font-weight: 850 !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-size: clamp(1.55rem, 2.3vw, 2rem) !important;
        font-weight: 950 !important;
        letter-spacing: -0.035em;
    }

    div[data-testid="stMetricDelta"] {
        font-weight: 800 !important;
        color: #2563EB !important;
    }

    .insight-card {
        position: relative;
        overflow: hidden;
        min-height: 190px;
        padding: 23px 24px;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.96);
        box-shadow: 0 12px 32px rgba(15,35,65,.075);
        transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    }

    .insight-card:before {
        content: "";
        position: absolute;
        inset: 0 0 auto 0;
        height: 4px;
        background: var(--blue);
    }

    .insight-card--down:before { background: var(--mint); }
    .insight-card--up:before { background: var(--red); }
    .insight-card--new:before { background: linear-gradient(90deg, var(--blue), var(--cyan)); }

    .insight-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 42px rgba(15,35,65,.13);
        border-color: #B9CCEA;
    }

    .insight-kicker {
        color: var(--muted);
        font-size: 0.7rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 13px;
    }

    .insight-value {
        color: var(--ink);
        font-size: 1.88rem;
        line-height: 1.1;
        font-weight: 950;
        letter-spacing: -0.035em;
        margin-bottom: 10px;
    }

    .insight-card--down .insight-value { color: #0F9F76; }
    .insight-card--up .insight-value { color: #DC2626; }

    .insight-company {
        color: #1D4ED8;
        font-size: 1rem;
        line-height: 1.3;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .insight-detail {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
    }

    .freshness-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px 20px;
        margin: -4px 0 17px;
        padding: 11px 15px;
        border: 1px solid #D7E2F0;
        border-radius: 13px;
        background: rgba(255,255,255,.78);
        color: #64748B;
        font-size: 0.74rem;
        font-weight: 730;
        backdrop-filter: blur(12px);
    }

    .freshness-status {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        color: #047857;
        font-weight: 900;
    }

    .freshness-status:before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--mint);
        box-shadow: 0 0 0 4px rgba(25,201,154,.10);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        background: rgba(255,255,255,0.88) !important;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 6px !important;
        margin-bottom: 20px;
        box-shadow: 0 9px 28px rgba(15,35,65,.07);
        backdrop-filter: blur(16px);
    }

    .stTabs [data-baseweb="tab"] {
        min-height: 45px !important;
        padding: 0 20px !important;
        border-radius: 10px !important;
        color: #526077 !important;
        font-weight: 850 !important;
        transition: all .2s ease;
    }

    .stTabs [data-baseweb="tab"] p {
        color: inherit !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.025em;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: #EEF4FF !important;
        color: #1D4ED8 !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #0D203A, #154872) !important;
        color: white !important;
        box-shadow: 0 9px 24px rgba(13,32,58,.22);
    }

    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        border-radius: 18px !important;
    }

    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.94);
        border: 1px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 9px 26px rgba(15,35,65,.055);
    }

    [data-testid="stDataFrame"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 12px 32px rgba(15,35,65,.075);
    }

    [data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 12px;
        box-shadow: 0 14px 36px rgba(15,35,65,.08);
        overflow: hidden;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    [data-testid="stTextInput"] input {
        background: white !important;
        border-color: #CBD8E8 !important;
        color: var(--ink) !important;
        border-radius: 11px !important;
        box-shadow: 0 5px 16px rgba(15,35,65,.04) !important;
    }

    .stButton > button {
        border: 1px solid rgba(72, 145, 203, 0.32) !important;
        background: linear-gradient(135deg, #0D203A, #174B78) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.05rem !important;
        font-weight: 850 !important;
        box-shadow: 0 9px 24px rgba(13,32,58,.18);
        transition: transform .2s ease, box-shadow .2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        background: linear-gradient(135deg, #174B78, #2563EB) !important;
        box-shadow: 0 14px 30px rgba(37,99,235,.22);
        border-color: rgba(56,189,248,.55) !important;
    }

    .stDownloadButton > button {
        border: 1px solid #C9D6E6 !important;
        background: rgba(255,255,255,.92) !important;
        color: #17365D !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.05rem !important;
        font-weight: 850 !important;
        box-shadow: 0 7px 20px rgba(15,35,65,.07);
        transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        color: #1D4ED8 !important;
        border-color: #93B4E5 !important;
        box-shadow: 0 11px 26px rgba(15,35,65,.11);
    }

    .st-key-refresh_action .stButton > button {
        min-height: 54px;
        border-color: rgba(103,232,249,.46) !important;
        background: linear-gradient(135deg, #0E7490, #2563EB) !important;
        box-shadow: 0 13px 32px rgba(14,116,144,.24);
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.94);
        box-shadow: 0 8px 24px rgba(15,35,65,.05);
    }

    h1, h2, h3 {
        color: var(--ink) !important;
        letter-spacing: -0.032em;
    }

    .section-head {
        position: relative;
        overflow: hidden;
        padding: 24px 26px 23px 29px;
        margin: 4px 0 18px 0;
        border-radius: 20px;
        border: 1px solid var(--line);
        background:
            radial-gradient(circle at 94% 8%, rgba(6,182,212,.09), transparent 25%),
            linear-gradient(135deg, rgba(255,255,255,.98), rgba(247,250,255,.96));
        box-shadow: 0 12px 34px rgba(15,35,65,.075);
    }

    .section-head:before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 5px;
        background: linear-gradient(var(--blue), var(--cyan));
    }

    .section-head-kicker {
        color: #2563EB;
        font-size: 0.68rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-bottom: 7px;
    }

    .section-head-title {
        color: var(--ink);
        font-size: 2.2rem;
        font-size: clamp(1.65rem, 3vw, 2.45rem);
        line-height: 1.05;
        font-weight: 950;
        letter-spacing: -0.045em;
        margin: 0;
    }

    .section-head-copy {
        color: var(--muted);
        font-size: 0.93rem;
        line-height: 1.55;
        margin-top: 9px;
        max-width: 850px;
    }

    p, label, .stCaption, [data-testid="stCaptionContainer"] { color: #526077; }
    hr { border-color: rgba(148, 163, 184, 0.20) !important; }

    .about-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 15px;
        margin: 4px 0 18px;
    }

    .about-card {
        min-height: 205px;
        padding: 22px;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(255,255,255,.94);
        box-shadow: 0 12px 30px rgba(15,35,65,.065);
    }

    .about-card-number {
        color: #2563EB;
        font-size: 0.68rem;
        font-weight: 950;
        letter-spacing: 0.14em;
    }

    .about-card h3 {
        margin: 13px 0 8px;
        font-size: 1.15rem;
    }

    .about-card p {
        margin: 0;
        color: var(--muted);
        font-size: 0.87rem;
        line-height: 1.65;
    }

    .about-card a {
        color: #1D4ED8;
        text-decoration: none;
        font-weight: 800;
    }

    .method-note {
        padding: 16px 18px;
        border: 1px solid #CBD8E8;
        border-left: 4px solid #2563EB;
        border-radius: 14px;
        background: rgba(239, 246, 255, 0.72);
        color: #526077;
        font-size: 0.86rem;
        line-height: 1.65;
    }

    .method-note strong { color: #17365D; }

    @media (max-width: 900px) {
        .hero-grid { grid-template-columns: 1fr; gap: 26px; }
        .terminal-card { max-width: 620px; }
        .about-grid { grid-template-columns: 1fr; }
    }

    @media (max-width: 760px) {
        section.main > div.block-container, .block-container {
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }
        .hero {
            padding: 25px 21px 20px;
            border-radius: 22px;
        }
        .hero-title { font-size: 3rem; }
        .hero-foot { flex-direction: column; gap: 6px; }
        .terminal-meta { flex-direction: column; gap: 5px; }
        .stTabs [data-baseweb="tab"] { padding: 0 11px !important; }
        .stTabs [data-baseweb="tab"] p { font-size: 0.72rem !important; }
    }

    /* ---------------- V2: MINIMAL PROFESSIONAL ---------------- */
    :root {
        --canvas: #F6F7F9;
        --surface: #FFFFFF;
        --ink: #111827;
        --ink-2: #1F2937;
        --blue: #1D4ED8;
        --cyan: #0F766E;
        --mint: #059669;
        --red: #BE123C;
        --muted: #667085;
        --line: #E3E7ED;
        --shadow: none;
    }

    .stApp {
        background: var(--canvas);
        background-image: none;
        color: var(--ink-2);
    }

    [data-testid="stHeader"] {
        background: rgba(246, 247, 249, 0.94);
        border-bottom: 1px solid var(--line);
        backdrop-filter: blur(12px);
    }

    section.main > div.block-container, .block-container {
        max-width: 1360px !important;
        padding-top: 1.25rem !important;
        padding-bottom: 4rem !important;
    }

    .hero {
        padding: 30px 34px 22px;
        margin: 6px 0 22px;
        border: 1px solid #DDE2E8;
        border-radius: 16px;
        background: #FFFFFF;
        box-shadow: none;
    }

    .hero:before { display: none; }

    .hero-grid {
        display: block;
    }

    .brand-lockup {
        margin-bottom: 32px;
        gap: 10px;
    }

    .brand-mark {
        width: 34px;
        height: 34px;
        border: 1px solid #111827;
        border-radius: 7px;
        background: #111827;
        color: #FFFFFF;
        box-shadow: none;
    }

    .brand-name {
        color: #111827;
        font-size: 0.78rem;
        letter-spacing: 0.11em;
    }

    .brand-sub {
        color: #98A2B3;
        font-size: 0.58rem;
        letter-spacing: 0.12em;
    }

    .hero-kicker {
        color: #475467;
        margin-bottom: 11px;
        font-size: 0.68rem;
        letter-spacing: 0.11em;
    }

    .hero-kicker:before {
        width: 6px;
        height: 6px;
        background: #059669;
        box-shadow: none;
    }

    .hero .hero-title {
        color: #111827 !important;
        font-size: 3.55rem;
        line-height: 0.98;
        letter-spacing: -0.055em;
    }

    .hero .hero-title span { color: #1D4ED8 !important; }

    .hero-lead {
        color: #344054 !important;
        max-width: 760px;
        margin-top: 15px;
        font-size: 1.25rem;
        font-weight: 650;
        line-height: 1.35;
    }

    .hero-lead span { color: #1D4ED8; }

    .hero-copy {
        color: #667085 !important;
        max-width: 800px;
        margin-top: 13px;
        font-size: 0.92rem;
        line-height: 1.65;
    }

    .hero-badges {
        margin-top: 19px;
        gap: 7px;
    }

    .hero-badge {
        padding: 6px 10px;
        border: 1px solid #E3E7ED;
        border-radius: 7px;
        background: #F9FAFB;
        color: #475467;
        font-size: 0.66rem;
        letter-spacing: 0.035em;
    }

    .terminal-card { display: none; }

    .hero-foot {
        margin-top: 28px;
        padding-top: 15px;
        border-top: 1px solid #EAECF0;
        color: #98A2B3;
        font-size: 0.64rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2px !important;
        padding: 0 !important;
        margin-bottom: 22px;
        border: 0;
        border-bottom: 1px solid #DDE2E8;
        border-radius: 0;
        background: transparent !important;
        box-shadow: none;
        backdrop-filter: none;
    }

    .stTabs [data-baseweb="tab"] {
        min-height: 44px !important;
        padding: 0 17px !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        background: transparent !important;
        color: #667085 !important;
        font-weight: 700 !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: transparent !important;
        color: #111827 !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom-color: #1D4ED8 !important;
        background: transparent !important;
        color: #111827 !important;
        box-shadow: none;
    }

    .section-head {
        padding: 11px 0 18px;
        margin: 0 0 20px;
        border: 0;
        border-bottom: 1px solid #E3E7ED;
        border-radius: 0;
        background: transparent;
        box-shadow: none;
    }

    .section-head:before { display: none; }

    .section-head-kicker {
        color: #667085;
        font-size: 0.65rem;
        letter-spacing: 0.11em;
    }

    .section-head-title {
        color: #111827;
        font-size: 2rem;
        letter-spacing: -0.038em;
    }

    .section-head-copy {
        max-width: 780px;
        color: #667085;
        font-size: 0.89rem;
    }

    .freshness-bar {
        margin: 0 0 15px;
        padding: 10px 13px;
        border: 1px solid #E3E7ED;
        border-radius: 9px;
        background: #FFFFFF;
        color: #667085;
        box-shadow: none;
    }

    .freshness-status { color: #047857; }
    .freshness-status:before { box-shadow: none; }

    div[data-testid="stMetric"] {
        min-height: 122px;
        padding: 18px 19px;
        border: 1px solid #E3E7ED;
        border-radius: 11px;
        background: #FFFFFF;
        box-shadow: none;
    }

    div[data-testid="stMetric"]:before { display: none; }
    div[data-testid="stMetric"]:hover { transform: none; border-color: #D0D5DD; box-shadow: none; }

    div[data-testid="stMetric"] label {
        color: #667085 !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.07em;
    }

    div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 1.72rem !important;
    }

    .insight-card {
        min-height: 174px;
        padding: 19px 20px;
        border: 1px solid #E3E7ED;
        border-radius: 11px;
        background: #FFFFFF;
        box-shadow: none;
    }

    .insight-card:before { display: none; }
    .insight-card:hover { transform: none; border-color: #D0D5DD; box-shadow: none; }
    .insight-kicker { color: #667085; font-size: 0.68rem; }
    .insight-value { color: #111827; font-size: 1.62rem; }
    .insight-company { color: #1D4ED8; }
    .insight-card--down .insight-value { color: #047857; }
    .insight-card--up .insight-value { color: #BE123C; }

    [data-testid="stExpander"],
    [data-testid="stDataFrame"],
    [data-testid="stPlotlyChart"] {
        border-color: #E3E7ED;
        border-radius: 11px;
        background: #FFFFFF;
        box-shadow: none;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    [data-testid="stTextInput"] input {
        border-color: #D0D5DD !important;
        border-radius: 9px !important;
        box-shadow: none !important;
    }

    .stButton > button,
    .st-key-refresh_action .stButton > button {
        border-color: #111827 !important;
        border-radius: 9px !important;
        background: #111827 !important;
        box-shadow: none;
    }

    .stButton > button:hover,
    .st-key-refresh_action .stButton > button:hover {
        border-color: #1D4ED8 !important;
        background: #1D4ED8 !important;
        box-shadow: none;
        transform: none;
    }

    .stDownloadButton > button {
        border-color: #D0D5DD !important;
        border-radius: 9px !important;
        background: #FFFFFF !important;
        color: #344054 !important;
        box-shadow: none;
    }

    .stDownloadButton > button:hover {
        border-color: #98A2B3 !important;
        background: #F9FAFB !important;
        color: #111827 !important;
        box-shadow: none;
        transform: none;
    }

    div[data-testid="stAlert"] {
        border-color: #E3E7ED;
        border-radius: 10px;
        box-shadow: none;
    }

    .about-card {
        border-color: #E3E7ED;
        border-radius: 11px;
        background: #FFFFFF;
        box-shadow: none;
    }

    .about-card-number { color: #667085; }

    .method-note {
        border-color: #DDE2E8;
        border-left-color: #1D4ED8;
        border-radius: 9px;
        background: #F9FAFB;
    }

    /* Euronext-inspirert markedstavle: stramme flater, tydelige kurser. */
    .market-board-label { margin: 28px 0 11px; color: #6F849B; font-size: 0.74rem; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase; }
    .market-card { min-height: 334px; border: 1px solid #DDE4EA; border-top: 3px solid #2FB3D8; background: #FFFFFF; }
    .market-card-head { padding: 18px 20px 15px; border-bottom: 1px solid #E5E9EE; }
    .market-card-title { color: #71879F; font-size: 1.02rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
    .market-card-subtitle { margin-top: 4px; color: #8B98A7; font-size: 0.71rem; }
    .market-list { padding: 8px 20px 12px; }
    .market-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 14px; min-height: 45px; border-bottom: 1px solid #E9EDF1; }
    .market-row:last-child { border-bottom: 0; }
    .market-name { overflow: hidden; color: #0877E8; font-size: 0.82rem; font-weight: 650; text-overflow: ellipsis; white-space: nowrap; }
    .market-meta { display: block; margin-top: 2px; color: #8A97A6; font-size: 0.65rem; font-weight: 500; }
    .market-value { color: #243447; font-size: 0.83rem; font-variant-numeric: tabular-nums; font-weight: 750; white-space: nowrap; }
    .market-value--up { color: #E33B57; }
    .market-value--down { color: #149447; }
    .market-value--new { color: #0877E8; }
    .market-status { margin-top: 12px; padding: 15px 18px; border: 1px solid #DDE4EA; background: #FFFFFF; }
    .market-status-bar { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 13px 15px; background: #07953A; color: #FFFFFF; font-size: 0.83rem; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; }
    .market-status-dot { width: 9px; height: 9px; border-radius: 50%; background: #A9F2BF; box-shadow: 0 0 0 4px rgba(255,255,255,0.18); }
    .st-key-market_chart_card { min-height: 334px; padding: 0 12px 8px; border: 1px solid #DDE4EA; border-top: 3px solid #2FB3D8; background: #FFFFFF; }
    .st-key-market_chart_card [data-testid="stPlotlyChart"] { border: 0; border-radius: 0; }

    /* ---------------- V3: EURONEXT REGISTER SYSTEM ---------------- */
    :root {
        --canvas: #EFF2F5;
        --surface: #FFFFFF;
        --ink: #172232;
        --ink-2: #364152;
        --blue: #0877E8;
        --cyan: #3CB4DC;
        --teal: #007E73;
        --mint: #149447;
        --red: #E33B57;
        --muted: #75869A;
        --line: #DDE4EA;
    }

    .stApp { background: #EFF2F5; color: var(--ink-2); }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,0.96);
        border-bottom: 3px solid #00776D;
        backdrop-filter: blur(12px);
    }

    section.main > div.block-container,
    .block-container {
        max-width: 1460px !important;
        padding-left: 1.6rem !important;
        padding-right: 1.6rem !important;
    }

    .hero {
        padding: 0;
        overflow: hidden;
        border: 1px solid #D9E1E8;
        border-top: 5px solid #00776D;
        border-radius: 2px;
        background: #FFFFFF;
    }

    .hero-grid { padding: 29px 34px 25px; }
    .brand-lockup { margin-bottom: 23px; }
    .brand-mark { border: 0; border-radius: 2px; background: linear-gradient(135deg, #00776D, #38B5D8); }
    .brand-name { color: #16324A; font-size: 0.82rem; }
    .hero-kicker { color: #00776D; }
    .hero-kicker:before { background: #22A447; }
    .hero .hero-title { color: #18344C !important; font-size: clamp(2.55rem, 5vw, 4.45rem); }
    .hero .hero-title span { color: #941919 !important; }
    .hero-lead { color: #3C4B5C !important; }
    .hero-copy { color: #758292 !important; }
    .hero-badge { border-radius: 2px; border-color: #DCE4EA; background: #F4F7F9; color: #526274; }
    .hero-foot { margin: 0; padding: 12px 34px; border-top: 1px solid #E2E7EB; background: #F7F9FA; color: #7C8997; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        margin: 0 0 24px;
        padding: 0 12px !important;
        border: 0;
        border-radius: 0;
        background: #3CB4DC !important;
    }

    .stTabs [data-baseweb="tab"] {
        min-height: 53px !important;
        padding: 0 22px !important;
        border: 0 !important;
        border-bottom: 4px solid transparent !important;
        color: #10283B !important;
        font-size: 0.76rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.035em;
    }

    .stTabs [data-baseweb="tab"]:hover { background: rgba(255,255,255,0.14) !important; color: #071A29 !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { border-bottom-color: #0B263B !important; background: rgba(255,255,255,0.16) !important; color: #071A29 !important; }

    .section-head {
        margin: 0 0 18px;
        padding: 21px 25px 20px;
        border: 1px solid #DCE3E9;
        border-top: 3px solid #71879F;
        background: #FFFFFF;
    }

    .section-head-kicker { color: #008376; }
    .section-head-title { margin-top: 5px; color: #243A4E; font-size: 2rem; }
    .section-head-copy { color: #748293; }

    .freshness-bar {
        border: 1px solid #DCE4EA;
        border-radius: 0;
        background: #FFFFFF;
    }

    div[data-testid="stMetric"] {
        min-height: 126px;
        border: 1px solid #DCE4EA;
        border-top: 3px solid #3CB4DC;
        border-radius: 0;
        background: #FFFFFF;
    }

    div[data-testid="stMetric"] label { color: #71859B !important; }
    div[data-testid="stMetricValue"] { color: #233A50 !important; }
    div[data-testid="stMetricDelta"] { color: #0877E8 !important; }

    .table-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        margin-top: 18px;
        padding: 16px 18px 14px;
        border: 1px solid #DCE4EA;
        border-bottom: 0;
        border-top: 3px solid #71879F;
        background: #FFFFFF;
    }

    .table-head-title { color: #6F849B; font-size: 0.91rem; font-weight: 800; letter-spacing: 0.115em; text-transform: uppercase; }
    .table-head-copy { margin-top: 4px; color: #8693A1; font-size: 0.71rem; line-height: 1.45; }
    .table-head-badge { flex: 0 0 auto; padding: 5px 8px; border: 1px solid #B8DFEB; background: #EDF9FC; color: #00776D; font-size: 0.63rem; font-weight: 850; letter-spacing: 0.08em; }

    .quick-summary {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1px;
        margin: 5px 0 14px;
        border: 1px solid #DCE4EA;
        background: #DCE4EA;
    }

    .quick-summary > div {
        display: grid;
        grid-template-columns: auto 1fr;
        grid-template-rows: auto auto;
        column-gap: 14px;
        align-items: center;
        padding: 14px 16px;
        background: #F8FAFB;
    }

    .quick-summary span { grid-column: 1 / -1; color: #71859B; font-size: 0.61rem; font-weight: 850; letter-spacing: 0.11em; }
    .quick-summary strong { color: #263D51; font-size: 1.55rem; line-height: 1.1; }
    .quick-summary small { color: #8794A2; font-size: 0.68rem; }

    .st-key-live_filter_panel,
    .st-key-db_filter_panel {
        margin-bottom: 16px;
        padding: 17px 18px 7px;
        border: 1px solid #DCE4EA;
        border-top: 0;
        background: #FFFFFF;
    }

    .st-key-live_filter_panel [data-testid="stHorizontalBlock"],
    .st-key-db_filter_panel [data-testid="stHorizontalBlock"] { align-items: end; }

    .register-status-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        margin: 8px 0 4px;
        padding: 19px 22px;
        border: 1px solid #DCE4EA;
        border-left: 5px solid #149447;
        background: #FFFFFF;
    }

    .register-status-card > div:first-child { display: grid; gap: 3px; }
    .register-status-card strong { color: #263D51; font-size: 0.98rem; }
    .register-status-card small { color: #81909F; }
    .register-status-eyebrow { color: #149447; font-size: 0.62rem; font-weight: 850; letter-spacing: 0.12em; }
    .register-status-count { color: #243A50; font-size: 1.65rem; font-weight: 800; line-height: 1; text-align: right; }
    .register-status-count span { display: block; margin-top: 6px; color: #8794A2; font-size: 0.61rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }

    [data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid #DCE4EA !important;
        border-radius: 0 !important;
        background: #FFFFFF !important;
    }

    [data-testid="stDataFrame"] > div { border-radius: 0 !important; }
    [data-testid="stDataFrame"] button { color: #0877E8 !important; }

    [data-testid="stPlotlyChart"] {
        overflow: hidden;
        border: 1px solid #DCE4EA;
        border-top: 3px solid #71879F;
        border-radius: 0;
        background: #FFFFFF;
    }

    [data-testid="stExpander"] {
        border: 1px solid #DCE4EA;
        border-radius: 0;
        background: #FFFFFF;
    }

    [data-testid="stExpander"] summary { color: #334A60; font-weight: 750; }
    [data-testid="stTextInput"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stMultiSelect"] label,
    [data-testid="stToggle"] label { color: #516173 !important; font-size: 0.76rem !important; font-weight: 700 !important; }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    [data-testid="stTextInput"] input,
    [data-baseweb="tag"] {
        border-radius: 2px !important;
    }

    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within { border-color: #3CB4DC !important; box-shadow: 0 0 0 2px rgba(60,180,220,0.13) !important; }

    .stButton > button,
    .st-key-refresh_action .stButton > button {
        border: 1px solid #00776D !important;
        border-radius: 2px !important;
        background: #00776D !important;
        color: #FFFFFF !important;
    }

    .stButton > button:hover,
    .st-key-refresh_action .stButton > button:hover { border-color: #00645C !important; background: #00645C !important; }

    .stDownloadButton > button {
        border-color: #9CCFDC !important;
        border-radius: 2px !important;
        background: #FFFFFF !important;
        color: #00776D !important;
    }

    .stDownloadButton > button:hover { border-color: #3CB4DC !important; background: #EDF9FC !important; color: #00645C !important; }

    div[data-testid="stAlert"] { border-radius: 0; }
    hr { border-color: #DCE3E9 !important; }
    h2, h3 { color: #2B4156 !important; }

    .market-board-label { padding: 12px 15px; margin-bottom: 0; border: 1px solid #DCE4EA; border-bottom: 0; background: #FFFFFF; }
    .market-card, .market-status, .st-key-market_chart_card { border-radius: 0; }
    .market-card, .st-key-market_chart_card { border-top-color: #3CB4DC; }
    .about-card { border-radius: 0; border-top: 3px solid #71879F; }
    .method-note { border-radius: 0; border-left-color: #00776D; }

    @media (max-width: 760px) {
        section.main > div.block-container,
        .block-container {
            width: 100% !important;
            max-width: 100% !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-bottom: 2.5rem !important;
        }

        /* Streamlit-kolonner blir én tydelig vertikal mobilflyt. */
        [data-testid="stHorizontalBlock"] {
            width: 100% !important;
            flex-direction: column !important;
            flex-wrap: nowrap !important;
            gap: 0.75rem !important;
        }

        [data-testid="column"],
        [data-testid="stColumn"] {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
            flex: 1 1 100% !important;
        }

        .hero-grid,
        .about-grid {
            grid-template-columns: minmax(0, 1fr) !important;
            gap: 1rem !important;
        }

        .hero,
        .terminal-card,
        .section-head,
        .insight-card,
        .about-card,
        .method-note {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
        }

        .hero {
            padding: 23px 18px 18px;
            border-radius: 12px;
        }

        .brand-lockup { margin-bottom: 24px; }

        .hero .hero-title {
            font-size: clamp(2.2rem, 11vw, 3rem) !important;
            overflow-wrap: anywhere;
        }

        .hero-lead { font-size: 1.08rem; }

        .hero-foot,
        .terminal-meta,
        .freshness-bar {
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 0.45rem !important;
        }

        .insight-card,
        .about-card {
            min-height: 0 !important;
            padding: 18px !important;
        }

        .market-card,
        .st-key-market_chart_card { min-height: 0 !important; }

        .table-head { align-items: flex-start; padding: 14px; }
        .table-head-badge { display: none; }
        .register-status-card { align-items: flex-start; flex-direction: column; }
        .register-status-count { text-align: left; }
        .quick-summary { grid-template-columns: 1fr; }

        .insight-company,
        .insight-detail,
        .section-head-copy,
        .pipeline-step span {
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        [data-testid="stDataFrame"],
        [data-testid="stPlotlyChart"] {
            width: 100% !important;
            max-width: 100% !important;
            min-width: 0 !important;
        }

        [data-testid="stPlotlyChart"] {
            padding: 6px !important;
        }

        /* Fanene kan sveipes vannrett i stedet for å presse siden bredere. */
        .stTabs [data-baseweb="tab-list"] {
            max-width: 100%;
            overflow-x: auto;
            overflow-y: hidden;
            flex-wrap: nowrap;
            scrollbar-width: none;
        }

        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none;
        }

        .stTabs [data-baseweb="tab"] {
            flex: 0 0 auto;
            padding: 0 10px !important;
        }
    }

    /* ---------------- V5: TAILWIND PRO / COMPILED CSS ----------------
       Ingen CDN eller JavaScript: robust i Streamlit 1.50 og på Cloud. */
    :root {
        --tw-slate-950: #020617;
        --tw-slate-900: #0F172A;
        --tw-slate-800: #1E293B;
        --tw-slate-700: #334155;
        --tw-slate-500: #64748B;
        --tw-slate-200: #E2E8F0;
        --tw-slate-100: #F1F5F9;
        --tw-blue-600: #2563EB;
        --tw-cyan-500: #06B6D4;
        --tw-emerald-500: #10B981;
        --tw-rose-500: #F43F5E;
        --tw-ring: rgba(37, 99, 235, 0.18);
        --tw-shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.05);
        --tw-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
        --tw-shadow-lg: 0 24px 70px rgba(2, 6, 23, 0.20);
    }

    html { scroll-behavior: smooth; }

    .stApp {
        background:
            radial-gradient(circle at 7% 1%, rgba(37,99,235,.10), transparent 27rem),
            radial-gradient(circle at 96% 10%, rgba(6,182,212,.09), transparent 28rem),
            #F8FAFC;
        color: var(--tw-slate-800);
    }

    ::selection { background: rgba(6,182,212,.24); color: #082F49; }

    [data-testid="stHeader"] {
        background: rgba(248,250,252,.82);
        border-bottom: 1px solid rgba(148,163,184,.22);
        backdrop-filter: blur(18px) saturate(160%);
    }

    .hero {
        position: relative;
        isolation: isolate;
        overflow: hidden;
        padding: 0;
        border: 1px solid rgba(148,163,184,.34);
        border-top: 4px solid #2BA6A0;
        border-radius: 20px;
        background:
            radial-gradient(circle at 92% -18%, rgba(79,180,191,.16), transparent 29rem),
            linear-gradient(124deg, #0A1A2B 0%, #102B40 60%, #12364A 100%);
        box-shadow: 0 22px 58px rgba(15,23,42,.18);
    }

    .hero:before {
        display: block;
        content: "";
        position: absolute;
        width: 520px;
        height: 520px;
        right: -235px;
        bottom: -345px;
        z-index: -1;
        border: 1px solid rgba(148,210,218,.15);
        border-radius: 50%;
        box-shadow:
            0 0 0 74px rgba(148,210,218,.025),
            0 0 0 148px rgba(148,210,218,.018);
    }

    .hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.42fr) minmax(310px, .68fr);
        gap: 58px;
        align-items: center;
        padding: 43px 44px 36px;
    }

    .brand-lockup { margin-bottom: 35px; }
    .brand-mark {
        width: 38px;
        height: 38px;
        border: 1px solid rgba(181,220,225,.38);
        border-radius: 7px;
        background: rgba(255,255,255,.075);
        color: #DDF4F5;
        box-shadow: inset 0 1px rgba(255,255,255,.10);
    }
    .brand-name { color: #DCE8F0; font-size: .72rem; letter-spacing: .17em; }
    .hero-kicker {
        color: #88C9CE;
        margin-bottom: 15px;
        font-size: .68rem;
        letter-spacing: .14em;
    }
    .hero-kicker:before {
        width: 24px;
        height: 1px;
        border-radius: 0;
        background: #4CB6B4;
        box-shadow: none;
        animation: none;
    }
    .hero .hero-title {
        overflow: visible;
        color: #F8FAFC !important;
        font-size: clamp(3rem, 5.35vw, 4.8rem);
        line-height: .96;
        font-weight: 820;
        letter-spacing: -.052em;
    }
    .hero .hero-title-main {
        display: block;
        color: #F8FAFC !important;
    }
    .hero .hero-title-accent {
        display: block;
        width: max-content;
        max-width: 100%;
        padding: .06em .15em .06em 0;
        color: #FFFFFF !important;
        font-weight: 760;
        letter-spacing: -.045em;
    }
    .hero-lead {
        max-width: 720px;
        color: #D7E3EB !important;
        font-size: clamp(1.08rem, 1.65vw, 1.32rem);
        font-weight: 650;
        letter-spacing: -.015em;
        margin-top: 22px;
    }
    .hero-copy {
        max-width: 650px;
        color: #91A7B8 !important;
        font-size: .94rem;
        line-height: 1.68;
        margin-top: 10px;
    }
    .hero-badges {
        gap: 18px;
        margin-top: 25px;
    }
    .hero-badge {
        padding: 1px 0 1px 11px;
        border: 0;
        border-left: 1px solid rgba(115,198,204,.45);
        border-radius: 0;
        background: transparent;
        color: #AFC1CD;
        font-size: .66rem;
        letter-spacing: .06em;
        backdrop-filter: none;
    }
    .hero-badge:hover { border-color: #73C6CC; background: transparent; color: #E1EEF2; }
    .hero-foot {
        margin: 0;
        padding: 13px 44px;
        border-top: 1px solid rgba(148,163,184,.13);
        background: rgba(3,13,24,.25);
        color: #71899A;
        font-size: .64rem;
        letter-spacing: .075em;
    }

    .tw-terminal {
        overflow: hidden;
        border: 1px solid rgba(226,232,240,.92);
        border-radius: 14px;
        background: rgba(255,255,255,.965);
        box-shadow: 0 18px 42px rgba(2,12,27,.21);
        backdrop-filter: blur(12px);
    }
    .tw-terminal-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 15px 17px 13px;
        border-bottom: 1px solid #E7EDF2;
        color: #16364D;
        font-size: .65rem;
        font-weight: 850;
        letter-spacing: .13em;
    }
    .tw-live-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: #12805C;
        font-size: .61rem;
        letter-spacing: .08em;
    }
    .tw-live-status i {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #21A67A;
        box-shadow: 0 0 0 3px rgba(33,166,122,.10);
    }
    .tw-terminal-body { display: grid; gap: 0; padding: 6px 17px 15px; }
    .tw-terminal-row { display: flex; align-items: center; justify-content: space-between; gap: 15px; }
    .tw-terminal-row { min-height: 46px; }
    .tw-terminal-row span { color: #728395; font-size: .65rem; font-weight: 700; letter-spacing: .055em; }
    .tw-terminal-row strong { color: #17364C; font-size: .74rem; font-weight: 800; text-align: right; }
    .tw-terminal-line { height: 1px; background: #E9EEF2; }
    .tw-terminal-status {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 8px;
        padding: 11px 12px;
        border: 1px solid #CFE8E2;
        border-radius: 8px;
        background: #F0F8F6;
        color: #55716B;
        font-size: .65rem;
        font-weight: 700;
        letter-spacing: .035em;
    }
    .tw-terminal-status strong { color: #13775B; font-size: .67rem; letter-spacing: .06em; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 7px !important;
        margin: 0 0 25px;
        padding: 7px !important;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        background: rgba(255,255,255,.88) !important;
        box-shadow: var(--tw-shadow-sm);
        backdrop-filter: blur(14px);
    }
    .stTabs [data-baseweb="tab"] {
        min-height: 43px !important;
        padding: 0 18px !important;
        border: 0 !important;
        border-radius: 10px !important;
        color: #64748B !important;
        transition: color .18s ease, background .18s ease, transform .18s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { background: #F1F5F9 !important; color: #0F172A !important; transform: translateY(-1px); }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border: 0 !important;
        background: linear-gradient(135deg, #2563EB, #0891B2) !important;
        color: #FFFFFF !important;
        box-shadow: 0 8px 22px rgba(37,99,235,.22);
    }

    .section-head {
        position: relative;
        overflow: hidden;
        margin-bottom: 20px;
        padding: 23px 25px 22px;
        border: 1px solid #E2E8F0;
        border-top: 1px solid #E2E8F0;
        border-radius: 16px;
        background: rgba(255,255,255,.94);
        box-shadow: var(--tw-shadow-sm);
    }
    .section-head:before {
        display: block;
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 4px;
        background: linear-gradient(180deg, #2563EB, #06B6D4);
    }
    .section-head-kicker { color: #0284C7; }
    .section-head-title { color: #0F172A; }

    .freshness-bar {
        border: 1px solid #E2E8F0;
        border-radius: 13px;
        background: rgba(255,255,255,.92);
        box-shadow: var(--tw-shadow-sm);
    }

    div[data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        border-top: 1px solid #E2E8F0;
        border-radius: 16px;
        background: linear-gradient(145deg, #FFFFFF, #F8FAFC);
        box-shadow: var(--tw-shadow-sm);
        transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
    }
    div[data-testid="stMetric"]:before {
        display: block;
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 3px;
        background: linear-gradient(180deg, #2563EB, #06B6D4);
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: #BFDBFE; box-shadow: var(--tw-shadow); }

    .market-board-label {
        padding: 13px 17px;
        border: 1px solid #E2E8F0;
        border-bottom: 0;
        border-radius: 14px 14px 0 0;
        background: rgba(255,255,255,.92);
    }
    .market-card,
    .st-key-market_chart_card {
        overflow: hidden;
        border: 1px solid #E2E8F0;
        border-top: 3px solid #22D3EE;
        border-radius: 16px;
        background: #FFFFFF;
        box-shadow: var(--tw-shadow-sm);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .market-card:hover,
    .st-key-market_chart_card:hover { transform: translateY(-2px); box-shadow: var(--tw-shadow); }
    .market-card-head { background: linear-gradient(135deg, #F8FAFC, #FFFFFF); }
    .market-name { color: #2563EB; }
    .market-status { border: 1px solid #D1FAE5; border-radius: 14px; box-shadow: var(--tw-shadow-sm); }
    .market-status-bar { border-radius: 9px; background: linear-gradient(135deg, #059669, #10B981); }

    .table-head {
        margin-top: 19px;
        padding: 17px 19px 15px;
        border: 1px solid #E2E8F0;
        border-bottom: 0;
        border-top: 1px solid #E2E8F0;
        border-radius: 14px 14px 0 0;
        background: linear-gradient(135deg, #F8FAFC, #FFFFFF);
    }
    .table-head-title { color: #334155; }
    .table-head-badge { border-color: #BAE6FD; border-radius: 999px; background: #F0F9FF; color: #0369A1; }
    [data-testid="stDataFrame"] {
        border-color: #E2E8F0 !important;
        border-radius: 0 0 14px 14px !important;
        box-shadow: var(--tw-shadow-sm);
    }
    [data-testid="stDataFrame"] > div { border-radius: 0 0 14px 14px !important; }

    .quick-summary {
        gap: 10px;
        border: 0;
        background: transparent;
    }
    .quick-summary > div {
        border: 1px solid #E2E8F0;
        border-radius: 13px;
        background: linear-gradient(145deg, #FFFFFF, #F8FAFC);
        box-shadow: var(--tw-shadow-sm);
    }

    .st-key-live_filter_panel,
    .st-key-db_filter_panel {
        border-color: #E2E8F0;
        border-radius: 0 0 14px 14px;
        background: #FFFFFF;
        box-shadow: var(--tw-shadow-sm);
    }

    [data-testid="stPlotlyChart"] {
        border: 1px solid #E2E8F0;
        border-top: 1px solid #E2E8F0;
        border-radius: 16px;
        box-shadow: var(--tw-shadow-sm);
    }
    [data-testid="stExpander"] {
        overflow: hidden;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        box-shadow: var(--tw-shadow-sm);
    }
    [data-testid="stExpander"] summary { background: linear-gradient(135deg, #FFFFFF, #F8FAFC); }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    [data-testid="stTextInput"] input,
    [data-baseweb="tag"] { border-radius: 10px !important; }
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within { border-color: #60A5FA !important; box-shadow: 0 0 0 4px var(--tw-ring) !important; }

    .stButton > button,
    .st-key-refresh_action .stButton > button {
        border: 0 !important;
        border-radius: 10px !important;
        background: linear-gradient(135deg, #2563EB, #0891B2) !important;
        box-shadow: 0 9px 24px rgba(37,99,235,.20);
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
    }
    .stButton > button:hover,
    .st-key-refresh_action .stButton > button:hover { transform: translateY(-1px); filter: brightness(1.05); box-shadow: 0 13px 30px rgba(37,99,235,.27); }
    .stDownloadButton > button {
        border-color: #CBD5E1 !important;
        border-radius: 10px !important;
        background: #FFFFFF !important;
        color: #334155 !important;
    }
    .stDownloadButton > button:hover { border-color: #93C5FD !important; background: #EFF6FF !important; color: #1D4ED8 !important; transform: translateY(-1px); }

    .register-status-card {
        border: 1px solid #D1FAE5;
        border-left: 4px solid #10B981;
        border-radius: 14px;
        background: linear-gradient(135deg, #FFFFFF, #F0FDF4);
        box-shadow: var(--tw-shadow-sm);
    }
    .about-card {
        border: 1px solid #E2E8F0;
        border-top: 1px solid #E2E8F0;
        border-radius: 16px;
        box-shadow: var(--tw-shadow-sm);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .about-card:hover { transform: translateY(-3px); box-shadow: var(--tw-shadow); }
    .method-note { border-radius: 14px; border-left-color: #2563EB; background: #EFF6FF; }
    div[data-testid="stAlert"] { border-radius: 12px; box-shadow: var(--tw-shadow-sm); }

    @keyframes tw-pulse {
        0%, 100% { opacity: .72; transform: scale(.95); }
        50% { opacity: 1; transform: scale(1.08); }
    }

    @media (max-width: 900px) {
        .hero-grid { grid-template-columns: 1fr; gap: 24px; }
        .tw-terminal { max-width: 640px; }
    }

    @media (max-width: 760px) {
        .hero { padding: 0; border-radius: 15px; }
        .hero-grid { padding: 28px 20px 23px; }
        .brand-lockup { margin-bottom: 27px; }
        .hero .hero-title { font-size: clamp(2.62rem, 13vw, 3.45rem); line-height: .98; }
        .hero .hero-title-accent { padding-right: .18em; }
        .hero-badges { display: grid; gap: 9px; }
        .hero-badge { padding-left: 9px; }
        .hero-foot { padding: 12px 19px; }
        .hero-foot span:last-child { display: none; }
        .tw-terminal { border-radius: 14px; }
        .stTabs [data-baseweb="tab-list"] { border-radius: 13px; padding: 5px !important; }
        .stTabs [data-baseweb="tab"] { min-height: 39px !important; padding: 0 11px !important; }
        .section-head { border-radius: 14px; padding: 20px 18px 18px; }
        .market-card, .st-key-market_chart_card, [data-testid="stPlotlyChart"] { border-radius: 14px; }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; scroll-behavior: auto !important; }
    }
    </style>

    <div class="hero">
        <div class="hero-grid">
            <div>
                <div class="brand-lockup">
                    <div class="brand-mark">SR</div>
                    <div>
                        <span class="brand-name">VELKOMMEN!</span>
                    </div>
                </div>
                <div class="hero-kicker">Offentlig SSR-data · analyse og historikk</div>
                <h1 class="hero-title">
                    <span class="hero-title-main">Shortregister</span>
                    <span class="hero-title-accent">for Oslo Børs</span>
                </h1>
                <p class="hero-lead">Denne plattformen gir et samlet markedsbilde av offentlig rapporterte shortposisjoner.</p>
                <p class="hero-copy">
                    Her kan du søke etter selskaper og posisjonsholdere, sammenligne nivåer og følge utviklingen over tid.
                </p>
                <div class="hero-badges">
                    <span class="hero-badge">FINANSTILSYNETS SSR V2-API</span>
                    <span class="hero-badge">OFFENTLIG TERSKEL FOR RAPPORTERING ER: ≥ 0,50 %</span>
                    <span class="hero-badge">PLATTFORMEN BESTÅR AV INTERAKTIVE TABELLER OG GRAFER</span>
                </div>
            </div>
            <div class="tw-terminal" aria-label="Registerstatus">
                <div class="tw-terminal-top">
                    <span>REGISTERSTATUS</span>
                    <span class="tw-live-status"><i></i> LIVE</span>
                </div>
                <div class="tw-terminal-body">
                    <div class="tw-terminal-row"><span>DATAKILDE</span><strong>FINANSTILSYNET SSR V2</strong></div>
                    <div class="tw-terminal-line"></div>
                    <div class="tw-terminal-row"><span>DEKNING</span><strong>LIVE + HISTORIKK</strong></div>
                    <div class="tw-terminal-line"></div>
                    <div class="tw-terminal-row"><span>TERSKEL</span><strong>≥ 0,50 %</strong></div>
                    <div class="tw-terminal-status"><span>OFFENTLIG REGISTER</span><strong>OPPDATERT</strong></div>
                </div>
            </div>
        </div>
        <div class="hero-foot">
            <span>Utviklet av Andreas Bolton Seielstad</span>
            <span>Uavhengig analyseverktøy</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Registeret ligger i en delt ressurs-cache. Ingen kopier lagres i brukernes session_state.
with st.spinner("Laster delt datagrunnlag …"):
    df_live = hent_fullt_register()
    df_holders = hent_posisjonsholdere()
    df_exempt = hent_unntatte_instrumenter()

# SQLite-data leses også fra en delt cache og blir ikke lagret per bruker.
df_db = hent_database_data()

tab_live, tab_db, tab_top10, tab_about = st.tabs(
    ["●  LIVE", "⌕  SELSKAPSSØK", "▥  TOPP 10", "ⓘ  OM"]
)

with tab_live:
    title_col, refresh_col = st.columns([4, 1.15], vertical_alignment="center")

    with title_col:
        _render_section_header(
            "Marked · offentlig SSR-data",
            "Markedsoversikt",
            "Her er siste offentlig rapporterte posisjoner, endringer og nye signaler fra Finanstilsynets register.",
        )

    with refresh_col:
        with st.container(key="refresh_action"):
            if st.button(
                "↻  Oppdater data",
                key="force_refresh",
                width="stretch",
                help="Tømmer den delte én-timescachen og henter registeret på nytt.",
            ):
                st.session_state["show_refresh_success"] = True
                tving_ny_nedlasting()
                st.rerun()


    if st.session_state.pop("show_refresh_success", False):
        st.success("Registeret er oppdatert med de nyeste dataene fra Finanstilsynet.")

    if df_live.empty:
        st.error("Klarte ikke hente data fra Finanstilsynet akkurat nå.")
    else:
        latest_date = pd.to_datetime(df_live["date"], errors="coerce").max()

        # Bruk siste registrerte, aggregerte shortandel per selskap.
        # Dette samsvarer med "SUM SHORT %" i Finanstilsynets oversikt.
        live_data = _standardiser_shortpercent(df_live)
        current_positions = hent_siste_posisjon_per_selskap(live_data)
        total_short = (
            current_positions["shortPercent"].sum()
            if not current_positions.empty
            else 0.0
        )

        if current_positions.empty:
            max_short = 0.0
            max_short_company = "Ukjent selskap"
            max_short_holder = "Aggregert shortandel"
            max_short_date = "Ukjent dato"
        else:
            max_short_row = current_positions.iloc[0]
            max_short = float(max_short_row["shortPercent"])
            max_short_company = str(max_short_row.get("issuerName") or "Ukjent selskap")
            max_short_holder = "Aggregert shortandel"
            parsed_max_date = pd.to_datetime(max_short_row.get("date"), errors="coerce")
            max_short_date = (
                parsed_max_date.strftime("%d.%m.%Y")
                if pd.notna(parsed_max_date)
                else "Ukjent dato"
            )

        latest_date_text = (
            latest_date.strftime("%d.%m.%Y")
            if pd.notna(latest_date)
            else "ukjent dato"
        )
        st.markdown(
            f"""
            <div class="freshness-bar">
                <span class="freshness-status">Datagrunnlag klart</span>
                <span>Siste observasjon: <strong>{latest_date_text}</strong></span>
                <span>Selskaper: <strong>{len(current_positions):,}</strong></span>
                <span>Offentlig terskel er: <strong>≥ 0,50 %</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Live-posisjoner", f"{len(df_live):,}")
        col2.metric("Unike selskaper", f"{df_live['issuerName'].nunique():,}")
        col3.metric("Sum av rapporterte shortposisjoner", f"{total_short:,.2f} %")
        col4.metric(
            "Største gjeldende shortandel",
            f"{max_short:,.2f} %",
            delta=max_short_company,
            delta_color="off",
        )

        with st.expander("Hvor er Frontline og andre selskaper som man ikke finner i listen?", expanded=False):
            st.markdown(
                "**Frontline mangler ikke på grunn av en feil i appen.** Finanstilsynet "
                "har unntatt enkelte aksjer fra SSR-rapportering. I tillegg viser API-et "
                "bare offentlig rapporterbare nettoposisjoner på minst 0,5 %, ikke en "
                "komplett liste over alle selskaper på Oslo Børs."
            )
            if df_exempt is not None and not df_exempt.empty:
                exempt_view = df_exempt.rename(
                    columns={
                        "issuerName": "Selskap",
                        "isin": "ISIN",
                        "status": "Status",
                        "effectiveFrom": "Unntatt fra",
                    }
                ).copy()
                exempt_view["Unntatt fra"] = pd.to_datetime(
                    exempt_view["Unntatt fra"], errors="coerce"
                ).dt.strftime("%d.%m.%Y")
                _render_table_header(
                    "Instrumenter unntatt SSR-rapportering",
                    "Listen kommer fra Finanstilsynets offentlige unntaksoversikt.",
                    f"{len(exempt_view):,} INSTRUMENTER",
                )
                st.dataframe(
                    exempt_view[["Selskap", "ISIN", "Status", "Unntatt fra"]],
                    width="stretch",
                    hide_index=True,
                )
            st.caption(
                "Manglende selskap eller tall betyr ikke automatisk 0 % short. "
                "Det betyr at Finanstilsynets offentlige SSR-kilde ikke har en "
                "rapporterbar observasjon å vise."
            )

        # Gjør individuelle aktive posisjonsholdere lett tilgjengelige høyt på siden.
        st.divider()
        vis_posisjonsholdere(df_holders, "live_holders")
        st.divider()

        # Tre raske markedssignaler. Vi viser største reduksjon og økning
        # separat for å unngå å gjenta "største gjeldende shortandel" fra KPI-kortene.
        changes = beregn_storste_endringer(live_data)

        if changes.empty:
            decrease_value = "Ingen endring"
            decrease_company = "Ingen tilgjengelige data"
            decrease_detail = "Kan beregnes når minst to observasjoner finnes."

            increase_value = "Ingen endring"
            increase_company = "Ingen tilgjengelige data"
            increase_detail = "Kan beregnes når minst to observasjoner finnes."
        else:
            decreases = changes.loc[changes["endring"] < 0].copy()
            if decreases.empty:
                decrease_value = "Ingen reduksjon"
                decrease_company = "Ingen tilgjengelige data"
                decrease_detail = "Ingen siste reduksjoner funnet i datasettet."
            else:
                decrease_row = decreases.sort_values("endring", ascending=True).iloc[0]
                decrease_value = f"{float(decrease_row['endring']):+.2f} pp"
                decrease_company = str(decrease_row.get("issuerName") or "Ukjent selskap")
                decrease_from = float(decrease_row.get("forrige_short", 0.0))
                decrease_to = float(decrease_row.get("shortPercent", 0.0))
                decrease_date = pd.to_datetime(decrease_row.get("date"), errors="coerce")
                decrease_date_text = (
                    decrease_date.strftime("%d.%m.%Y")
                    if pd.notna(decrease_date)
                    else "ukjent dato"
                )
                decrease_detail = (
                    f"Fra {decrease_from:.2f} % til {decrease_to:.2f} % · "
                    f"{decrease_date_text}"
                )

            increases = changes.loc[changes["endring"] > 0].copy()
            if increases.empty:
                increase_value = "Ingen økning"
                increase_company = "Ingen tilgjengelige data"
                increase_detail = "Ingen siste økninger funnet i datasettet."
            else:
                increase_row = increases.sort_values("endring", ascending=False).iloc[0]
                increase_value = f"{float(increase_row['endring']):+.2f} pp"
                increase_company = str(increase_row.get("issuerName") or "Ukjent selskap")
                increase_from = float(increase_row.get("forrige_short", 0.0))
                increase_to = float(increase_row.get("shortPercent", 0.0))
                increase_date = pd.to_datetime(increase_row.get("date"), errors="coerce")
                increase_date_text = (
                    increase_date.strftime("%d.%m.%Y")
                    if pd.notna(increase_date)
                    else "ukjent dato"
                )
                increase_detail = (
                    f"Fra {increase_from:.2f} % til {increase_to:.2f} % · "
                    f"{increase_date_text}"
                )

        new_positions = finn_nye_shortposisjoner(live_data)
        if new_positions.empty:
            new_value = "Ingen nye"
            new_company = "Ingen nye posisjoner over 0,5 %"
            new_detail = "Basert på siste registrerte nivå per selskap."
        else:
            new_row = new_positions.sort_values(
                ["date", "shortPercent"], ascending=[False, False]
            ).iloc[0]
            new_value = f"{float(new_row['shortPercent']):.2f} %"
            new_company = str(new_row.get("issuerName") or "Ukjent selskap")
            new_holder = str(new_row.get("positionHolder") or "Ikke oppgitt")
            new_date = pd.to_datetime(new_row.get("date"), errors="coerce")
            new_date_text = (
                new_date.strftime("%d.%m.%Y")
                if pd.notna(new_date)
                else "ukjent dato"
            )
            new_detail = f"{new_holder} · Registrert {new_date_text}"

        st.markdown('<div class="market-board-label">Markedspuls</div>', unsafe_allow_html=True)
        board_left, board_middle, board_right = st.columns([1, 1.18, 1])

        top_rows = []
        for _, row in current_positions.head(5).iterrows():
            company = html.escape(str(row.get("issuerName") or "Ukjent selskap"))
            value = float(row.get("shortPercent", 0.0))
            top_rows.append(
                f'<div class="market-row"><div class="market-name" title="{company}">{company}'
                f'<span class="market-meta">Gjeldende aggregert posisjon</span></div>'
                f'<div class="market-value market-value--up">{value:.2f} %</div></div>'
            )

        with board_left:
            st.markdown(
                f"""
                <div class="market-card">
                    <div class="market-card-head"><div class="market-card-title">Mest shortet</div><div class="market-card-subtitle">Siste rapporterte nivå per selskap</div></div>
                    <div class="market-list">{''.join(top_rows)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with board_middle:
            with st.container(key="market_chart_card"):
                st.markdown('<div class="market-card-head"><div class="market-card-title">Shortdiagram</div><div class="market-card-subtitle">Topp 5 · offentlig shortandel</div></div>', unsafe_allow_html=True)
                chart_data = current_positions.head(5).sort_values("shortPercent")
                pulse_fig = px.bar(chart_data, x="shortPercent", y="issuerName", orientation="h", text="shortPercent")
                pulse_fig.update_traces(marker_color="#35B7D8", texttemplate="%{text:.2f}%", textposition="outside", cliponaxis=False, hovertemplate="%{y}<br>%{x:.2f} %<extra></extra>")
                pulse_fig.update_layout(height=250, margin=dict(l=8, r=42, t=8, b=12), paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", showlegend=False, xaxis=dict(visible=False, rangemode="tozero"), yaxis=dict(title=None, tickfont=dict(size=10, color="#354052")), font=dict(family="Inter, sans-serif", color="#354052"))
                st.plotly_chart(pulse_fig, width="stretch", config={"displayModeBar": False})

        movement_rows = (
            f'<div class="market-row"><div class="market-name">{html.escape(increase_company)}<span class="market-meta">Største siste økning</span></div><div class="market-value market-value--up">{html.escape(increase_value)}</div></div>'
            f'<div class="market-row"><div class="market-name">{html.escape(decrease_company)}<span class="market-meta">Største siste reduksjon</span></div><div class="market-value market-value--down">{html.escape(decrease_value)}</div></div>'
            f'<div class="market-row"><div class="market-name">{html.escape(new_company)}<span class="market-meta">Nyeste posisjon over terskel</span></div><div class="market-value market-value--new">{html.escape(new_value)}</div></div>'
        )

        with board_right:
            st.markdown(
                f"""
                <div class="market-card">
                    <div class="market-card-head"><div class="market-card-title">Siste bevegelser</div><div class="market-card-subtitle">Endring i prosentpoeng</div></div>
                    <div class="market-list">{movement_rows}</div>
                </div>
                <div class="market-status"><div class="market-card-subtitle">Sist observert {latest_date_text}</div><div class="market-status-bar"><span>Data online</span><span class="market-status-dot"></span></div></div>
                """,
                unsafe_allow_html=True,
            )

        action_left, action_right = st.columns(2)

        with action_left:
            if st.button(
                "Oppdater historikk",
                key="save_live",
                width="stretch",
                help="Lagrer bare nye rader i SQLite-databasen.",
            ):
                with st.spinner("Sammenligner og lagrer nye rader …"):
                    new_rows = lagre_i_database(df_live)
                st.success(f"Ferdig. {new_rows:,} nye rader ble lagret.")

        with action_right:
            st.download_button(
                "Last ned registeret som CSV",
                data=dataframe_to_csv(df_live),
                file_name="shortregister.csv",
                mime="text/csv",
                width="stretch",
            )

        st.caption(
            "Oppdater-knappen øverst henter helt ferske data fra Finanstilsynet. "
            "Historikk-knappen lagrer bare registreringer som ikke allerede finnes i databasen."
        )

        vis_hurtiginnsikt(df_live, expanded=True)
        vis_sok_og_graf(df_live, "live", df_exempt)

    st.divider()
    latest_time, total_rows = hent_siste_oppdatering()
    if latest_time:
        st.markdown(
            f"""
            <div class="register-status-card">
                <div><span class="register-status-eyebrow">HISTORIKKREGISTER</span><strong>SQLite-data er tilgjengelig</strong><small>Sist oppdatert {html.escape(str(latest_time))}</small></div>
                <div class="register-status-count">{total_rows:,}<span>lagrede rader</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Ingen lagringshistorikk er registrert ennå.")

with tab_db:
    _render_section_header(
        "Historikk · selskapssøk",
        "Finn selskapet og se utviklingen.",
        "Søk på selskapsnavn eller ISIN, filtrer historikken og eksporter akkurat det utsnittet du trenger.",
    )
    if df_db.empty:
        st.info("SQLite-databasen er tom. Lagre live-registeret først.")
    else:
        st.success(f"Databasen inneholder {len(df_db):,} rader.")
        vis_hurtiginnsikt(df_db)
        vis_sok_og_graf(df_db, "db", df_exempt)


with tab_top10:
    _render_section_header(
        "Rangering · markedsbilde",
        "Markedets mest shortede selskaper",
        "Her kan man sammenligne gjennomsnittlig offentlig rapportert shortandel og se hvordan toppsjiktet har beveget seg over tid.",
    )
    top10_sources = [
        frame
        for frame in (df_db, df_live)
        if frame is not None and not frame.empty
    ]
    if not top10_sources:
        st.info("Ingen markedsdata er tilgjengelig akkurat nå.")
    else:
        # Kombiner historikk og live-data. Overlapp fjernes, slik at en ny deploy
        # fortsatt kan vise rangeringen selv om SQLite-filen er tom eller gammel.
        data = pd.concat(top10_sources, ignore_index=True, sort=False).drop_duplicates()
        data = _agg_issuer_date(data)

        period = st.selectbox("Velg tidsperiode", ["30 dager", "90 dager", "180 dager", "365 dager"])
        days = int(period.split()[0])
        latest_available = data["date"].max() if not data.empty else pd.NaT
        daily_series = pd.DataFrame(columns=["issuerName", "date", "shortPercent"])

        if pd.isna(latest_available):
            start_date = pd.NaT
        else:
            latest_available = latest_available.normalize()
            # Inkludert både start- og sluttdato gir dette nøyaktig valgt
            # antall kalenderdager (f.eks. 90 daglige observasjoner).
            start_date = latest_available - pd.Timedelta(days=days - 1)
            daily_series = bygg_daglig_shortserie(
                data,
                start_date=start_date,
                end_date=latest_available,
            )

            today = pd.Timestamp.today().normalize()
            age_days = max(0, int((today - latest_available).days))
            freshness = (
                "Siste observasjon er oppdatert."
                if age_days <= 1
                else f"Siste observasjon er {age_days} dager gammel."
            )
            st.caption(
                f"Analysevindu: {start_date.strftime('%d.%m.%Y')}–"
                f"{latest_available.strftime('%d.%m.%Y')} · {days} kalenderdager · "
                f"{freshness}"
            )

        if daily_series.empty:
            st.info("Fant ingen gyldige daterte observasjoner i datagrunnlaget.")
        else:
            top10 = (
                daily_series.groupby("issuerName", as_index=False)["shortPercent"]
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
                title=f"Topp 10 – tidsvektet shortandel siste {days} dager",
                labels={"issuerName": "Selskap", "shortPercent": "Shortandel (%)"},
                color_discrete_sequence=[CHART_PALETTE[0]],
            )
            _style_plotly_chart(fig_bar, height=500)
            fig_bar.update_xaxes(tickangle=-28)
            fig_bar.update_yaxes(ticksuffix=" %")
            fig_bar.update_traces(
                marker=dict(
                    color="#2563EB",
                    line=dict(color="#C7DBFF", width=1.2),
                ),
                texttemplate="%{y:.2f} %",
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}<br>%{y:.2f} %<extra></extra>",
            )
            st.plotly_chart(fig_bar, width="stretch", key="top10_bar_chart")

            top10_table = top10.copy().reset_index(drop=True)
            top10_table.insert(0, "Rangering", range(1, len(top10_table) + 1))
            top10_table = top10_table.rename(
                columns={"issuerName": "Selskap", "shortPercent": "Gjennomsnittlig short %"}
            )
            _render_table_header(
                "Topp 10-rangering",
                f"Tidsvektet dagsgjennomsnitt av offentlig shortandel i valgt {days}-dagersvindu.",
                f"{start_date.strftime('%d.%m')}–{latest_available.strftime('%d.%m.%Y')}",
            )
            st.dataframe(
                top10_table,
                width="stretch",
                hide_index=True,
                column_config={
                    "Rangering": st.column_config.NumberColumn("#", format="%d", width="small"),
                    "Selskap": st.column_config.TextColumn("Selskap", width="large"),
                    "Gjennomsnittlig short %": st.column_config.ProgressColumn(
                        "Gjennomsnittlig short %",
                        format="%.2f %%",
                        min_value=0.0,
                        max_value=max(
                            1.0,
                            float(top10_table["Gjennomsnittlig short %"].max()) * 1.08,
                        ),
                        width="medium",
                    ),
                },
            )

            names = top10["issuerName"].tolist()
            development = daily_series.loc[
                daily_series["issuerName"].isin(names)
            ].copy()
            if not development.empty:
                fig_line = px.line(
                    development,
                    x="date",
                    y="shortPercent",
                    color="issuerName",
                    title="Utvikling over tid for Topp 10",
                    labels={"date": "Dato", "shortPercent": "Shortandel (%)", "issuerName": "Selskap"},
                    color_discrete_sequence=CHART_PALETTE,
                )
                _style_plotly_chart(fig_line, height=600, hovermode="x unified")
                fig_line.update_layout(legend_title_text="Utsteder")
                fig_line.update_yaxes(ticksuffix=" %")
                fig_line.update_traces(
                    line=dict(width=2.6),
                    hovertemplate="%{y:.2f} %<extra></extra>",
                )
                st.plotly_chart(fig_line, width="stretch", key="top10_line_chart")

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
                        color_continuous_scale=[
                            [0.0, "#10B981"],
                            [0.5, "#F8FAFC"],
                            [1.0, "#EF4444"],
                        ],
                        color_continuous_midpoint=0,
                    )
                    _style_plotly_chart(fig_heat, height=600)
                    st.plotly_chart(fig_heat, width="stretch", key="top10_heatmap")

with tab_about:
    _render_section_header(
        "Plattform · metode (alle høgskoler og universiteter er opptatt av metode...)",
        "Om Shortregister",
        "Dette her er et uavhengig analyseverktøy som gjør offentlige SSR-data enklere å utforske – med tydelige forbehold om hva tallene faktisk viser.",
    )
    st.markdown(
        """
        <div class="about-grid">
            <article class="about-card">
                <div class="about-card-number">01 · DATA</div>
                <h3>Offentlig SSR-kilde</h3>
                <p>
                    Data hentes fra Finanstilsynets offentlige Short Sale Register.
                    Registeret viser offentlig rapporterbare nettoposisjoner på minst
                    0,5 %. Fravær av et selskap betyr derfor ikke nødvendigvis 0 % short.
                </p>
                <a href="https://ssr.finanstilsynet.no/" target="_blank">Åpne det offisielle registeret ↗</a>
            </article>
            <article class="about-card">
                <div class="about-card-number">02 · ANALYSER OG ANNET</div>
                <h3>Innsikt</h3>
                <p>
                    Plattformen jeg har laget samler liveposisjoner, posisjonsholdere og historikk i
                    én søkbar oversikt – med rangeringer, endringsanalyse, interaktive
                    diagrammer og CSV-eksport.
                </p>
                <a href="https://ssr.finanstilsynet.no/api/v2/" target="_blank">Se API-kilden ↗</a>
            </article>
            <article class="about-card">
                <div class="about-card-number">03 · PLATTFORMEN ER BYGGET MED</div>
                <h3>Python-basert analyse</h3>
                <p>
                Hva er brukt her?
                    Soråket er Python, og dette er brukt til datainnhenting, beregninger og logikk.</br>
                    <b>Streamlit:</b> Brukes til selve webapplikasjonen.</br>
                    <b>Pandas:</b> Det er for strukturering, filtrering og analyse.</br>
                    <b>Plotly:</b> Brukes til interaktive grafer og tidsserier i appen her.</br>
                    <b>SQLite:</b> lagring av historiske observasjoner. </br></p> <p>Prosjektet ble videreutviklet som del av et
                    innleveringsprosjekt ved University of Oxford – Saïd Business School: Algorithmic Trading Programme. Utviklet av Andreas Bolton Seielstad
                </p>
            </article>
        </div>
        <div class="method-note">
            <strong>Uavhengig og uoffisiell.</strong> Dette shortregister er ikke utviklet,
            godkjent eller drevet av Finanstilsynet. Tjenesten er laget for læring,
            markedsinnsikt og enklere tilgang til offentlige historiske data – ikke som
            investeringsråd. Jeg har foretatt et lite re-design av plattformen i etterkant når jeg var ferdig med faget og sertifiseringen.
        </div>
        """
        ,
        unsafe_allow_html=True,
    )
