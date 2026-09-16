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
    st.subheader("Hvem shorter aksjene?")
    st.caption(
        "Dette er individuelle offentlige shortposisjoner fra Finanstilsynets activePositions. "
        "Disse holdes separat fra den aggregerte historikken for å unngå dobbelttelling."
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
    view = view[["Selskap", "Posisjonsholder", "Dato", "Short %", "Aksjer", "ISIN"]]

    st.dataframe(
        view,
        width="stretch",
        hide_index=True,
        column_config={
            "Selskap": st.column_config.TextColumn("Selskap", width="large"),
            "Posisjonsholder": st.column_config.TextColumn("Posisjonsholder", width="large"),
            "Dato": st.column_config.TextColumn("Dato", width="small"),
            "Short %": st.column_config.NumberColumn("Short %", format="%.2f %%"),
            "Aksjer": st.column_config.NumberColumn("Aksjer", format="%d"),
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
        left, right = st.columns([1, 1], gap="medium")

        with left:
            st.markdown("### Største endringer")
            changes = beregn_storste_endringer(df)

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
                    column_config={
                        "Selskap": st.column_config.TextColumn("Selskap", width=200),
                        "Retning": st.column_config.TextColumn("Retning", width=90),
                        "Fra → til": st.column_config.TextColumn("Fra → til", width=155),
                        "Endring (pp)": st.column_config.NumberColumn(
                            "Endring (pp)", format="%.2f", width=95
                        ),
                        "Dato": st.column_config.TextColumn("Dato", width=90),
                    },
                )

        with right:
            st.markdown("### Nye posisjoner over 0,5 %")
            new_positions = finn_nye_shortposisjoner(df)

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
                    column_config={
                        "Selskap": st.column_config.TextColumn("Selskap", width=220),
                        "Fra → til": st.column_config.TextColumn("Fra → til", width=175),
                        "Ny short %": st.column_config.NumberColumn(
                            "Ny short %", format="%.2f %%", width=95
                        ),
                        "Dato": st.column_config.TextColumn("Dato", width=90),
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

    base_columns = ["Selskap", "Dato", "Short %", "Endring (pp)", "Trend"]
    advanced_columns = ["ISIN", "Posisjonsholder", "Aksjer"]
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

    column_config = {
        "Selskap": st.column_config.TextColumn("Selskap", width="large"),
        "Dato": st.column_config.TextColumn("Dato", width="small"),
        "Short %": st.column_config.NumberColumn("Short %", format="%.2f %%"),
        "Endring (pp)": st.column_config.NumberColumn("Endring (pp)", format="%+.2f"),
        "Trend": st.column_config.TextColumn("Trend", width="small"),
        "ISIN": st.column_config.TextColumn("ISIN", width="medium"),
        "Posisjonsholder": st.column_config.TextColumn("Posisjonsholder", width="medium"),
        "Aksjer": st.column_config.NumberColumn("Aksjer", format="%d"),
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

    @media (max-width: 760px) {
        .hero { padding: 23px 20px 18px; border-radius: 12px; }
        .brand-lockup { margin-bottom: 24px; }
        .hero .hero-title { font-size: 2.2rem; }
        .hero-lead { font-size: 1.08rem; }
        .hero-foot {
            flex-direction: column;
            align-items: flex-start;
            gap: 7px;
        }
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            gap: 0.75rem !important;
        }
        [data-testid="column"], [data-testid="stColumn"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        .stTabs [data-baseweb="tab"] { padding: 0 10px !important; }
    }
    </style>

    <div class="hero">
        <div class="hero-grid">
            <div>
                <div class="brand-lockup">
                    <div class="brand-mark">SR</div>
                    <div>
                        <span class="brand-name">Velkommen!</span>
                    </div>
                </div>
                <div class="hero-kicker">Denne plattformen/appen ble laget og utviklet ved Universitetet i Oxford - Säid Business School</div>
                <h1 class="hero-title">Shortregister over selskaper på<span> Oslo Børs</span></h1>
                <p class="hero-lead">Se deg gjerne litt rundt og scroll nedover for å se mer i appen og resten av registeret</p>
                <p class="hero-copy">
                    Her kan man også søke, sammenligne og følge utviklingen i offentlig rapporterte shortposisjoner.
                </p>
                <div class="hero-badges">
                    <span class="hero-badge">FINANSTILSYNET SSR V2-API</span>
                    <span class="hero-badge">OFFENTLIG TERSKELEN ER: ≥ 0,50 %</span>
                </div>
            </div>
        </div>
        <div class="hero-foot">
            <span>Utviklet av Andreas Bolton Seielstad</span>
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
    ["Live-søk over shortede selskaper", "Søk i selskaper", "Topp 10-shortede selskaper", "Om plattformen"]
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

        with st.expander("Hvor er Frontline og andre manglende selskaper?", expanded=False):
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

        st.markdown("### Markedssignaler")
        st.info(
            "Her vises største siste reduksjon, største siste økning og nyeste posisjon over 0,5 %. Tallene bygger på siste registrerte nivå per selskap."
        )
        signal_col1, signal_col2, signal_col3 = st.columns(3)

        decrease_title = html.escape(
            f"{decrease_company} → {decrease_value} → {decrease_detail}"
        )
        increase_title = html.escape(
            f"{increase_company} → {increase_value} → {increase_detail}"
        )
        new_title = html.escape(
            f"{new_company} → {new_value} → {new_detail}"
        )

        with signal_col1:
            st.markdown(
                f"""
                <div class="insight-card insight-card--down" title="{decrease_title}">
                    <div class="insight-kicker"> Største siste reduksjon</div>
                    <div class="insight-value">{html.escape(decrease_value)}</div>
                    <div class="insight-company">{html.escape(decrease_company)}</div>
                    <div class="insight-detail">{html.escape(decrease_detail)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with signal_col2:
            st.markdown(
                f"""
                <div class="insight-card insight-card--up" title="{increase_title}">
                    <div class="insight-kicker"> Største siste økning</div>
                    <div class="insight-value">{html.escape(increase_value)}</div>
                    <div class="insight-company">{html.escape(increase_company)}</div>
                    <div class="insight-detail">{html.escape(increase_detail)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with signal_col3:
            st.markdown(
                f"""
                <div class="insight-card insight-card--new" title="{new_title}">
                    <div class="insight-kicker"> Nyeste posisjon over 0,5 %</div>
                    <div class="insight-value">{html.escape(new_value)}</div>
                    <div class="insight-company">{html.escape(new_company)}</div>
                    <div class="insight-detail">{html.escape(new_detail)}</div>
                </div>
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
        st.subheader("Søk og filtrering")
        vis_sok_og_graf(df_live, "live", df_exempt)

    st.divider()
    st.subheader("Status for SQLite-registeret")
    latest_time, total_rows = hent_siste_oppdatering()
    if latest_time:
        st.markdown(f" Historikk sist oppdatert: {latest_time}  \n Totalt antall lagrede rader: {total_rows:,}")
    else:
        st.info("Ingen lagringshistorikk er registrert ennå.")

with tab_db:
    _render_section_header(
        "Historikk · selskapssøk",
        "Finn selskapet. Se utviklingen.",
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
        data = _standardiser_shortpercent(data)
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["issuerName", "date", "shortPercent"])

        period = st.selectbox("Velg tidsperiode", ["30 dager", "90 dager", "180 dager", "365 dager"])
        days = int(period.split()[0])
        latest_available = data["date"].max() if not data.empty else pd.NaT

        if pd.isna(latest_available):
            recent = pd.DataFrame(columns=data.columns)
        else:
            latest_available = latest_available.normalize()
            start_date = latest_available - pd.Timedelta(days=days)
            recent = data.loc[
                (data["date"] >= start_date)
                & (data["date"] <= latest_available)
            ]

            today = pd.Timestamp.today().normalize()
            age_days = max(0, int((today - latest_available).days))
            freshness = (
                "Siste observasjon er oppdatert."
                if age_days <= 1
                else f"Siste observasjon er {age_days} dager gammel."
            )
            st.caption(
                f"Analysevindu: {start_date.strftime('%d.%m.%Y')}–"
                f"{latest_available.strftime('%d.%m.%Y')} · {freshness}"
            )

        if recent.empty:
            st.info("Fant ingen gyldige daterte observasjoner i datagrunnlaget.")
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
                color_discrete_sequence=[CHART_PALETTE[0]],
            )
            _style_plotly_chart(fig_bar, height=500)
            fig_bar.update_xaxes(tickangle=-28)
            fig_bar.update_traces(
                marker=dict(
                    color="#2563EB",
                    line=dict(color="#C7DBFF", width=1.2),
                ),
                textposition="outside",
                cliponaxis=False,
            )
            st.plotly_chart(fig_bar, width="stretch", key="top10_bar_chart")
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
                    color_discrete_sequence=CHART_PALETTE,
                )
                _style_plotly_chart(fig_line, height=600, hovermode="x unified")
                fig_line.update_layout(legend_title_text="Utsteder")
                fig_line.update_traces(line=dict(width=2.6))
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
        "Plattform · metode",
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
                    Utviklet av Andreas Bolton Seielstad med Python, Streamlit, Pandas,
                    Plotly og SQLite. Prosjektet ble videreutviklet som del av et
                    innleveringsprosjekt ved University of Oxford – Saïd Business School: Algorithmic Trading Programme.
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
