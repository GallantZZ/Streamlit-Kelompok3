from __future__ import annotations

import io
import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
px.defaults.template = "plotly_dark"

st.set_page_config(
    page_title="Instagram Engagement Modeling",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42
SAMPLE_SIZE = 50_000
TARGET = "engagement"
MODEL_FEATURES = ["stories", "reels_watched", "ads_clicked", "posts", "followers"]
DEFAULT_SOURCE_NAMES = {
    "DatasetProjectDatmod.csv",
    "DatasetProjectDatmod.xlsx",
    "dataset.csv",
    "dataset.xlsx",
    "df2_clean.csv",
    "df2_clean.xls",
    "df2_clean.xlsx",
}

LOG_MAPPING = {
    "daily_active_minutes_instagram": "active_minutes_log",
    "followers_count": "followers_log",
    "following_count": "following_log",
    "likes_given_per_day": "likes_log",
    "comments_written_per_day": "comments_log",
    "sessions_per_day": "sessions_log",
    "posts_created_per_week": "posts_log",
    "time_on_feed_per_day": "feed_time_log",
    "time_on_reels_per_day": "reels_time_log",
    "user_engagement_score": "engagement_log",
    "time_on_explore_per_day": "explore_time_log",
    "time_on_messages_per_day": "messages_time_log",
    "dms_sent_per_week": "dms_sent_log",
    "dms_received_per_week": "dms_received_log",
    "ads_viewed_per_day": "ads_viewed_log",
    "ads_clicked_per_day": "ads_clicked_log",
    "stories_viewed_per_day": "stories_log",
    "reels_watched_per_day": "reels_watched_log",
}

RENAME_MAPPING = {
    "engagement_log": "engagement",
    "likes_log": "likes",
    "comments_log": "comments",
    "followers_log": "followers",
    "following_log": "following",
    "active_minutes_log": "active_minutes",
    "feed_time_log": "feed_time",
    "reels_time_log": "reels_time",
    "sessions_log": "sessions",
    "posts_log": "posts",
    "explore_time_log": "explore_time",
    "messages_time_log": "messages_time",
    "dms_sent_log": "dms_sent",
    "dms_received_log": "dms_received",
    "ads_viewed_log": "ads_viewed",
    "ads_clicked_log": "ads_clicked",
    "stories_log": "stories",
    "reels_watched_log": "reels_watched",
}

DROP_COLUMNS = [
    "app_name",
    "user_id",
    "user_engagement_score",
    "likes_given_per_day",
    "comments_written_per_day",
    "followers_count",
    "following_count",
    "daily_active_minutes_instagram",
    "time_on_feed_per_day",
    "time_on_reels_per_day",
    "sessions_per_day",
    "posts_created_per_week",
    "last_login_date",
]

BINARY_COLUMNS = {
    "has_children": "has_children_binary",
    "uses_premium_features": "uses_premium_features_binary",
    "two_factor_auth_enabled": "two_factor_auth_enabled_binary",
    "biometric_login_used": "biometric_login_used_binary",
}

ORDINAL_MAPS = {
    "income_level": {
        "Low": 1,
        "Lower-middle": 2,
        "Middle": 3,
        "Upper-middle": 4,
        "High": 5,
    },
    "education_level": {
        "High school": 1,
        "Some college": 2,
        "Bachelor's": 3,
        "Master's": 4,
        "PhD": 5,
        "Other": 6,
    },
    "alcohol_frequency": {
        "Never": 0,
        "Rarely": 1,
        "Weekly": 2,
        "Several times a week": 3,
        "Daily": 4,
    },
    "privacy_setting_level": {
        "Public": 1,
        "Friends only": 2,
        "Private": 3,
    },
}

FEATURE_LABELS = {
    "stories": "Stories viewed",
    "reels_watched": "Reels watched",
    "ads_clicked": "Ads clicked",
    "posts": "Posts created",
    "followers": "Followers",
    "active_minutes": "Active minutes",
    "feed_time": "Feed time",
    "reels_time": "Reels time",
    "messages_time": "Messages time",
    "explore_time": "Explore time",
    "likes": "Likes",
    "comments": "Comments",
    "dms_received": "DMs received",
    "ads_viewed": "Ads viewed",
    "sessions": "Sessions",
    "dms_sent": "DMs sent",
    "following": "Following",
    "age": "Age",
    "self_reported_happiness": "Happiness score",
    "perceived_stress_score": "Stress score",
    "engagement": "Engagement",
}

STATIC_RESULTS = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Random Forest",
            "Random Forest (Optuna)",
            "Gradient Boosting",
            "XGBoost",
            "LightGBM",
            "CatBoost",
        ],
        "R2": [0.644, 0.884, 0.886, 0.885, 0.885, 0.885, 0.889],
        "MAE": [0.154, 0.063, 0.063, 0.063, 0.063, 0.063, 0.063],
        "RMSE": [0.223, 0.127, 0.126, 0.126, 0.127, 0.127, 0.125],
    }
)

STATIC_CATBOOST_IMPORTANCE = pd.DataFrame(
    {
        "Feature": ["stories", "reels_watched", "posts", "ads_clicked", "followers"],
        "Importance": [51.908273, 40.623315, 5.383483, 1.628856, 0.456073],
    }
)

PAGE_BG = "#030605"
SURFACE = "#08110f"
SURFACE_2 = "#0d1916"
CHART_INK = "#f7fff9"
CHART_MUTED = "#9fb9af"
CHART_GRID = "#173f37"
CHART_LINE = "#245246"
JADE = "#2dd4bf"
JADE_DARK = "#0f766e"
JADE_LIGHT = "#a7f3d0"
ROSE = "#fb7185"
SLATE_SOFT = "#5f7f74"
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
}

CORRELATION_COLUMNS = [
    "active_minutes",
    "feed_time",
    "reels_time",
    "messages_time",
    "explore_time",
    "stories",
    "likes",
    "comments",
    "dms_received",
    "ads_viewed",
    "sessions",
    "reels_watched",
    "ads_clicked",
    "dms_sent",
    "posts",
    "followers",
    "following",
    "age",
    "self_reported_happiness",
    "perceived_stress_score",
    "has_children_binary",
    "income_level_ordinal",
    "education_level_ordinal",
    "alcohol_frequency_ordinal",
    "content_type_preference",
    "preferred_content_theme",
    "engagement",
]


st.markdown(
    """
    <style>
    :root {
        --ink: #f7fff9;
        --muted: #9fb9af;
        --line: #173f37;
        --paper: #08110f;
        --soft: #0d1916;
        --primary: #2dd4bf;
        --teal: #2dd4bf;
        --negative: #fb7185;
        --slate: #d6fff3;
    }
    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #030605;
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: rgba(3, 6, 5, 0.9);
    }
    [data-testid="stExpandSidebarButton"],
    section[data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"]:has([data-testid="stIconMaterial"]) {
        background: #08110f !important;
        border: 1px solid #2dd4bf !important;
        border-radius: 8px !important;
        color: #f7fff9 !important;
        box-shadow: 0 0 0 1px rgba(45, 212, 191, 0.1), 0 12px 30px rgba(45, 212, 191, 0.14) !important;
    }
    [data-testid="stExpandSidebarButton"]:hover,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"]:has([data-testid="stIconMaterial"]):hover {
        background: #0d1916 !important;
        border-color: #5eead4 !important;
    }
    [data-testid="stExpandSidebarButton"] span,
    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"],
    section[data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"]:has([data-testid="stIconMaterial"]) span,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"]:has([data-testid="stIconMaterial"]) svg,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"]:has([data-testid="stIconMaterial"]) [data-testid="stIconMaterial"] {
        color: #f7fff9 !important;
        fill: #f7fff9 !important;
        opacity: 1 !important;
    }
    .main .block-container {
        max-width: 1260px;
        padding-top: 1.5rem;
        padding-bottom: 2.2rem;
    }
    section[data-testid="stSidebar"] {
        background: #050908;
        border-right: 1px solid #173f37;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #f7fff9 !important;
    }
    section[data-testid="stSidebar"] small,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"],
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #9fb9af !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #08110f;
        border: 1px dashed #2dd4bf;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {
        background: #08110f !important;
        border-color: #173f37 !important;
        color: #f7fff9 !important;
    }
    section[data-testid="stSidebar"] button {
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #173f37;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }
    .main h1,
    .main h2,
    .main h3,
    .main h4,
    .main p,
    .main label,
    .main span,
    .main [data-testid="stMarkdownContainer"] {
        color: var(--ink) !important;
    }
    .main .section-copy,
    .main .metric-card .label,
    .main .metric-card .caption,
    .main .insight-card p,
    .main .hero p {
        color: var(--muted) !important;
    }
    p, li, label, span {
        letter-spacing: 0;
    }
    button[data-baseweb="tab"] p {
        color: var(--slate) !important;
        font-weight: 650;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: var(--primary) !important;
    }
    [data-baseweb="tab-highlight"] {
        background-color: var(--primary) !important;
    }
    div[data-testid="stMetricValue"] {
        color: var(--ink);
    }
    .hero {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 26px 28px;
        background: #08110f;
        margin-bottom: 18px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(167, 243, 208, 0.08);
    }
    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 2.15rem;
        line-height: 1.1;
    }
    .hero p {
        color: var(--muted);
        max-width: 860px;
        margin: 0;
        font-size: 1rem;
        line-height: 1.6;
    }
    .eyebrow {
        color: var(--primary);
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        margin-bottom: 7px;
    }
    .metric-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--paper);
        padding: 16px 17px;
        min-height: 124px;
        box-shadow: 0 16px 34px rgba(0, 0, 0, 0.26);
    }
    .metric-card .label {
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 650;
        margin-bottom: 8px;
    }
    .metric-card .value {
        color: var(--primary);
        font-size: 1.72rem;
        font-weight: 760;
        line-height: 1.15;
        margin-bottom: 8px;
    }
    .metric-card .caption {
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.35;
    }
    .insight-card {
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        background: #08110f;
        border-top: 1px solid var(--line);
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        padding: 15px 16px;
        min-height: 112px;
    }
    .insight-card h4 {
        margin: 0 0 8px 0;
        color: var(--ink);
        font-size: 0.98rem;
    }
    .insight-card p {
        margin: 0;
        color: var(--muted);
        line-height: 1.5;
        font-size: 0.9rem;
    }
    .soft-note {
        border: 1px solid var(--line);
        background: var(--soft);
        color: var(--ink);
        border-radius: 8px;
        padding: 14px 16px;
        line-height: 1.55;
    }
    .section-title {
        margin-top: 8px;
        margin-bottom: 4px;
        font-size: 1.35rem;
        font-weight: 760;
        color: var(--ink);
    }
    .section-copy {
        color: var(--muted);
        margin-top: 0;
        margin-bottom: 14px;
        max-width: 900px;
        line-height: 1.55;
    }
    .main .js-plotly-plot .main-svg {
        background: #030605 !important;
    }
    .main .js-plotly-plot .gtitle,
    .main .js-plotly-plot .xtitle,
    .main .js-plotly-plot .ytitle,
    .main .js-plotly-plot .legendtext,
    .main .js-plotly-plot .legendtitletext,
    .main .js-plotly-plot .xtick text,
    .main .js-plotly-plot .ytick text,
    .main .js-plotly-plot .cbtitle text,
    .main .js-plotly-plot .colorbar text {
        fill: #f7fff9 !important;
        color: #f7fff9 !important;
        opacity: 1 !important;
    }
    .main .js-plotly-plot .gridlayer path {
        stroke: #173f37 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def first_available_column(df: pd.DataFrame, columns: list[str]) -> str | None:
    return next((col for col in columns if col in df.columns), None)


def format_number(value: float | int | None, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.{digits}f}"


def format_decimal(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def metric_card(label: str, value: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="insight-card">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, copy: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if copy:
        st.markdown(f'<p class="section-copy">{copy}</p>', unsafe_allow_html=True)


def soft_note(text: str) -> None:
    st.markdown(f'<div class="soft-note">{text}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def read_table_path(path: str) -> pd.DataFrame:
    with open(path, "rb") as file:
        return read_table_bytes(file.read(), path)


@st.cache_data(show_spinner=False)
def read_table_bytes(data: bytes, source_name: str) -> pd.DataFrame:
    source_ext = os.path.splitext(source_name)[1].lower()
    is_excel_binary = data.startswith(b"\xd0\xcf\x11\xe0") or data.startswith(b"PK\x03\x04")

    if is_excel_binary:
        try:
            return pd.read_excel(io.BytesIO(data), sheet_name=0)
        except Exception as exc:
            raise ValueError(
                "File Excel gagal dibaca. Pastikan file tidak corrupt dan dependencies deploy sudah terinstall."
            ) from exc

    if source_ext in {".csv", ".xls", ".xlsx"}:
        for encoding in ("utf-8", "utf-8-sig", "latin1"):
            try:
                return pd.read_csv(io.BytesIO(data), encoding=encoding)
            except UnicodeDecodeError:
                continue
            except Exception:
                if source_ext == ".csv":
                    raise
                break

    try:
        return pd.read_excel(io.BytesIO(data), sheet_name=0)
    except Exception as exc:
        raise ValueError("Dataset harus berupa CSV, XLS, atau XLSX yang valid.") from exc


def is_default_source_name(source_name: str | None) -> bool:
    return bool(source_name) and os.path.basename(source_name) in DEFAULT_SOURCE_NAMES


def read_dataset(uploaded_file) -> tuple[pd.DataFrame | None, str | None]:
    if uploaded_file is not None:
        return read_table_bytes(uploaded_file.getvalue(), uploaded_file.name), uploaded_file.name

    default_file_names = [
        "DatasetProjectDatmod.csv",
        "DatasetProjectDatmod.xlsx",
        "dataset.csv",
        "dataset.xlsx",
        "df2_clean.csv",
        "df2_clean.xls",
        "df2_clean.xlsx",
    ]

    app_dir = os.path.dirname(os.path.abspath(__file__))
    working_dir = os.getcwd()
    search_dirs = [
        app_dir,
        working_dir,
        os.path.join(app_dir, "data"),
        os.path.join(working_dir, "data"),
        os.path.join(working_dir, "datmod-streamlit"),
        os.path.join(working_dir, "datmod-streamlit", "data"),
    ]

    checked_paths = []
    for folder in search_dirs:
        for file_name in default_file_names:
            path = os.path.join(folder, file_name)
            if path not in checked_paths:
                checked_paths.append(path)

    for path in checked_paths:
        if os.path.exists(path):
            return read_table_path(path), path

    return None, None


def clean_text_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("\u2019", "'", regex=False)
        .str.replace("\u2018", "'", regex=False)
        .str.strip()
    )


@st.cache_data(show_spinner=False)
def prepare_data(df_raw: pd.DataFrame, sample_size: int = SAMPLE_SIZE) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = df_raw.copy()
    initial_rows = len(df)
    actual_sample_size = min(sample_size, initial_rows)

    if initial_rows > actual_sample_size:
        df = df.sample(n=actual_sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if "last_login_date" in df.columns:
        df["last_login_date"] = pd.to_datetime(df["last_login_date"], errors="coerce")
        latest_login = df["last_login_date"].max()
        df["days_since_last_login"] = (latest_login - df["last_login_date"]).dt.days

    for original, log_col in LOG_MAPPING.items():
        if original in df.columns:
            numeric = pd.to_numeric(df[original], errors="coerce")
            df[log_col] = np.log1p(numeric.clip(lower=0))

    for original, new_col in BINARY_COLUMNS.items():
        if original in df.columns:
            df[new_col] = clean_text_series(df[original]).map({"No": 0, "Yes": 1})

    for original, mapping in ORDINAL_MAPS.items():
        if original in df.columns:
            df[original] = clean_text_series(df[original])
            df[f"{original}_ordinal"] = df[original].map(mapping)

    df2 = df.copy()
    outlier_columns = [col for col in df2.columns if "outlier" in col.lower()]
    df2 = df2.drop(columns=available_columns(df2, DROP_COLUMNS) + outlier_columns, errors="ignore")

    rename_columns = {old: new for old, new in RENAME_MAPPING.items() if old in df2.columns}
    existing_renamed_columns = [
        new for old, new in rename_columns.items() if new in df2.columns and old != new
    ]
    df2 = df2.drop(columns=existing_renamed_columns, errors="ignore")
    df2 = df2.rename(columns=rename_columns)

    if TARGET in df2.columns and "engagement_raw" not in df2.columns:
        if "user_engagement_score" in df.columns:
            df2["engagement_raw"] = pd.to_numeric(df["user_engagement_score"], errors="coerce")
        else:
            df2["engagement_raw"] = np.expm1(pd.to_numeric(df2[TARGET], errors="coerce"))

    if df2.columns.duplicated().any():
        df2 = df2.loc[:, ~df2.columns.duplicated()]

    metadata = {
        "initial_rows": initial_rows,
        "sample_rows": len(df),
        "initial_columns": df_raw.shape[1],
        "clean_columns": df2.shape[1],
        "source_kind": "Raw dataset" if "user_engagement_score" in df_raw.columns else "Clean dataset",
        "missing_total": int(df.isna().sum().sum()),
    }

    return df, df2, metadata


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    with st.sidebar.expander("Optional filters", expanded=False):
        st.caption(
            "Filter dipakai kalau ingin melihat subset audience tertentu. Kosongkan semua pilihan untuk memakai seluruh data."
        )

        filter_columns = [
            ("content_type_preference", "Content type"),
            ("preferred_content_theme", "Content theme"),
            ("subscription_status", "Subscription"),
            ("employment_status", "Employment"),
        ]

        for column, label in filter_columns:
            if column not in filtered.columns:
                continue
            options = sorted(filtered[column].dropna().astype(str).unique().tolist())
            selected = st.multiselect(label, options, default=[], placeholder="All")
            if selected:
                filtered = filtered[filtered[column].astype(str).isin(selected)]

    return filtered


def validate_model_data(df: pd.DataFrame) -> list[str]:
    return [col for col in MODEL_FEATURES + [TARGET] if col not in df.columns]


def calculate_correlation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = available_columns(df, CORRELATION_COLUMNS)
    if TARGET not in selected:
        return pd.DataFrame(), pd.DataFrame()

    corr_df = df[selected].copy()
    dummy_columns = available_columns(corr_df, ["content_type_preference", "preferred_content_theme"])
    if dummy_columns:
        corr_df = pd.get_dummies(corr_df, columns=dummy_columns, drop_first=True)

    corr_df = corr_df.replace([np.inf, -np.inf], np.nan).dropna()
    if corr_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    corr = corr_df.corr(numeric_only=True)
    if TARGET not in corr.columns:
        return pd.DataFrame(), pd.DataFrame()

    ranking = (
        corr[TARGET]
        .drop(TARGET, errors="ignore")
        .sort_values()
        .reset_index()
        .rename(columns={"index": "Feature", TARGET: "Correlation"})
    )

    top_features = (
        corr[TARGET]
        .drop(TARGET, errors="ignore")
        .abs()
        .sort_values(ascending=False)
        .head(12)
        .index.tolist()
    )
    corr_focus = corr_df[top_features + [TARGET]].corr(numeric_only=True)
    return corr_focus, ranking


def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    if validate_model_data(df):
        return pd.DataFrame()

    vif_source = df[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vif_source) < len(MODEL_FEATURES) + 1:
        return pd.DataFrame()

    vif_df = pd.DataFrame({"Feature": MODEL_FEATURES})
    vif_df["VIF"] = [
        variance_inflation_factor(vif_source.values, i)
        for i in range(len(MODEL_FEATURES))
    ]
    return vif_df.sort_values("VIF", ascending=False)


def describe_engagement_distribution(df: pd.DataFrame) -> tuple[str, str]:
    if TARGET not in df.columns or df[TARGET].dropna().empty:
        return "Engagement belum tersedia", "Kolom engagement tidak ditemukan pada dataset aktif."

    values = pd.to_numeric(df[TARGET], errors="coerce").dropna()
    skewness = values.skew()
    mean = values.mean()
    median = values.median()

    if skewness >= 1:
        shape = "right-skew kuat"
    elif skewness >= 0.35:
        shape = "right-skew ringan"
    elif skewness <= -0.35:
        shape = "left-skew"
    else:
        shape = "cukup seimbang"

    return (
        "Distribusi engagement",
        f"Dataset aktif memiliki distribusi {shape}. Mean {mean:.2f} dan median {median:.2f}, sehingga ringkasan model sebaiknya tetap dibaca bersama sebaran datanya.",
    )


def describe_strongest_relation(df: pd.DataFrame) -> tuple[str, str]:
    _, ranking = calculate_correlation(df)
    if ranking.empty:
        return "Relasi belum terbaca", "Korelasi belum bisa dihitung karena kolom numerik yang dibutuhkan belum lengkap."

    ranked = ranking.copy()
    ranked["Abs"] = ranked["Correlation"].abs()
    strongest = ranked.sort_values("Abs", ascending=False).iloc[0]
    feature = FEATURE_LABELS.get(strongest["Feature"], strongest["Feature"])
    corr = strongest["Correlation"]
    direction = "positif" if corr > 0 else "negatif"
    strength = "kuat" if abs(corr) >= 0.7 else "sedang" if abs(corr) >= 0.35 else "lemah"

    return (
        "Relasi terkuat",
        f"Pada data aktif, {feature} punya korelasi {direction} {strength} terhadap engagement (r={corr:.2f}).",
    )


def describe_model_readiness(df: pd.DataFrame, source_name: str | None) -> tuple[str, str]:
    missing = validate_model_data(df)
    if missing:
        return "Model belum siap", "Kolom model yang belum ada: " + ", ".join(missing) + "."
    if is_default_source_name(source_name):
        return "Model siap dipakai", "Dataset bawaan memiliki semua fitur model dan hasil evaluasi CatBoost tersedia sebagai acuan."
    return (
        "Dataset upload terdeteksi",
        "Fitur model lengkap, tetapi metrik evaluasi perlu dihitung ulang jika dataset yang dipakai berbeda dari dataset bawaan.",
    )


def describe_feature_relation(df: pd.DataFrame, feature: str) -> str:
    if feature not in df.columns or TARGET not in df.columns:
        return "Fitur atau target engagement belum tersedia pada dataset aktif."

    pair = df[[feature, TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3:
        return "Data terlalu sedikit untuk membaca hubungan fitur ini dengan engagement."

    corr = pair[feature].corr(pair[TARGET])
    if pd.isna(corr):
        return "Hubungan fitur ini belum bisa dihitung karena variasi data terlalu kecil."

    direction = "positif" if corr > 0 else "negatif"
    strength = "kuat" if abs(corr) >= 0.7 else "sedang" if abs(corr) >= 0.35 else "lemah"
    label = FEATURE_LABELS.get(feature, feature)
    return (
        f"Pada data aktif, {label} memiliki hubungan {direction} {strength} terhadap engagement "
        f"(r={corr:.2f}). Gunakan pola garis median untuk melihat arah utamanya."
    )


def describe_category_average(df: pd.DataFrame, column: str, label: str) -> tuple[str, str]:
    if column not in df.columns or TARGET not in df.columns:
        return f"{label} belum tersedia", f"Kolom {label} atau engagement tidak ditemukan pada dataset aktif."

    grouped = (
        df.groupby(column, dropna=False)[TARGET]
        .mean()
        .sort_values(ascending=False)
    )
    if grouped.empty:
        return f"{label} belum terbaca", "Data kategori belum cukup untuk membuat ringkasan."

    top_name = str(grouped.index[0])
    top_value = grouped.iloc[0]
    if len(grouped) == 1:
        return f"{label}: {top_name}", f"Satu kategori tersedia dengan rata-rata engagement {top_value:.2f}."

    bottom_name = str(grouped.index[-1])
    bottom_value = grouped.iloc[-1]
    gap = top_value - bottom_value
    return (
        f"{label}: {top_name} tertinggi",
        f"Rata-rata engagement {top_name} adalah {top_value:.2f}, selisih {gap:.2f} dibanding {bottom_name}.",
    )


def build_activity_trend(df: pd.DataFrame, feature: str, bins: int = 22) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = df[[feature, TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
    if source.empty:
        return pd.DataFrame(), pd.DataFrame()

    sample = source.sample(min(len(source), 2500), random_state=RANDOM_STATE)
    unique_values = source[feature].nunique()
    if unique_values < 3:
        return pd.DataFrame(), sample

    source = source.copy()
    source["bin"] = pd.qcut(
        source[feature],
        q=min(bins, unique_values),
        duplicates="drop",
    )

    trend = (
        source.groupby("bin", observed=True)
        .agg(
            x=(feature, "mean"),
            median=(TARGET, "median"),
            q25=(TARGET, lambda values: values.quantile(0.25)),
            q75=(TARGET, lambda values: values.quantile(0.75)),
            users=(TARGET, "size"),
        )
        .reset_index(drop=True)
    )
    return trend, sample


def theme_layout(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=18, r=24, t=62, b=42),
        paper_bgcolor=PAGE_BG,
        plot_bgcolor=PAGE_BG,
        font=dict(family="Arial, sans-serif", color=CHART_INK, size=13),
        title=dict(font=dict(color=CHART_INK, size=17), x=0, xanchor="left"),
        xaxis_title_font=dict(color=CHART_INK, size=13),
        yaxis_title_font=dict(color=CHART_INK, size=13),
        legend=dict(
            font=dict(color=CHART_INK, size=12),
            bgcolor="rgba(8, 17, 15, 0.92)",
            bordercolor=CHART_LINE,
            borderwidth=1,
        ),
        legend_title_text="",
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=JADE,
            font=dict(color=CHART_INK, size=12),
        ),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=CHART_GRID,
        zeroline=False,
        linecolor=CHART_LINE,
        tickfont=dict(color=CHART_INK, size=12),
        title_font=dict(color=CHART_INK, size=13),
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=CHART_GRID,
        zeroline=False,
        linecolor=CHART_LINE,
        tickfont=dict(color=CHART_INK, size=12),
        title_font=dict(color=CHART_INK, size=13),
        automargin=True,
    )
    fig.update_coloraxes(
        colorbar=dict(
            tickfont=dict(color=CHART_INK, size=12),
            title=dict(font=dict(color=CHART_INK, size=12)),
            outlinecolor=CHART_LINE,
            tickformat=".2f",
        )
    )
    fig.update_traces(textfont=dict(color=CHART_INK), selector=dict(type="bar"))
    fig.update_traces(textfont=dict(color=CHART_INK), selector=dict(type="pie"))
    fig.update_traces(textfont=dict(color=CHART_INK), selector=dict(type="heatmap"))
    return fig


def render_chart(fig: go.Figure, height: int = 430) -> None:
    st.plotly_chart(
        theme_layout(fig, height),
        width="stretch",
        theme=None,
        config=PLOTLY_CONFIG,
    )


def render_hero(meta: dict, source_name: str | None) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">Instagram Engagement Analysis</div>
            <h1>Instagram Engagement Modeling</h1>
            <p>
                Dashboard ini menjelaskan pola engagement dari dataset ini: bagaimana aktivitas pengguna
                berhubungan dengan engagement, fitur apa yang paling berpengaruh, dan insight konten apa
                yang bisa dipakai untuk membaca perilaku audience.
                Sumber data: {source_name or "tidak ditemukan"}.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(df: pd.DataFrame, df_full: pd.DataFrame, meta: dict) -> None:
    source_name = st.session_state.get("source_name")
    render_hero(meta, source_name)

    best = STATIC_RESULTS.sort_values("R2", ascending=False).iloc[0]
    high_threshold = df_full[TARGET].quantile(0.75) if TARGET in df_full.columns else np.nan
    high_share = (df[TARGET] >= high_threshold).mean() * 100 if TARGET in df.columns else np.nan
    model_title, model_body = describe_model_readiness(df, source_name)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            "Data aktif",
            format_number(len(df)),
            f"{format_number(meta['sample_rows'])} baris tersedia setelah sampling",
        )
    with c2:
        metric_card(
            "Rata-rata engagement",
            format_number(df[TARGET].mean(), 2) if TARGET in df.columns else "-",
            "Target yang dianalisis",
        )
    with c3:
        metric_card(
            "User engagement tinggi",
            f"{high_share:.1f}%" if not pd.isna(high_share) else "-",
            "Proporsi di atas kuartil 3",
        )
    with c4:
        metric_card(
            model_title,
            best["Model"] if is_default_source_name(source_name) else "Lengkap",
            f"R2 {best['R2']:.2f}, RMSE {best['RMSE']:.2f}" if is_default_source_name(source_name) else "Evaluasi ulang untuk dataset upload",
        )

    st.write("")
    dist_title, dist_body = describe_engagement_distribution(df)
    rel_title, rel_body = describe_strongest_relation(df)
    i1, i2, i3 = st.columns(3)
    with i1:
        insight_card(dist_title, dist_body)
    with i2:
        insight_card(rel_title, rel_body)
    with i3:
        insight_card(model_title, model_body)

    st.write("")
    left, right = st.columns([1.3, 1])
    with left:
        section_title("Distribusi Engagement", "Satu grafik untuk melihat sebaran target yang sedang dianalisis.")
        hist_col = TARGET if TARGET in df.columns else first_available_column(df, ["engagement_raw"])
        if hist_col:
            fig = px.histogram(
                df,
                x=hist_col,
                nbins=55,
                color_discrete_sequence=[JADE],
                labels={hist_col: FEATURE_LABELS.get(hist_col, hist_col)},
                title="Sebaran Engagement",
            )
            fig.update_traces(
                hovertemplate=(
                    f"{FEATURE_LABELS.get(hist_col, hist_col)}=%{{x:.2f}}<br>"
                    "Jumlah user=%{y:,}<extra></extra>"
                )
            )
            render_chart(fig, 430)

    with right:
        section_title("Rata-rata Waktu Penggunaan", "Perbandingan waktu pengguna pada feed, reels, dan messages.")
        time_rows = []
        for label, columns in {
            "Feed": ["time_on_feed_per_day", "feed_time"],
            "Reels": ["time_on_reels_per_day", "reels_time"],
            "Messages": ["time_on_messages_per_day", "messages_time"],
        }.items():
            column = first_available_column(df, columns)
            if column:
                time_rows.append(
                    {
                        "Feature": label,
                        "Average": df[column].mean(),
                        "Scale": "Raw minutes" if column == columns[0] else "Log scale",
                    }
                )
        if time_rows:
            time_df = pd.DataFrame(time_rows).sort_values("Average", ascending=True)
            fig = px.bar(
                time_df,
                x="Average",
                y="Feature",
                color="Scale",
                orientation="h",
                color_discrete_map={"Raw minutes": JADE_DARK, "Log scale": JADE},
                title="Average Usage Time",
            )
            fig.update_traces(
                hovertemplate="Aktivitas=%{y}<br>Rata-rata=%{x:.2f}<extra></extra>"
            )
            render_chart(fig, 430)


def render_activity_patterns(df: pd.DataFrame) -> None:
    section_title(
        "Activity Patterns",
        "Lihat arah hubungan setiap aktivitas dengan engagement. Garis utama menunjukkan median engagement, sedangkan area tipis menunjukkan rentang kuartil.",
    )

    missing = validate_model_data(df)
    if missing:
        soft_note("Kolom berikut belum tersedia untuk activity patterns: " + ", ".join(missing))
        return

    driver_label = st.selectbox(
        "Model feature",
        MODEL_FEATURES,
        format_func=lambda col: FEATURE_LABELS.get(col, col),
    )

    left, right = st.columns([1.35, 1])
    with left:
        trend, sample = build_activity_trend(df, driver_label)
        fig = go.Figure()
        if not sample.empty:
            fig.add_trace(
                go.Scattergl(
                    x=sample[driver_label],
                    y=sample[TARGET],
                    mode="markers",
                    marker=dict(size=5, color="rgba(45, 212, 191, 0.18)"),
                    name="User sample",
                    hovertemplate=(
                        f"{FEATURE_LABELS.get(driver_label, driver_label)}=%{{x:.2f}}<br>"
                        "Engagement=%{y:.2f}<extra></extra>"
                    ),
                )
            )
        if not trend.empty:
            fig.add_trace(
                go.Scatter(
                    x=trend["x"],
                    y=trend["q25"],
                    mode="lines",
                    line=dict(width=0, color="rgba(45, 212, 191, 0)"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=trend["x"],
                    y=trend["q75"],
                    mode="lines",
                    line=dict(width=0, color="rgba(45, 212, 191, 0)"),
                    fill="tonexty",
                    fillcolor="rgba(45, 212, 191, 0.18)",
                    name="Rentang tengah 50%",
                    hovertemplate="Batas atas rentang tengah=%{y:.2f}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=trend["x"],
                    y=trend["median"],
                    mode="lines+markers",
                    line=dict(width=3, color=JADE),
                    marker=dict(size=7, color=JADE_LIGHT, line=dict(color=JADE, width=1)),
                    name="Median engagement",
                    hovertemplate="Median engagement=%{y:.2f}<extra></extra>",
                )
            )
        fig.update_layout(
            title=f"Trend Engagement: {FEATURE_LABELS.get(driver_label, driver_label)}",
            xaxis_title=FEATURE_LABELS.get(driver_label, driver_label),
            yaxis_title="Engagement",
        )
        render_chart(fig, 520)

    with right:
        soft_note(describe_feature_relation(df, driver_label))
        st.write("")
        corr_focus, ranking = calculate_correlation(df)
        if not ranking.empty:
            compact = ranking.copy()
            compact["Abs"] = compact["Correlation"].abs()
            compact = compact.sort_values("Abs", ascending=False).head(12)
            compact["Direction"] = np.where(compact["Correlation"] >= 0, "Positive", "Negative")
            compact["Feature"] = compact["Feature"].map(lambda x: FEATURE_LABELS.get(x, x))
            fig = px.bar(
                compact.sort_values("Correlation"),
                x="Correlation",
                y="Feature",
                color="Direction",
                orientation="h",
                custom_data=["Direction"],
                color_discrete_map={"Positive": JADE, "Negative": ROSE},
                title="Korelasi Terkuat terhadap Engagement",
            )
            fig.update_traces(
                hovertemplate=(
                    "Fitur=%{y}<br>"
                    "Korelasi=%{x:.2f}<br>"
                    "Arah=%{customdata[0]}<extra></extra>"
                )
            )
            fig.add_vline(x=0, line_width=1, line_color=CHART_LINE)
            render_chart(fig, 400)

    corr_focus, _ = calculate_correlation(df)
    if not corr_focus.empty:
        section_title("Matriks Korelasi", "Ringkasan variabel yang paling kuat hubungannya dengan engagement.")
        label_map = {col: FEATURE_LABELS.get(col, col) for col in corr_focus.columns}
        corr_named = corr_focus.rename(index=label_map, columns=label_map)
        fig = px.imshow(
            corr_named,
            text_auto=".2f",
            color_continuous_scale=[ROSE, PAGE_BG, JADE],
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Focused Correlation Matrix",
        )
        fig.update_traces(
            hovertemplate=(
                "Baris=%{y}<br>"
                "Kolom=%{x}<br>"
                "Korelasi=%{z:.2f}<extra></extra>"
            )
        )
        render_chart(fig, 650)


def render_model_evidence(df: pd.DataFrame, source_name: str | None) -> None:
    section_title(
        "Model Evidence",
        "Model menggunakan stories, reels_watched, ads_clicked, posts, dan followers untuk memprediksi engagement.",
    )

    is_default_source = is_default_source_name(source_name)
    missing = validate_model_data(df)
    usable_rows = (
        len(df[MODEL_FEATURES + [TARGET]].replace([np.inf, -np.inf], np.nan).dropna())
        if not missing
        else 0
    )

    c1, c2, c3 = st.columns(3)
    if is_default_source:
        with c1:
            metric_card("Train test split", "80:20", "40,000 data train dan 10,000 data test")
        with c2:
            metric_card("Best R2", "0.89", "CatBoost regression")
        with c3:
            metric_card("Best RMSE", "0.13", "Error paling rendah di antara model pembanding")

        left, right = st.columns([1.2, 1])
        with left:
            results_sorted = STATIC_RESULTS.sort_values("R2", ascending=True)
            fig = px.bar(
                results_sorted,
                x="R2",
                y="Model",
                orientation="h",
                text="R2",
                color="Model",
                custom_data=["MAE", "RMSE"],
                color_discrete_sequence=[
                    SLATE_SOFT,
                    JADE_DARK,
                    "#0d9488",
                    "#14b8a6",
                    JADE,
                    JADE_LIGHT,
                    "#f7fff9",
                ],
                title="Model Comparison by R2",
            )
            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside",
                hovertemplate=(
                    "Model=%{y}<br>"
                    "R2=%{x:.2f}<br>"
                    "MAE=%{customdata[0]:.2f}<br>"
                    "RMSE=%{customdata[1]:.2f}<extra></extra>"
                ),
            )
            fig.update_layout(showlegend=False, xaxis_range=[0, 1])
            render_chart(fig, 500)

        with right:
            importance = STATIC_CATBOOST_IMPORTANCE.copy()
            importance["Feature Label"] = importance["Feature"].map(FEATURE_LABELS)
            fig = px.bar(
                importance.sort_values("Importance", ascending=True),
                x="Importance",
                y="Feature Label",
                orientation="h",
                color="Importance",
                color_continuous_scale=[SURFACE_2, JADE_DARK, JADE],
                labels={"Feature Label": "Fitur", "Importance": "Importance"},
                title="CatBoost Feature Importance",
            )
            fig.update_traces(
                hovertemplate="Fitur=%{y}<br>Importance=%{x:.2f}<extra></extra>"
            )
            render_chart(fig, 500)

        st.dataframe(
            STATIC_RESULTS.sort_values("R2", ascending=False).style.format(
                {"R2": "{:.2f}", "MAE": "{:.2f}", "RMSE": "{:.2f}"}
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        with c1:
            metric_card("Fitur model tersedia", f"{len(MODEL_FEATURES) - len(missing)}/5", "Jumlah fitur utama yang ditemukan")
        with c2:
            metric_card("Data siap evaluasi", f"{usable_rows:,}", "Baris lengkap untuk fitur dan target")
        with c3:
            metric_card("Evaluasi model", "Perlu retrain", "Metrik model tidak diasumsikan untuk dataset upload")

        if missing:
            soft_note("Kolom berikut belum tersedia untuk evaluasi model: " + ", ".join(missing))
        else:
            feature_check = df[MODEL_FEATURES + [TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
            corr_rows = [
                {
                    "Feature": FEATURE_LABELS.get(feature, feature),
                    "Correlation": feature_check[feature].corr(feature_check[TARGET]),
                    "Mean": feature_check[feature].mean(),
                }
                for feature in MODEL_FEATURES
            ]
            corr_df = pd.DataFrame(corr_rows).sort_values("Correlation")
            fig = px.bar(
                corr_df,
                x="Correlation",
                y="Feature",
                orientation="h",
                color="Correlation",
                custom_data=["Mean"],
                color_continuous_scale=[ROSE, PAGE_BG, JADE],
                title="Hubungan Fitur Model pada Dataset Upload",
            )
            fig.update_traces(
                hovertemplate=(
                    "Fitur=%{y}<br>"
                    "Korelasi=%{x:.2f}<br>"
                    "Rata-rata fitur=%{customdata[0]:.2f}<extra></extra>"
                )
            )
            fig.add_vline(x=0, line_width=1, line_color=CHART_LINE)
            render_chart(fig, 440)
            st.dataframe(
                corr_df.style.format({"Correlation": "{:.2f}", "Mean": "{:.2f}"}),
                width="stretch",
                hide_index=True,
            )

    vif_df = calculate_vif(df)
    if not vif_df.empty:
        section_title(
            "Feature Overlap Check",
            "VIF membaca apakah sebuah fitur membawa informasi yang tumpang tindih dengan fitur lain. Nilai tinggi berarti fitur itu sebaiknya dibaca bersama konteks fitur lain.",
        )
        vif_plot = vif_df.copy()
        vif_plot["Feature Label"] = vif_plot["Feature"].map(FEATURE_LABELS)
        vif_plot["VIF Label"] = vif_plot["VIF"].map(lambda value: f"{value:.2f}")
        fig = px.bar(
            vif_plot.sort_values("VIF", ascending=True),
            x="VIF",
            y="Feature Label",
            orientation="h",
            text="VIF Label",
            custom_data=["Feature Label", "VIF"],
            color_discrete_sequence=[JADE],
            labels={"Feature Label": "Fitur", "VIF": "VIF"},
            title="VIF Score",
            log_x=True,
        )
        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "Fitur=%{customdata[0]}<br>"
                "VIF=%{customdata[1]:.2f}<extra></extra>"
            ),
        )
        fig.update_layout(showlegend=False)
        max_vif = max(vif_plot["VIF"].max(), 10)
        fig.update_xaxes(
            title="VIF",
            tickmode="array",
            tickvals=[1, 10, 100, max_vif],
            ticktext=["1", "10", "100", format_decimal(max_vif, 2)],
            range=[0, np.log10(max_vif * 1.35)],
        )
        fig.add_vline(x=10, line_dash="dash", line_color=ROSE)
        render_chart(fig, 420)
        soft_note(
            "Patokan VIF 10 dipakai sebagai batas praktis: di atas itu, fitur cenderung membawa informasi yang tumpang tindih dengan fitur lain. Ini bukan error, tetapi tandanya interpretasi fitur perlu dibaca sebagai satu kelompok pola, bukan berdiri sendiri-sendiri."
        )


def render_creator_insight(df: pd.DataFrame) -> None:
    section_title(
        "Creator Insight",
        "Ringkasan kategori konten yang memiliki rata-rata engagement lebih tinggi pada dataset ini.",
    )

    left, right = st.columns(2)
    with left:
        if {"content_type_preference", TARGET}.issubset(df.columns):
            content_avg = (
                df.groupby("content_type_preference", dropna=False)[TARGET]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            fig = px.bar(
                content_avg,
                x="content_type_preference",
                y=TARGET,
                color=TARGET,
                color_continuous_scale=[SURFACE_2, JADE],
                labels={"content_type_preference": "Content type", TARGET: "Average engagement"},
                title="Average Engagement by Content Type",
            )
            fig.update_traces(
                hovertemplate=(
                    "Content type=%{x}<br>"
                    "Rata-rata engagement=%{y:.2f}<extra></extra>"
                )
            )
            render_chart(fig, 450)
            st.dataframe(
                content_avg.rename(
                    columns={
                        "content_type_preference": "Content Type",
                        TARGET: "Average Engagement Log",
                    }
                ).style.format({"Average Engagement Log": "{:.2f}"}),
                width="stretch",
                hide_index=True,
            )

    with right:
        if {"preferred_content_theme", TARGET}.issubset(df.columns):
            theme_avg = (
                df.groupby("preferred_content_theme", dropna=False)[TARGET]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            fig = px.bar(
                theme_avg,
                x="preferred_content_theme",
                y=TARGET,
                color=TARGET,
                color_continuous_scale=[SURFACE_2, JADE_DARK, JADE],
                labels={"preferred_content_theme": "Content theme", TARGET: "Average engagement"},
                title="Average Engagement by Preferred Theme",
            )
            fig.update_traces(
                hovertemplate=(
                    "Content theme=%{x}<br>"
                    "Rata-rata engagement=%{y:.2f}<extra></extra>"
                )
            )
            render_chart(fig, 450)
            st.dataframe(
                theme_avg.rename(
                    columns={
                        "preferred_content_theme": "Preferred Theme",
                        TARGET: "Average Engagement Log",
                    }
                ).style.format({"Average Engagement Log": "{:.2f}"}),
                width="stretch",
                hide_index=True,
            )

    st.write("")
    content_title, content_body = describe_category_average(
        df, "content_type_preference", "Content type"
    )
    theme_title, theme_body = describe_category_average(
        df, "preferred_content_theme", "Content theme"
    )
    i1, i2, i3 = st.columns(3)
    with i1:
        insight_card(content_title, content_body)
    with i2:
        insight_card(theme_title, theme_body)
    with i3:
        insight_card(
            "Baca sebagai konteks",
            "Insight kategori membantu melihat segmentasi audience. Untuk keputusan model, tetap cek korelasi dan feature importance.",
        )

    if {"subscription_status", TARGET}.issubset(df.columns):
        section_title("Subscription Context", "Distribusi subscription dipakai sebagai konteks audience.")
        sub_counts = df["subscription_status"].value_counts().reset_index()
        sub_counts.columns = ["Subscription", "Users"]
        fig = px.pie(
            sub_counts,
            names="Subscription",
            values="Users",
            hole=0.55,
            color_discrete_sequence=[JADE, JADE_LIGHT, JADE_DARK, "#34d399", "#14b8a6", SLATE_SOFT],
            title="Distribusi Subscription Status",
        )
        fig.update_traces(
            texttemplate="%{label}<br>%{percent:.2%}",
            textposition="outside",
            marker=dict(line=dict(color=PAGE_BG, width=2)),
            hovertemplate=(
                "Subscription=%{label}<br>"
                "User=%{value:,}<br>"
                "Share=%{percent:.2%}<extra></extra>"
            ),
        )
        render_chart(fig, 430)


st.sidebar.title("Instagram Engagement")
st.sidebar.caption("Dashboard visual untuk dataset ini.")

with st.sidebar.expander("Dataset source", expanded=False):
    st.caption(
        "Upload dipakai kalau dataset default belum ikut deploy atau ingin mengganti data. Dashboard menerima CSV, XLS, dan XLSX."
    )
    uploaded_file = st.file_uploader("Replace dataset", type=["csv", "xls", "xlsx"])

try:
    raw_df, source_name = read_dataset(uploaded_file)
except Exception as exc:
    st.error(f"Dataset gagal dibaca: {exc}")
    st.stop()

st.session_state["source_name"] = source_name

if raw_df is None:
    st.title("Instagram Engagement Modeling")
    soft_note(
        "Dataset default belum ditemukan di server deploy. Upload file CSV/XLS/XLSX lewat sidebar, atau pastikan df2_clean.csv / df2_clean.xls / df2_clean.xlsx ikut masuk ke repository deploy."
    )
    st.stop()

try:
    df_sample, df2, meta = prepare_data(raw_df, SAMPLE_SIZE)
except Exception as exc:
    st.error(f"Data gagal diproses: {exc}")
    st.stop()

filtered_df = apply_sidebar_filters(df2)

st.sidebar.divider()
st.sidebar.metric("Data shown", f"{len(filtered_df):,}")
st.sidebar.metric("Clean columns", f"{df2.shape[1]:,}")
st.sidebar.download_button(
    "Download cleaned data",
    data=df2.to_csv(index=False).encode("utf-8"),
    file_name="df2_clean_streamlit.csv",
    mime="text/csv",
    width="stretch",
)

if filtered_df.empty:
    st.warning("Tidak ada data setelah filter. Kurangi filter di sidebar.")
    st.stop()

tab_overview, tab_activity, tab_model, tab_creator = st.tabs(
    ["Overview", "Activity Patterns", "Model Evidence", "Creator Insight"]
)

with tab_overview:
    render_overview(filtered_df, df2, meta)

with tab_activity:
    render_activity_patterns(filtered_df)

with tab_model:
    render_model_evidence(df2, source_name)

with tab_creator:
    render_creator_insight(filtered_df)
