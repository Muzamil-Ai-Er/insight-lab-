"""Small presentation helpers for a consistent InsightLab interface."""

from __future__ import annotations

import html

import streamlit as st


def apply_theme(mode: str) -> None:
    """Inject a restrained product-style visual system for the selected appearance."""
    is_dark = mode == "Dark"
    background = "#0B1120" if is_dark else "#F5F7FB"
    surface = "#131C31" if is_dark else "#FFFFFF"
    surface_alt = "#1A2743" if is_dark else "#F8FAFC"
    text = "#E7EEF9" if is_dark else "#10213D"
    muted = "#AAB7CC" if is_dark else "#5F6F89"
    border = "#2A3A58" if is_dark else "#E2E8F0"
    st.markdown(
        f"""
        <style>
            :root {{ --il-bg: {background}; --il-surface: {surface}; --il-alt: {surface_alt}; --il-text: {text}; --il-muted: {muted}; --il-border: {border}; --il-brand: #5B5CE2; --il-accent: #17B6A4; }}
            [data-testid="stAppViewContainer"] {{ background: var(--il-bg); }}
            [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
            [data-testid="stSidebar"] > div:first-child {{ background: var(--il-surface); border-right: 1px solid var(--il-border); }}
            h1, h2, h3 {{ color: var(--il-text); letter-spacing: -0.025em; }}
            p, label, [data-testid="stMarkdownContainer"] {{ color: var(--il-text); }}
            .il-hero {{
                background: linear-gradient(120deg, #4038B8 0%, #5B5CE2 52%, #1DAF9F 120%);
                color: #fff; border-radius: 20px; padding: 2.1rem 2.2rem 1.9rem;
                margin: 0.25rem 0 1.5rem; box-shadow: 0 14px 34px rgba(50, 52, 150, .22);
            }}
            .il-hero h1, .il-hero p {{ color: #fff !important; margin: 0; }}
            .il-hero p {{ color: rgba(255,255,255,.86) !important; margin-top: .55rem; max-width: 46rem; font-size: 1.05rem; }}
            .il-card {{ background: var(--il-surface); border: 1px solid var(--il-border); border-radius: 15px; padding: 1.05rem 1.15rem; min-height: 106px; }}
            .il-card-label {{ color: var(--il-muted) !important; font-size: .77rem; font-weight: 700; letter-spacing: .075em; text-transform: uppercase; margin: 0 0 .35rem; }}
            .il-card-value {{ color: var(--il-text) !important; font-size: 1.5rem; font-weight: 750; margin: 0; }}
            .il-card-subtitle {{ color: var(--il-muted) !important; font-size: .84rem; margin: .25rem 0 0; }}
            .il-section-kicker {{ color: var(--il-brand) !important; font-size: .74rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; margin-bottom: .15rem; }}
            .il-section-title {{ color: var(--il-text) !important; font-size: 1.5rem; font-weight: 750; margin: 0 0 .25rem; }}
            .il-section-copy {{ color: var(--il-muted) !important; margin: 0 0 1rem; }}
            [data-testid="stMetric"] {{ background: var(--il-surface); border: 1px solid var(--il-border); border-radius: 14px; padding: .8rem .95rem; }}
            [data-testid="stMetricLabel"] {{ color: var(--il-muted); }}
            [data-testid="stMetricValue"] {{ color: var(--il-text); }}
            [data-testid="stDataFrame"] {{ border: 1px solid var(--il-border); border-radius: 12px; overflow: hidden; }}
            .stButton > button, .stDownloadButton > button {{ border-radius: 9px; font-weight: 650; }}
            [data-testid="stCaptionContainer"] {{ color: var(--il-muted); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, copy: str) -> None:
    """Display the product header."""
    st.markdown(
        f'<div class="il-hero"><h1>{html.escape(title)}</h1><p>{html.escape(copy)}</p></div>',
        unsafe_allow_html=True,
    )


def section(title: str, copy: str, kicker: str = "Workspace") -> None:
    """Display a compact page heading."""
    st.markdown(
        f'<p class="il-section-kicker">{html.escape(kicker)}</p>'
        f'<p class="il-section-title">{html.escape(title)}</p>'
        f'<p class="il-section-copy">{html.escape(copy)}</p>',
        unsafe_allow_html=True,
    )


def card(label: str, value: str, subtitle: str = "") -> None:
    """Display a lightweight informational card inside a Streamlit column."""
    st.markdown(
        f'<div class="il-card"><p class="il-card-label">{html.escape(label)}</p>'
        f'<p class="il-card-value">{html.escape(value)}</p>'
        f'<p class="il-card-subtitle">{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, copy: str) -> None:
    """Use a friendly empty state instead of a silent blank page."""
    st.info(f"**{title}**\n\n{copy}")

