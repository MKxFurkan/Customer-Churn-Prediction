"""Churn Insight - customer churn prediction frontend (Streamlit).

Pages (custom top navigation driven by ``st.session_state``):
    * Home          - landing page
    * Predict       - customer assessment form + prediction
    * Model Details - what the trained KNN model is and how it was evaluated

Artifacts used (never modified or refitted here):
    * model.pkl   - tuned KNeighborsClassifier
    * scaler.pkl  - StandardScaler fitted on the 800 training rows
    * customer_churn_data.csv - only used to reproduce the held-out evaluation
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
DATA_PATH = BASE_DIR / "customer_churn_data.csv"

# Autumn Luxe palette (the CSS mirrors these as custom properties).
C_BG = "#E2E1EB"
C_GREY = "#AAAAAE"
C_TAUPE = "#7F7265"
C_AMBER = "#BF8440"
C_INK = "#322D27"

# The exact column order the model and scaler were trained on (notebook cell 49).
FEATURE_ORDER = [
    "Age",
    "Gender",
    "Tenure",
    "MonthlyCharges",
    "TotalCharges",
    "TechSupport",
    "ContractType_Month-to-Month",
    "ContractType_One-Year",
    "ContractType_Two-Year",
    "InternetService_",
    "InternetService_DSL",
    "InternetService_Fiber Optic",
]
RAW_COLUMNS = [
    "Age",
    "Gender",
    "Tenure",
    "MonthlyCharges",
    "ContractType",
    "InternetService",
    "TotalCharges",
    "TechSupport",
]

# LabelEncoder sorts alphabetically: Female=0 / Male=1 and No=0 / Yes=1.
GENDER_CODES = {"Female": 0, "Male": 1}
SUPPORT_CODES = {"No": 0, "Yes": 1}

GENDERS = list(GENDER_CODES)
CONTRACTS = ["Month-to-Month", "One-Year", "Two-Year"]
INTERNET_OPTIONS = ["Fiber Optic", "DSL", "No Internet Service"]
# Missing InternetService was filled with "" in the notebook -> column "InternetService_".
INTERNET_TO_CATEGORY = {"Fiber Optic": "Fiber Optic", "DSL": "DSL", "No Internet Service": ""}
SUPPORT_OPTIONS = list(SUPPORT_CODES)

# Ranges observed in customer_churn_data.csv (used for soft "outside training data" warnings).
TRAIN_RANGES = {
    "age": (12, 83, "Age"),
    "tenure": (0, 122, "Tenure"),
    "monthly": (30.0, 119.96, "Monthly charges"),
}

PAGES = {"home": "Home", "predict": "Predict", "model": "Model Details"}

DEFAULTS = {
    "in_age": 45,
    "in_gender": "Male",
    "in_tenure": 12,
    "in_monthly": 70.00,
    "in_contract": "One-Year",
    "in_internet": "DSL",
    "in_support": "No",
}

# Values transcribed from notebook.ipynb outputs (cells 38, 40 and 45).
NOTEBOOK_MODEL_COMPARISON = [
    ("K-Nearest Neighbors", 0.970),
    ("Logistic Regression", 0.965),
    ("Support Vector Machine", 0.965),
    ("Naive Bayes", 0.870),
]
NOTEBOOK_CV_BEST_F1 = 0.9782300078752544  # GridSearchCV, cv=5, scoring="f1", training data only
# Fallback only used if customer_churn_data.csv cannot be found (cells 47 of the notebook).
NOTEBOOK_REPORT = {
    "0": {"precision": 1.00, "recall": 0.78, "f1-score": 0.88, "support": 23},
    "1": {"precision": 0.97, "recall": 1.00, "f1-score": 0.99, "support": 177},
    "accuracy": 0.97,
    "macro avg": {"precision": 0.99, "recall": 0.89, "f1-score": 0.93, "support": 200},
    "weighted avg": {"precision": 0.98, "recall": 0.97, "f1-score": 0.97, "support": 200},
}

HEADLINE_ACCURACY = "97%"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Pontano+Sans:wght@300..700&family=Stint+Ultra+Expanded&display=swap');"
)

BASE_CSS = """
:root{
  --bg:#E2E1EB; --grey:#AAAAAE; --taupe:#7F7265; --amber:#BF8440; --ink:#322D27;
  /* light tints derived from the palette */
  --card:#F1F0F6; --field:#F8F7FB; --soft:#5C5349; --line:rgba(127,114,101,.38);
  --amber-tint:rgba(191,132,64,.18); --ink-tint:rgba(50,45,39,.08);
  --display:'Stint Ultra Expanded','Georgia',serif;
  --body:'Pontano Sans','Helvetica Neue',Arial,sans-serif;
}
html, body, .stApp{ background:var(--bg) !important; color:var(--ink); font-family:var(--body); }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp textarea,
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] [role="option"],
[data-baseweb="popover"] li{ font-family:var(--body); }
[data-testid="stHeader"]{ background:transparent !important; }
[data-testid="stMainBlockContainer"], .block-container{
  max-width:1180px; padding:2.4rem 2rem 4rem !important; }
.stApp p{ color:var(--ink); }
a{ color:var(--amber); }
:focus-visible{ outline:3px solid var(--amber); outline-offset:2px; }

/* ---------- typography helpers ---------- */
[class*="ci-"] :where(span, b, small, em, code, li, dt, dd, td, th, div, p){ font-family:var(--body); }
.ci-card-title, .ci-note b, .ci-brand-name{ font-family:var(--display) !important; font-weight:400 !important; }
/* figures use the body face: Stint Ultra Expanded draws 0 like the letter O */
.ci-card .ci-num, .ci-big, .ci-prob-value, .ci-stat b, .ci-cm .cell b{
  font-family:var(--body) !important; font-weight:700 !important; letter-spacing:-.01em; font-variant-numeric:tabular-nums; }
.ci-display, .ci-h1, .ci-h2, .ci-h3{
  font-family:var(--display) !important; font-weight:400 !important; color:var(--ink);
  line-height:1.18; margin:0; letter-spacing:0; overflow-wrap:anywhere; }
/* Streamlit wraps heading text in an inner span: make it inherit the heading face */
.ci-display *, .ci-h1 *, .ci-h2 *, .ci-h3 *{ font-family:inherit !important; font-weight:inherit !important; }
.ci-h1{ font-size:clamp(1.55rem, 3.3vw, 2.7rem); }
.ci-h2{ font-size:clamp(1.2rem, 2.2vw, 1.7rem); }
.ci-h3{ font-size:1.02rem; line-height:1.3; }
.ci-eyebrow{ font-family:var(--body); font-weight:700; font-size:.78rem; letter-spacing:.18em;
  text-transform:uppercase; color:var(--soft); margin:0 0 1rem; display:flex; align-items:center; gap:.6rem; }
.ci-eyebrow::before{ content:""; width:28px; height:2px; background:var(--amber); display:inline-block; }
.ci-lead{ font-size:1.08rem; line-height:1.65; color:var(--soft); max-width:60ch; margin:1.1rem 0 0; }
.ci-muted{ color:var(--soft); font-size:.92rem; line-height:1.6; }
.ci-section{ margin:4.2rem 0 1.6rem; }
.ci-section .ci-lead{ margin-top:.8rem; }

/* ---------- navigation bar (columns + buttons, not st.navigation) ---------- */
.st-key-navbar{
  background:rgba(248,247,251,.72); border:1px solid var(--grey); border-radius:999px;
  padding:.5rem .6rem .5rem 1.5rem; margin-bottom:3rem; }
.st-key-navbar [data-testid="stHorizontalBlock"]{ align-items:center; gap:.5rem; }
.st-key-navbar [data-testid="stMarkdownContainer"]{ margin-bottom:0 !important; }
.ci-brand{ display:flex; align-items:center; gap:.7rem; min-height:2.6rem; white-space:nowrap; }
.ci-brand-mark{ width:1.55rem; height:1.55rem; border-radius:50%; background:var(--ink); position:relative; flex:none; }
.ci-brand-mark::after{ content:""; position:absolute; width:.62rem; height:.62rem; border-radius:50%;
  background:var(--amber); right:.18rem; top:.18rem; }
.ci-brand-name{ font-family:var(--display) !important; font-size:.98rem; color:var(--ink); line-height:1; }
.st-key-navbar button{
  background:transparent !important; border:1px solid transparent !important; color:var(--ink) !important;
  border-radius:999px !important; padding:.5rem 1.15rem !important; min-height:2.6rem !important;
  height:auto !important; white-space:nowrap !important; overflow:visible !important;
  transition:background .18s ease, border-color .18s ease, transform .18s ease; box-shadow:none !important; }
.st-key-navbar button p, .st-key-navbar button div, .st-key-navbar button span{
  color:inherit !important; font-family:var(--body) !important; font-weight:700 !important;
  font-size:.95rem !important; line-height:1.25 !important; margin:0 !important; white-space:nowrap !important;
  overflow:visible !important; opacity:1 !important; }
.st-key-navbar button:hover{ background:var(--ink-tint) !important; border-color:var(--grey) !important; transform:translateY(-1px); }
.st-key-navbar button:focus-visible{ outline:3px solid var(--amber) !important; outline-offset:2px !important; }

/* ---------- generic buttons ---------- */
.stButton, [data-testid="stButton"]{ width:100%; }
.stButton > button, [data-testid="stButton"] > button{
  width:100%; border-radius:999px; font-weight:700; transition:all .18s ease; min-height:2.9rem; height:auto; }
.stButton > button p, [data-testid="stButton"] > button p{
  font-family:var(--body) !important; font-weight:700 !important; font-size:1rem !important; margin:0; line-height:1.25; color:inherit; }
.st-key-cta button, .st-key-predict_btn button{
  background:var(--amber) !important; color:var(--ink) !important; border:1px solid var(--amber) !important;
  padding:.7rem 1.6rem !important; box-shadow:0 12px 24px -14px rgba(50,45,39,.7) !important; }
.st-key-cta button:hover, .st-key-predict_btn button:hover{
  background:var(--ink) !important; color:var(--bg) !important; border-color:var(--ink) !important; transform:translateY(-2px); }
.st-key-reset_btn button{
  background:transparent !important; color:var(--ink) !important; border:1.5px solid var(--taupe) !important; padding:.7rem 1.4rem !important; }
.st-key-reset_btn button:hover{ background:var(--ink-tint) !important; border-color:var(--ink) !important; transform:translateY(-1px); }
.st-key-cta{ margin-top:1.8rem; max-width:320px; }

/* ---------- cards ---------- */
.ci-card{ background:var(--card); border:1px solid var(--line); border-radius:22px; padding:1.6rem;
  box-shadow:0 18px 38px -28px rgba(50,45,39,.55); height:100%; transition:transform .2s ease, box-shadow .2s ease; }
.ci-card:hover{ transform:translateY(-3px); box-shadow:0 24px 44px -26px rgba(50,45,39,.6); }
.ci-card .ci-num{ font-size:1.6rem; color:var(--amber); display:block; margin-bottom:1.1rem; }
.ci-card p{ margin:.7rem 0 0; color:var(--soft); line-height:1.6; font-size:.98rem; }
.ci-grid{ display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:1.2rem; }
.ci-split{ display:grid; grid-template-columns:minmax(0,1.25fr) minmax(0,1fr); gap:1.2rem; align-items:stretch; }
.ci-grid-2{ display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:1.2rem; }

.ci-hero-art{ background:var(--card); border:1px solid var(--line); border-radius:28px; padding:1.1rem 1.1rem .9rem;
  box-shadow:0 24px 50px -32px rgba(50,45,39,.6); }
.ci-hero-art svg{ width:100%; height:auto; display:block; }
.ci-legend{ display:flex; flex-wrap:wrap; gap:.4rem 1.2rem; padding:.4rem .4rem 0; font-size:.85rem; color:var(--soft); }
.ci-legend i{ display:inline-block; width:.8rem; height:.8rem; border-radius:50%; margin-right:.45rem; vertical-align:-1px; }

.ci-band{ background:var(--ink); color:var(--bg); border-radius:28px; padding:2.2rem 2.2rem 2rem; }
.ci-band .ci-eyebrow{ color:var(--grey); }
.ci-band .ci-h2{ color:var(--bg); }
.ci-stats{ display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:1.6rem; margin-top:1.8rem; }
.ci-stat b{ display:block; font-size:clamp(1.7rem, 3vw, 2.4rem); color:var(--amber); line-height:1.2; }
.ci-stat span{ display:block; color:var(--bg); font-size:.95rem; margin-top:.45rem; line-height:1.45; }
.ci-stat small{ display:block; color:var(--grey); font-size:.82rem; margin-top:.15rem; }

/* ---------- form cards (st.container(key=...)) ---------- */
.st-key-card_personal, .st-key-card_billing{
  background:var(--card); border:1px solid var(--line); border-radius:22px; padding:1.6rem 1.6rem 1.3rem;
  box-shadow:0 18px 38px -28px rgba(50,45,39,.55); }
.ci-card-title{ font-family:var(--display); font-size:1.02rem; line-height:1.3; color:var(--ink); margin:0; }
.ci-card-sub{ color:var(--soft); font-size:.92rem; margin:.35rem 0 .4rem; line-height:1.5; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label{
  font-family:var(--body) !important; font-weight:700 !important; font-size:.9rem !important; color:var(--ink) !important; }

/* number inputs (new react-aria DOM + older BaseWeb DOM) */
[data-testid="stNumberInputContainer"], [data-baseweb="input"]{
  background:var(--field) !important; border:1px solid var(--grey) !important; border-radius:12px !important;
  overflow:hidden; transition:border-color .15s, box-shadow .15s; }
[data-baseweb="base-input"]{ background:transparent !important; }
[data-testid="stNumberInputField"], [data-baseweb="input"] input{
  background:transparent !important; color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important;
  font-size:1rem !important; font-family:var(--body) !important; }
[data-testid="stNumberInputContainer"]:hover, [data-baseweb="input"]:hover{ border-color:var(--taupe) !important; }
[data-testid="stNumberInputContainer"]:focus-within, [data-baseweb="input"]:focus-within{
  border-color:var(--amber) !important; box-shadow:0 0 0 3px rgba(191,132,64,.35) !important; }
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"]{
  background:var(--bg) !important; color:var(--ink) !important; border:0 !important; }
[data-testid="stNumberInputStepUp"]:hover, [data-testid="stNumberInputStepDown"]:hover{
  background:var(--amber) !important; color:var(--ink) !important; }
[data-testid="stNumberInputStepUp"] svg, [data-testid="stNumberInputStepDown"] svg{ fill:currentColor !important; }

/* select boxes */
[data-testid="stSelectbox"] [role="group"], [data-baseweb="select"] > div{
  background:var(--field) !important; border:1px solid var(--grey) !important; border-radius:12px !important;
  min-height:2.75rem; color:var(--ink) !important; transition:border-color .15s, box-shadow .15s; }
[data-testid="stSelectbox"] [role="group"]:hover, [data-baseweb="select"] > div:hover{ border-color:var(--taupe) !important; }
[data-testid="stSelectbox"] [role="group"]:focus-within, [data-baseweb="select"] > div:focus-within{
  border-color:var(--amber) !important; box-shadow:0 0 0 3px rgba(191,132,64,.35) !important; }
[data-testid="stSelectbox"] input{ background:transparent !important; color:var(--ink) !important;
  -webkit-text-fill-color:var(--ink) !important; font-family:var(--body) !important; font-size:1rem !important; }
[data-testid="stSelectbox"] button{ background:transparent !important; color:var(--ink) !important; border:0 !important; box-shadow:none !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"] > div{ background:transparent !important; }
[data-baseweb="select"] *{ color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; font-family:var(--body) !important; }
[data-baseweb="select"] svg, [data-testid="stSelectbox"] svg{ fill:var(--ink) !important; }

/* dropdown menu (rendered in a portal outside the page content) */
[data-testid="stSelectboxVirtualDropdown"], [data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul[role="listbox"]{
  background:var(--field) !important; border:1px solid var(--grey) !important; border-radius:12px !important;
  box-shadow:0 18px 36px -18px rgba(50,45,39,.55) !important; overflow:hidden; }
[data-testid="stSelectboxVirtualDropdown"] [role="listbox"]{ background:transparent !important; }
[data-baseweb="popover"], [data-baseweb="popover"] > div{ background:transparent !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"], [data-baseweb="popover"] li{
  background:transparent !important; color:var(--ink) !important; font-size:.98rem; cursor:pointer; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"] *, [data-baseweb="popover"] li *{ color:var(--ink) !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover, [data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused],
[data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"],
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"]{ background:var(--amber-tint) !important; }

/* ---------- notes, result, meter ---------- */
.ci-note{ display:flex; gap:.8rem; align-items:flex-start; border-radius:14px; padding:.85rem 1.1rem;
  font-size:.95rem; line-height:1.5; margin:1rem 0; border:1px solid var(--taupe); color:var(--ink); background:rgba(255,255,255,.5); }
.ci-note b{ font-family:var(--display); font-weight:400; flex:none; width:1.5rem; height:1.5rem; border-radius:50%;
  display:inline-flex; align-items:center; justify-content:center; font-size:.8rem; background:var(--amber); color:var(--ink); }
.ci-note strong{ font-weight:700; }
.ci-note.error{ background:var(--ink); border-color:var(--ink); color:var(--bg); }
.ci-note.error b{ background:var(--amber); }

.ci-result{ border-radius:26px; padding:2rem 2.1rem; margin-top:1.6rem; }
.ci-result.churn{ background:var(--ink); color:var(--bg); border:1px solid var(--ink); }
.ci-result.stay{ background:var(--card); color:var(--ink); border:2px solid var(--amber); }
.ci-badge{ display:inline-flex; align-items:center; gap:.5rem; padding:.35rem .9rem; border-radius:999px; font-weight:700;
  font-size:.8rem; letter-spacing:.12em; text-transform:uppercase; }
.churn .ci-badge{ background:var(--amber); color:var(--ink); }
.stay .ci-badge{ background:var(--ink); color:var(--bg); }
.ci-result .ci-h1{ margin:1rem 0 .3rem; font-size:clamp(1.6rem, 3.4vw, 2.5rem); }
.churn .ci-h1{ color:var(--bg); }
.ci-result p{ margin:.4rem 0 0; line-height:1.6; }
.churn p{ color:var(--bg) !important; } .stay p{ color:var(--soft) !important; }
.ci-prob-row{ display:flex; justify-content:space-between; align-items:baseline; margin-top:1.6rem; gap:1rem; flex-wrap:wrap; }
.ci-prob-label{ font-weight:700; font-size:.9rem; letter-spacing:.04em; }
.ci-prob-value{ font-size:2rem; color:var(--amber); }
.stay .ci-prob-value{ color:var(--ink); }
.ci-meter{ height:14px; border-radius:999px; margin-top:.7rem; overflow:hidden; background:rgba(170,170,174,.45); }
.churn .ci-meter{ background:rgba(226,225,235,.2); }
.ci-meter-fill{ height:100%; border-radius:999px; background:var(--amber); }
.ci-ticks{ display:flex; justify-content:space-between; font-size:.78rem; margin-top:.4rem; opacity:.85; }
.ci-disclaimer{ font-size:.86rem !important; margin-top:1.4rem !important; opacity:.92; }

/* ---------- tables, confusion matrix, bars ---------- */
.ci-table{ width:100%; border-collapse:collapse; font-size:.95rem; }
.ci-table th{ text-align:right; font-weight:700; color:var(--soft); font-size:.78rem; letter-spacing:.08em; text-transform:uppercase;
  padding:.55rem .6rem; border-bottom:2px solid var(--ink); }
.ci-table td{ text-align:right; padding:.7rem .6rem; border-bottom:1px solid var(--line); color:var(--ink); font-variant-numeric:tabular-nums; }
.ci-table th:first-child, .ci-table td:first-child{ text-align:left; }
.ci-table tr.sum td{ background:var(--ink-tint); font-weight:700; }
.ci-table-wrap{ overflow-x:auto; }
.ci-kv{ display:grid; grid-template-columns:1fr auto; gap:0; margin:0; }
.ci-kv dt, .ci-kv dd{ margin:0; padding:.7rem 0; border-bottom:1px solid var(--line); }
.ci-kv dt{ color:var(--soft); }
.ci-kv dd{ text-align:right; font-weight:700; font-variant-numeric:tabular-nums; }
.ci-kv dd.hl{ color:var(--ink); }
.ci-kv dd.hl span{ background:var(--amber-tint); border-radius:8px; padding:.15rem .6rem; }
.ci-cm{ display:grid; grid-template-columns:auto 1fr 1fr; gap:.5rem; align-items:stretch; margin-top:1rem; }
.ci-cm .h{ font-size:.78rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--soft);
  display:flex; align-items:center; justify-content:center; text-align:center; padding:.3rem; }
.ci-cm .cell{ border-radius:14px; padding:1.1rem .5rem; text-align:center; border:1px solid var(--line); background:var(--field); }
.ci-cm .cell b{ display:block; font-size:1.9rem; color:var(--ink); }
.ci-cm .cell small{ display:block; font-size:.78rem; color:var(--soft); margin-top:.25rem; }
.ci-cm .cell.ok{ background:var(--amber); border-color:var(--amber); }
.ci-cm .cell.ok small{ color:var(--ink); }
.ci-bar-row{ display:grid; grid-template-columns:minmax(120px,190px) 1fr 3.4rem; gap:.9rem; align-items:center; margin:.75rem 0; font-size:.95rem; }
.ci-bar{ height:14px; border-radius:999px; background:rgba(170,170,174,.5); overflow:hidden; }
.ci-bar i{ display:block; height:100%; border-radius:999px; background:var(--taupe); }
.ci-bar-row.top .ci-bar i{ background:var(--amber); }
.ci-bar-row b{ text-align:right; font-variant-numeric:tabular-nums; }
.ci-tag{ display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; padding:.28rem .7rem;
  border-radius:999px; margin-bottom:.9rem; }
.ci-tag.test{ background:var(--amber); color:var(--ink); }
.ci-tag.cv{ background:transparent; color:var(--ink); border:1.5px solid var(--taupe); }
.ci-big{ font-size:clamp(1.8rem,3.2vw,2.6rem); color:var(--ink); line-height:1.15; }
.ci-chips{ display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1rem; }
.ci-chip{ background:var(--field); border:1px solid var(--line); border-radius:999px; padding:.3rem .8rem; font-size:.84rem; color:var(--ink); }
.ci-chip em{ font-style:normal; color:var(--amber); font-weight:700; margin-right:.35rem; }
.ci-steps{ list-style:none; counter-reset:s; margin:0; padding:0; }
.ci-steps li{ counter-increment:s; display:grid; grid-template-columns:2.6rem 1fr; gap:1rem; padding:1rem 0; border-bottom:1px solid var(--line); line-height:1.6; }
.ci-steps li::before{ content:counter(s); width:2.2rem; height:2.2rem; border-radius:50%; background:var(--ink); color:var(--amber);
  display:flex; align-items:center; justify-content:center; font-family:var(--display); font-size:.85rem; }
.ci-steps li b{ display:block; color:var(--ink); }
.ci-steps li span{ color:var(--soft); font-size:.95rem; }
[class*="ci-"] code{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.85em;
  color:var(--ink); background:var(--ink-tint); border-radius:6px; padding:.1rem .4rem; }
.ci-footer{ margin-top:4rem; padding-top:1.4rem; border-top:1px solid var(--line); color:var(--soft); font-size:.86rem; line-height:1.6; }

[data-testid="stExpander"]{ margin-top:1.1rem; border:1px solid var(--line) !important; border-radius:16px !important; background:var(--card); }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p{ color:var(--ink) !important; font-weight:700; }
[data-testid="stExpander"] details{ border:0 !important; }

/* ---------- responsive ---------- */
@media (max-width: 900px){
  .ci-split{ grid-template-columns:1fr; }
  [data-testid="stMainBlockContainer"], .block-container{ padding:1.4rem 1rem 3rem !important; }
  .st-key-navbar{ border-radius:28px; padding:.7rem 1rem; }
  .st-key-navbar [data-testid="stHorizontalBlock"]{ flex-wrap:wrap !important; }
  .st-key-navbar [data-testid="stColumn"]{ min-width:0 !important; flex:1 1 auto !important; width:auto !important; }
  .st-key-navbar [data-testid="stColumn"]:first-child{ flex:1 1 100% !important; }
  .st-key-navbar button{ padding:.5rem .8rem !important; }
  .ci-band{ padding:1.6rem 1.3rem; }
  .ci-result{ padding:1.5rem 1.3rem; }
  .ci-bar-row{ grid-template-columns:1fr 3.4rem; }
  .ci-bar-row .ci-bar{ grid-column:1 / -1; order:3; }
}
@media (prefers-reduced-motion: reduce){ *{ transition:none !important; } }
"""


def collapse(markup: str) -> str:
    """Flatten indented HTML so Markdown never mistakes it for a code block."""
    return " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())


def html(markup: str) -> None:
    st.markdown(collapse(markup), unsafe_allow_html=True)


def inject_css(active_page: str) -> None:
    """Global CSS plus the active-page rule for the custom navigation bar."""
    css = re.sub(r"\n\s*\n", "\n", BASE_CSS)
    active_rule = (
        f".st-key-nav_{active_page} button{{background:{C_AMBER} !important;"
        f"border-color:{C_AMBER} !important;color:{C_INK} !important;"
        f"box-shadow:0 8px 18px -12px rgba(50,45,39,.8) !important;}}"
        f".st-key-nav_{active_page} button:hover{{background:{C_AMBER} !important;"
        f"border-color:{C_INK} !important;}}"
    )
    # Static block first (identical on every rerun, so the browser keeps the loaded fonts);
    # only the small active-page rule below changes between pages.
    st.markdown(f"<style>{FONT_IMPORT}\n{css}</style>", unsafe_allow_html=True)
    st.markdown(f"<style>{active_rule}</style>", unsafe_allow_html=True)


def stretch_button(label: str, **kwargs) -> bool:
    """st.button that fills its column across Streamlit versions."""
    try:
        return st.button(label, width="stretch", **kwargs)
    except TypeError:  # older Streamlit without the ``width`` argument
        return st.button(label, use_container_width=True, **kwargs)


def note(kind: str, text: str) -> None:
    icon = "!" if kind == "error" else "i"
    html(f'<div class="ci-note {kind}"><b>{icon}</b><div>{text}</div></div>')


# ---------------------------------------------------------------------------
# Navigation (st.session_state + on_click callbacks)
# ---------------------------------------------------------------------------


def init_state() -> None:
    st.session_state.setdefault("page", "home")
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def go_to(page: str) -> None:
    """Callback: runs before the next script run, so the new page renders immediately."""
    st.session_state["page"] = page


def reset_form() -> None:
    """Callback: widget keys may be changed here because it runs before widgets are drawn."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.pop("result", None)


def render_navbar() -> None:
    current = st.session_state["page"]
    with st.container(key="navbar"):
        brand, _, c_home, c_predict, c_model = st.columns(
            [3.2, 2.2, 1.1, 1.2, 1.7], vertical_alignment="center"
        )
        with brand:
            html(
                '<div class="ci-brand"><span class="ci-brand-mark"></span>'
                '<span class="ci-brand-name">Churn Insight</span></div>'
            )
        for column, page in ((c_home, "home"), (c_predict, "predict"), (c_model, "model")):
            with column:
                stretch_button(
                    PAGES[page],
                    key=f"nav_{page}",
                    on_click=go_to,
                    args=(page,),
                    help=None,
                    type="primary" if page == current else "secondary",
                )


# ---------------------------------------------------------------------------
# Model, preprocessing, prediction
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def load_artifacts():
    """Return (model, scaler, error_message). Never raises."""
    missing = [p.name for p in (MODEL_PATH, SCALER_PATH) if not p.exists()]
    if missing:
        return None, None, f"Missing file(s): {', '.join(missing)}. Place them next to app.py."
    try:
        model, scaler = _load_artifacts()
    except Exception as exc:  # corrupted pickle, incompatible scikit-learn, ...
        return None, None, f"The model files could not be loaded ({type(exc).__name__}: {exc})."
    if getattr(model, "n_features_in_", len(FEATURE_ORDER)) != len(FEATURE_ORDER):
        return None, None, "The saved model does not expect 12 features; refusing to predict."
    return model, scaler, None


def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook preprocessing and return the 12 model features in order.

    Notebook steps (cells 6, 26, 28, 29): fillna("") on InternetService, LabelEncoder on
    Gender and TechSupport, pd.get_dummies on ContractType / InternetService and finally
    ``astype(int)``. The cast truncates MonthlyCharges and TotalCharges to whole numbers;
    the saved scaler was fitted on those truncated values, so it is reproduced here.
    """
    df = raw[RAW_COLUMNS].copy()
    df["Gender"] = df["Gender"].map(GENDER_CODES)
    df["TechSupport"] = df["TechSupport"].map(SUPPORT_CODES)
    df["InternetService"] = df["InternetService"].fillna("")
    df = pd.get_dummies(df, columns=["ContractType", "InternetService"])
    df = df.reindex(columns=FEATURE_ORDER, fill_value=0)  # absent dummies -> 0, order enforced
    return df.astype(int)


def raw_row(values: dict) -> pd.DataFrame:
    # TotalCharges is not asked for on the form; it is derived from tenure and monthly charges.
    # Rounded to 2 decimals first so float noise (e.g. 28.999999999999996) cannot change the
    # value; build_feature_frame() then applies the notebook's astype(int) truncation.
    total_charges = round(values["in_tenure"] * values["in_monthly"], 2)
    return pd.DataFrame(
        [
            {
                "Age": values["in_age"],
                "Gender": values["in_gender"],
                "Tenure": values["in_tenure"],
                "MonthlyCharges": values["in_monthly"],
                "ContractType": values["in_contract"],
                "InternetService": INTERNET_TO_CATEGORY[values["in_internet"]],
                "TotalCharges": total_charges,
                "TechSupport": values["in_support"],
            }
        ]
    )


def predict_churn(model, scaler, features: pd.DataFrame) -> dict:
    """Scale with the already-fitted scaler (transform only) and query the KNN model."""
    scaled = scaler.transform(features)  # never fit / fit_transform on user input
    label = int(model.predict(scaled)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled)[0]
        probability = float(proba[list(model.classes_).index(1)])
    return {"label": label, "probability": probability, "scaled": scaled[0], "features": features.iloc[0]}


def validate_inputs(values: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    labels = {
        "in_age": "Age",
        "in_tenure": "Tenure",
        "in_monthly": "Monthly charges",
    }
    for key, label in labels.items():
        if values.get(key) is None:
            errors.append(f"{label} is required.")
    if errors:
        return errors, warnings

    if values["in_monthly"] <= 0:
        errors.append("Monthly charges must be greater than 0.")
    if values["in_gender"] not in GENDER_CODES or values["in_support"] not in SUPPORT_CODES:
        errors.append("Choose a valid gender and tech support option.")
    if values["in_contract"] not in CONTRACTS or values["in_internet"] not in INTERNET_OPTIONS:
        errors.append("Choose a valid contract type and internet service.")
    if errors:
        return errors, warnings

    for key, (low, high, label) in (
        ("in_age", TRAIN_RANGES["age"]),
        ("in_tenure", TRAIN_RANGES["tenure"]),
        ("in_monthly", TRAIN_RANGES["monthly"]),
    ):
        if not low <= values[key] <= high:
            warnings.append(
                f"{label} is outside the range seen in the training data ({low:g} to {high:,g}); "
                "the estimate may be less reliable."
            )
    return errors, warnings


# ---------------------------------------------------------------------------
# Held-out evaluation (reproduced from the notebook; cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def compute_evaluation(_model, _scaler, data_path: str, mtime: float):
    """Re-create the notebook's 80/20 stratified split and score the saved model on the test part."""
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(data_path)
    y = (df["Churn"] == "Yes").astype(int).to_numpy()
    X = build_feature_frame(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pred = _model.predict(_scaler.transform(X_test))
    mtm = df["ContractType"] == "Month-to-Month"
    return {
        "report": classification_report(y_test, pred, output_dict=True),
        "matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "n_total": int(len(df)),
        "n_churn": int(y.sum()),
        "n_mtm": int(mtm.sum()),
        "n_mtm_churn": int(y[mtm.to_numpy()].sum()),
        "n_test": int(len(y_test)),
    }


def get_evaluation(model, scaler):
    if not DATA_PATH.exists():
        return None
    try:
        return compute_evaluation(model, scaler, str(DATA_PATH), DATA_PATH.stat().st_mtime)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Illustration for the landing page
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def neighbours_svg() -> str:
    """Decorative k-nearest-neighbours picture: a new point and the 7 closest points around it."""
    rng = np.random.RandomState(11)
    width, height = 460, 380
    query = np.array([236.0, 188.0])
    pts = []
    while len(pts) < 78:
        p = np.array([rng.uniform(26, width - 26), rng.uniform(26, height - 26)])
        if np.linalg.norm(p - query) > 26 and all(np.linalg.norm(p - q) > 17 for q in pts):
            pts.append(p)
    pts = np.array(pts)
    dist = np.linalg.norm(pts - query, axis=1)
    nearest = set(np.argsort(dist)[:7].tolist())
    radius = float(np.sort(dist)[6]) + 12
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Illustration of the 7 nearest neighbours of a new customer">']
    parts.append(f'<circle cx="{query[0]}" cy="{query[1]}" r="{radius:.1f}" fill="{C_AMBER}" fill-opacity=".10" '
                 f'stroke="{C_AMBER}" stroke-width="1.5" stroke-dasharray="5 6"/>')
    for i in sorted(nearest):
        x, y = pts[i]
        parts.append(f'<line x1="{query[0]}" y1="{query[1]}" x2="{x:.1f}" y2="{y:.1f}" stroke="{C_AMBER}" stroke-width="1.6"/>')
    for i, (x, y) in enumerate(pts):
        if i in nearest:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{C_AMBER}" stroke="{C_INK}" stroke-width="1.5"/>')
        else:
            fill = C_TAUPE if i % 5 else C_GREY
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" fill-opacity=".8"/>')
    parts.append(f'<circle cx="{query[0]}" cy="{query[1]}" r="11" fill="{C_INK}"/>')
    parts.append(f'<circle cx="{query[0]}" cy="{query[1]}" r="4" fill="{C_AMBER}"/></svg>')
    return "".join(parts)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_footer() -> None:
    html(
        '<div class="ci-footer">Churn Insight - built on a K-Nearest Neighbors classifier. '
        "Predictions are statistical estimates meant to support, not replace, human judgement.</div>"
    )


def render_home() -> None:
    left, right = st.columns([1.2, 1], gap="large", vertical_alignment="center")
    with left:
        html(
            """
            <div class="ci-eyebrow">Customer retention analytics</div>
            <h1 class="ci-h1">Know who is leaving before they go.</h1>
            <p class="ci-lead">Churn Insight turns a customer's profile and billing details into a churn
            prediction from a trained K-Nearest Neighbors model, so retention teams can act earlier.</p>
            """
        )
        stretch_button("Explore Predictions", key="cta", on_click=go_to, args=("predict",))
    with right:
        html(
            f"""
            <div class="ci-hero-art" aria-hidden="true">{neighbours_svg()}
            <div class="ci-legend">
              <span><i style="background:{C_INK}"></i>New customer</span>
              <span><i style="background:{C_AMBER}"></i>7 nearest neighbours</span>
              <span><i style="background:{C_TAUPE}"></i>Other customers</span>
            </div></div>
            """
        )

    html(
        """
        <div class="ci-section">
          <div class="ci-eyebrow">How it works</div>
          <h2 class="ci-h2">From profile to prediction in three steps</h2>
        </div>
        <div class="ci-grid">
          <div class="ci-card"><span class="ci-num">01</span><h3 class="ci-h3">Enter Customer Details</h3>
            <p>Provide age, gender, tenure, charges, contract, internet service and tech support on the Predict page.</p></div>
          <div class="ci-card"><span class="ci-num">02</span><h3 class="ci-h3">Run the ML Model</h3>
            <p>Inputs are encoded exactly as in training, scaled with the saved StandardScaler and passed to the KNN classifier.</p></div>
          <div class="ci-card"><span class="ci-num">03</span><h3 class="ci-h3">Review the Prediction</h3>
            <p>See the predicted class and an estimated churn probability to help prioritise retention outreach.</p></div>
        </div>
        """
    )

    html(
        f"""
        <div class="ci-section"></div>
        <div class="ci-band">
          <div class="ci-eyebrow">Model overview</div>
          <h2 class="ci-h2">A compact, tuned nearest-neighbours model</h2>
          <div class="ci-stats">
            <div class="ci-stat"><b>KNN</b><span>K-Nearest Neighbors</span><small>k = 7, Manhattan distance</small></div>
            <div class="ci-stat"><b>{HEADLINE_ACCURACY}</b><span>Held-out test accuracy</span><small>200 customers never used for training</small></div>
            <div class="ci-stat"><b>12</b><span>Model input features</span><small>Derived from 8 customer fields</small></div>
            <div class="ci-stat"><b>1,000</b><span>Customers in the dataset</span><small>80% train / 20% test split</small></div>
          </div>
        </div>
        """
    )
    page_footer()


def current_inputs() -> dict:
    return {key: st.session_state.get(key) for key in DEFAULTS}


def render_predict() -> None:
    html(
        """
        <div class="ci-eyebrow">Customer assessment</div>
        <h1 class="ci-h1">Assess a customer</h1>
        <p class="ci-lead">Fill in the customer's details below. The model compares them with 800 past
        customers and estimates whether this one is likely to churn.</p>
        <div style="height:2rem"></div>
        """
    )

    model, scaler, error = load_artifacts()
    if error:
        note("error", f"<strong>Model unavailable.</strong> {error}")
        return

    col_a, col_b = st.columns([1, 1.35], gap="large")
    with col_a:
        with st.container(key="card_personal"):
            html('<div class="ci-card-title">Personal Details</div><div class="ci-card-sub">Who the customer is.</div>')
            st.number_input("Age (years)", min_value=12, max_value=100, step=1, key="in_age", format="%d")
            st.selectbox("Gender", GENDERS, key="in_gender")
            st.number_input("Tenure (months)", min_value=0, max_value=240, step=1, key="in_tenure", format="%d")
    with col_b:
        with st.container(key="card_billing"):
            html('<div class="ci-card-title">Service &amp; Billing</div><div class="ci-card-sub">What the customer pays for.</div>')
            st.number_input("Monthly charges", min_value=0.0, max_value=1000.0, step=1.0, key="in_monthly", format="%.2f")
            s1, s2 = st.columns(2)
            with s1:
                st.selectbox("Contract type", CONTRACTS, key="in_contract")
            with s2:
                st.selectbox("Internet service", INTERNET_OPTIONS, key="in_internet")
            st.selectbox("Tech support", SUPPORT_OPTIONS, key="in_support")

    values = current_inputs()
    errors, warnings = validate_inputs(values)
    for message in warnings:
        note("warning", message)

    st.write("")
    b1, b2, _ = st.columns([2, 1.2, 3.8])
    with b1:
        run = stretch_button("Predict Customer Churn", key="predict_btn")
    with b2:
        stretch_button("Reset", key="reset_btn", on_click=reset_form)

    if run:
        if errors:
            st.session_state.pop("result", None)
            for message in errors:
                note("error", message)
        else:
            try:
                outcome = predict_churn(model, scaler, build_feature_frame(raw_row(values)))
                outcome["inputs"] = dict(values)
                st.session_state["result"] = outcome
            except Exception as exc:
                st.session_state.pop("result", None)
                note("error", f"The prediction could not be completed ({type(exc).__name__}: {exc}).")

    result = st.session_state.get("result")
    if result:
        render_result(result, stale=result["inputs"] != values)
    page_footer()


def render_result(result: dict, stale: bool) -> None:
    churn = result["label"] == 1
    prob = result["probability"]
    if churn:
        css, badge, icon = "churn", "High churn risk", "&#8599;"
        headline = "Churn"
        blurb = "Customers with a profile like this one mostly left in the training data."
    else:
        css, badge, icon = "stay", "Likely to stay", "&#10003;"
        headline = "No Churn"
        blurb = "Customers with a profile like this one mostly stayed in the training data."

    meter = ""
    if prob is not None:
        pct = prob * 100
        meter = f"""
        <div class="ci-prob-row"><span class="ci-prob-label">Estimated churn probability</span>
          <span class="ci-prob-value">{pct:.1f}%</span></div>
        <div class="ci-meter" role="img" aria-label="Estimated churn probability {pct:.1f} percent">
          <div class="ci-meter-fill" style="width:{pct:.1f}%"></div></div>
        <div class="ci-ticks"><span>0%</span><span>50%</span><span>100%</span></div>
        """
    html(
        f"""
        <div class="ci-result {css}">
          <span class="ci-badge">{icon} {badge}</span>
          <h2 class="ci-h1">{headline}</h2>
          <p>{blurb}</p>
          {meter}
          <p class="ci-disclaimer">This is a model estimate, not a guarantee. The probability is the
          distance-weighted share of the 7 most similar training customers who churned.</p>
        </div>
        """
    )
    if stale:
        note("warning", "The inputs have changed since this prediction. Press <strong>Predict Customer Churn</strong> to refresh it.")

    with st.expander("See how this customer was encoded for the model"):
        table = pd.DataFrame(
            {
                "Feature": FEATURE_ORDER,
                "Encoded value": result["features"].to_numpy(),
                "Scaled value": np.round(result["scaled"], 3),
            }
        )
        st.dataframe(table, hide_index=True)


def metric_card(label: str, value: str, sub: str) -> str:
    return (
        f'<div class="ci-card"><div class="ci-muted" style="font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:.78rem">{label}</div>'
        f'<div class="ci-big" style="margin-top:.8rem">{value}</div><p>{sub}</p></div>'
    )


def render_model_details() -> None:
    html(
        """
        <div class="ci-eyebrow">Model details</div>
        <h1 class="ci-h1">How the model works</h1>
        <p class="ci-lead">What was trained, how it was tested, and what happens to a customer's details
        between the form and the prediction.</p>
        """
    )
    model, scaler, error = load_artifacts()
    if error:
        note("error", error)
        return
    evaluation = get_evaluation(model, scaler)
    report = evaluation["report"] if evaluation else NOTEBOOK_REPORT
    matrix = evaluation["matrix"] if evaluation else [[18, 5], [0, 177]]
    n_test = int(sum(map(sum, matrix)))
    correct = matrix[0][0] + matrix[1][1]
    params = model.get_params()

    html(
        '<div class="ci-section"></div><div class="ci-grid">'
        + metric_card("Algorithm", "KNN", "K-Nearest Neighbors classifier from scikit-learn.")
        + metric_card("Test accuracy", HEADLINE_ACCURACY, f"{correct} of {n_test} held-out customers classified correctly.")
        + metric_card("Input features", str(getattr(model, "n_features_in_", 12)), "Eight customer fields, encoded into twelve numeric features.")
        + metric_card("Training rows", f"{getattr(model, 'n_samples_fit_', 800):,}", f"Stored by the model; {n_test} more customers were held back for testing.")
        + "</div>"
    )

    # --- hyperparameters + why KNN ---
    html('<div class="ci-section"><div class="ci-eyebrow">Configuration</div><h2 class="ci-h2">Selected hyperparameters</h2></div>')
    html(
        f"""
        <div class="ci-grid-2">
          <div class="ci-card"><h3 class="ci-h3">Read from the saved model.pkl</h3>
            <dl class="ci-kv" style="margin-top:1rem">
              <dt>n_neighbors</dt><dd class="hl"><span>{params['n_neighbors']}</span></dd>
              <dt>weights</dt><dd class="hl"><span>{params['weights']}</span></dd>
              <dt>metric</dt><dd class="hl"><span>{params['metric']}</span></dd>
              <dt>algorithm</dt><dd>{params['algorithm']}</dd>
              <dt>leaf_size</dt><dd>{params['leaf_size']}</dd>
            </dl>
            <p class="ci-muted" style="margin-top:1rem">Highlighted values were chosen by grid search over
            n_neighbors (3 to 15, odd values), weights (uniform, distance) and metric (euclidean, manhattan).
            The rest are scikit-learn defaults.</p></div>
          <div class="ci-card"><h3 class="ci-h3">Why KNN was selected</h3>
            <p>Four classifiers were trained on the same scaled data. With default settings K-Nearest Neighbors
            had the highest held-out accuracy, so it was tuned further with grid search.</p>
            <p>KNN is easy to explain: a new customer is compared with the most similar past customers, and
            their outcomes decide the prediction.</p>
            <p class="ci-muted">The lead is small. One test customer is 0.5 percentage points, and KNN led
            Logistic Regression and SVM by exactly one customer.</p></div>
        </div>
        """
    )

    # --- comparison bars ---
    tuned_acc = correct / n_test
    bars = [("KNN, tuned (final model)", tuned_acc)] + [
        (f"{name}, default" if name.startswith("K-Nearest") else name, acc) for name, acc in NOTEBOOK_MODEL_COMPARISON
    ]
    rows = "".join(
        f'<div class="ci-bar-row{" top" if i == 0 else ""}"><span>{name}</span>'
        f'<div class="ci-bar"><i style="width:{acc * 100:.1f}%"></i></div><b>{acc:.1%}</b></div>'
        for i, (name, acc) in enumerate(bars)
    )
    html(
        f"""
        <div class="ci-section"><div class="ci-eyebrow">Training &amp; testing</div>
        <h2 class="ci-h2">How the model was trained and tested</h2></div>
        <div class="ci-grid-2">
          <div class="ci-card"><h3 class="ci-h3">Overview</h3>
            <ol class="ci-steps">
              <li><div><b>Dataset</b><span>1,000 customers with 8 input fields and a Yes/No churn label.</span></div></li>
              <li><div><b>Split</b><span>80% training (800) and 20% held-out test (200), stratified by churn, random_state 42.</span></div></li>
              <li><div><b>Scaling</b><span>StandardScaler fitted on the training rows only, then applied to the test rows.</span></div></li>
              <li><div><b>Model choice and tuning</b><span>Four default models compared, then KNN tuned with 5-fold GridSearchCV optimising F1.</span></div></li>
            </ol></div>
          <div class="ci-card"><h3 class="ci-h3">Held-out test accuracy by model</h3>
            <p class="ci-muted">Defaults as recorded in the notebook, then the tuned KNN that was saved as model.pkl.</p>
            <div style="margin-top:1rem">{rows}</div></div>
        </div>
        """
    )

    # --- test results vs cross-validation ---
    html('<div class="ci-section"><div class="ci-eyebrow">Evaluation</div><h2 class="ci-h2">Test-set results versus cross-validation</h2><p class="ci-lead">These are different measurements on different data and should not be compared directly.</p></div>')
    def fmt(x):
        return f"{x:.2f}"

    def line(name, key):
        r = report[key]
        return (f"<tr><td>{name}</td><td>{fmt(r['precision'])}</td><td>{fmt(r['recall'])}</td>"
                f"<td>{fmt(r['f1-score'])}</td><td>{int(r['support'])}</td></tr>")

    acc_row = f"<tr class='sum'><td>Accuracy</td><td></td><td></td><td>{fmt(report['accuracy'])}</td><td>{n_test}</td></tr>"
    html(
        f"""
        <div class="ci-split">
          <div class="ci-card"><span class="ci-tag test">Held-out test set - {n_test} customers</span>
            <h3 class="ci-h3">Classification report</h3>
            <div class="ci-table-wrap"><table class="ci-table" style="margin-top:1rem">
              <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
              {line('No Churn (0)', '0')}{line('Churn (1)', '1')}{acc_row}
              {line('Macro avg', 'macro avg')}{line('Weighted avg', 'weighted avg')}
            </table></div></div>
          <div class="ci-card"><span class="ci-tag test">Held-out test set - {n_test} customers</span>
            <h3 class="ci-h3">Confusion matrix</h3>
            <div class="ci-cm">
              <div></div><div class="h">Predicted<br>No Churn</div><div class="h">Predicted<br>Churn</div>
              <div class="h">Actual<br>No Churn</div>
              <div class="cell ok"><b>{matrix[0][0]}</b><small>correct</small></div>
              <div class="cell"><b>{matrix[0][1]}</b><small>false alarms</small></div>
              <div class="h">Actual<br>Churn</div>
              <div class="cell"><b>{matrix[1][0]}</b><small>missed churners</small></div>
              <div class="cell ok"><b>{matrix[1][1]}</b><small>correct</small></div>
            </div></div>
        </div>
        """
    )

    html(
        f"""
        <div class="ci-grid-2">
          <div class="ci-card"><span class="ci-tag cv">5-fold cross-validation - training data only</span>
            <h3 class="ci-h3">Best grid-search score</h3>
            <div class="ci-big" style="margin-top:.8rem">{NOTEBOOK_CV_BEST_F1:.3f}</div>
            <p>Mean F1 of the Churn class across 5 folds of the 800 training rows, for the winning setting
            (k = 7, distance weights, Manhattan). It guided hyperparameter choice and is not an accuracy figure.</p></div>
          <div class="ci-card"><span class="ci-tag test">Held-out test set - {n_test} customers</span>
            <h3 class="ci-h3">Final accuracy</h3>
            <div class="ci-big" style="margin-top:.8rem">{HEADLINE_ACCURACY}</div>
            <p>{correct} of {n_test} customers the model had never seen were classified correctly
            (reproduced here from the saved model, scaler and dataset).</p></div>
        </div>
        """
    )

    if evaluation:
        base = evaluation["report"]["1"]["support"] / evaluation["n_test"]
        note(
            "info",
            f"<strong>Read the accuracy with context.</strong> "
            f"{evaluation['n_churn']} of {evaluation['n_total']} customers in the data churned, so predicting "
            f"&ldquo;Churn&rdquo; every time would already score {base:.1%} on the test set. Also, all "
            f"{evaluation['n_mtm_churn']} of {evaluation['n_mtm']} Month-to-Month customers churned, so the "
            "model leans towards Churn for that contract type.",
        )

    # --- pipeline ---
    chips = "".join(f'<span class="ci-chip"><em>{i}</em>{name}</span>' for i, name in enumerate(FEATURE_ORDER, 1))
    html(
        f"""
        <div class="ci-section"><div class="ci-eyebrow">Prediction pipeline</div>
        <h2 class="ci-h2">What happens when you press Predict</h2></div>
        <div class="ci-card"><ol class="ci-steps">
          <li><div><b>Collect the form values</b><span>Age, gender, tenure, monthly and total charges, contract type, internet service and tech support are validated first.</span></div></li>
          <li><div><b>Encode like the training data</b><span>Gender: Female 0, Male 1. Tech support: No 0, Yes 1. Contract type and internet service are one-hot encoded (&ldquo;No Internet Service&rdquo; is the empty category <code>InternetService_</code>). Charges are cast to whole numbers, as in the notebook.</span></div></li>
          <li><div><b>Order the 12 features</b><span>Columns are arranged in the exact training order and any absent one-hot column is filled with 0.</span></div></li>
          <li><div><b>Scale with the fitted StandardScaler</b><span><code>scaler.transform()</code> only. The scaler is never refitted on customer input.</span></div></li>
          <li><div><b>Predict</b><span>The KNN model returns the class, and <code>predict_proba()</code> gives the estimated churn probability.</span></div></li>
        </ol>
        <div class="ci-muted" style="margin-top:1.2rem;font-weight:700">Feature order</div>
        <div class="ci-chips">{chips}</div></div>
        """
    )
    page_footer()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Churn Insight",
        page_icon="\U0001F342",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={"Get help": None, "Report a bug": None, "About": None},
    )
    init_state()
    page = st.session_state["page"]
    if page not in PAGES:
        page = st.session_state["page"] = "home"
    inject_css(page)
    render_navbar()
    {"home": render_home, "predict": render_predict, "model": render_model_details}[page]()


main()