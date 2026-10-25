import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             silhouette_score, davies_bouldin_score, calinski_harabasz_score)
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Atmos | Air Quality Intelligence", page_icon="🌿", layout="wide")

# ───────────────────────── Tokens ─────────────────────────
BG, SIDE, GREEN, TEAL, INK, LINE, MUTED = "#F7F8F5", "#0F2A22", "#65A765", "#4F9D91", "#202923", "#E3E8E1", "#5E6B63"
POLL = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
LABEL = {"PM2.5": "PM₂.₅", "PM10": "PM₁₀", "NO2": "NO₂", "SO2": "SO₂", "CO": "CO", "O3": "O₃"}
UNIT = {"PM2.5": "µg/m³", "PM10": "µg/m³", "NO2": "µg/m³", "SO2": "µg/m³", "CO": "mg/m³", "O3": "µg/m³"}
LIME = "#B8F34A"
BANDS = [("Good", 0, 50, "#3FA66B"), ("Satisfactory", 50, 100, "#9BC53D"), ("Moderate", 100, 200, "#F2C230"),
         ("Poor", 200, 300, "#F08A24"), ("Very Poor", 300, 400, "#D9483B"), ("Severe", 400, 500, "#7A2E5C")]
CATS = [b[0] for b in BANDS]
CCOL = {b[0]: b[3] for b in BANDS}
TXT = {c: "#fff" if c in ("Very Poor", "Severe") else INK for c in CATS}
CL = [TEAL, GREEN, "#E0A93B", "#D9783A", "#8A6FB0", "#3E7CB1", "#B8567A", "#7F8C84"]
ADVICE = {
    "Good": "Air is clean. Outdoor activity is fine for everyone.",
    "Satisfactory": "Acceptable air. Unusually sensitive people may notice mild discomfort.",
    "Moderate": "People with asthma or heart and lung disease should ease off long outdoor exertion.",
    "Poor": "Most people may feel discomfort with prolonged exposure. Limit time outdoors.",
    "Very Poor": "Prolonged exposure can affect breathing. Avoid outdoor exercise.",
    "Severe": "Health risk for everyone. Stay indoors and use filtered air if possible.",
}
# Breakpoints (concentration -> sub-index 0,50,100,200,300,400,500); CPCB-style, CO in mg/m³
BP = {"PM2.5": [0, 30, 60, 90, 120, 250, 380], "PM10": [0, 50, 100, 250, 350, 430, 510],
      "NO2": [0, 40, 80, 180, 280, 400, 520], "SO2": [0, 40, 80, 380, 800, 1600, 2100],
      "CO": [0, 1, 2, 10, 17, 34, 50], "O3": [0, 50, 100, 168, 208, 748, 1000]}
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
          6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Post-monsoon", 11: "Post-monsoon"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ───────────────────────── Theme ─────────────────────────
CSS = f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Manrope:wght@600;700;800&display=swap');
html,body,.stApp,[data-testid=stMarkdownContainer],button,input,textarea{{font-family:Inter,'Segoe UI',system-ui,sans-serif}}
h1,h2,h3,h4{{font-family:Manrope,Inter,sans-serif!important;color:{INK};letter-spacing:-.02em}}
.stApp{{background:{BG};color:{INK}}}
header[data-testid=stHeader]{{background:transparent}}
[data-testid=stDecoration],.stAppDeployButton,#MainMenu,footer{{display:none!important}}
.block-container{{max-width:1240px;padding:2.4rem 2.2rem 4rem}}
[data-testid=stSidebar]{{background:{SIDE};border-right:0}}
[data-testid=stSidebar] *{{color:#E6EFE9}}
[data-testid=stSidebarNav]::before{{content:"Atmos";display:block;font:800 1.6rem Manrope,sans-serif;padding:1.2rem .8rem .7rem;color:#fff}}
[data-testid=stSidebarNavLink]{{border-radius:10px;padding:.55rem .8rem;margin:2px 0;font-weight:500}}
[data-testid=stSidebarNavLink]:hover{{background:rgba(255,255,255,.07)}}
[data-testid=stSidebarNavLink][aria-current=page]{{background:rgba(101,167,101,.18);box-shadow:inset 3px 0 0 {GREEN};font-weight:600}}
.side{{margin-top:1.6rem;padding:1rem .8rem 0;border-top:1px solid rgba(255,255,255,.12);font-size:.82rem}}
.side h6{{font:600 .88rem Inter;margin:0 0 .5rem;color:#fff}}
.side div{{display:flex;justify-content:space-between;padding:.22rem 0;color:#B9CCC2}}
.side b{{color:#fff;font-weight:600}}
.dot{{display:inline-block;width:8px;height:8px;border-radius:50%;background:{GREEN};margin-right:.4rem}}
.ph{{margin:0 0 1.6rem}}.ph h1{{font-size:2.15rem;font-weight:800;margin:0;padding:0}}
.ph p{{color:{MUTED};margin:.35rem 0 0;font-size:1.02rem;max-width:62ch}}
h3.sec{{font-size:1.15rem;font-weight:700;margin:2.2rem 0 .1rem;padding:0}}
p.sub{{color:{MUTED};margin:0 0 .6rem;font-size:.92rem}}
.strip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));border-top:1px solid {LINE};border-bottom:1px solid {LINE};margin:.8rem 0}}
.strip>div{{padding:.9rem 1.1rem;border-left:1px solid {LINE};display:flex;flex-direction:column}}
.strip>div:first-child{{border-left:0;padding-left:0}}
.strip span{{font-size:.8rem;color:{MUTED}}}.strip b{{font:800 1.6rem Manrope,sans-serif}}.strip i{{font-style:normal;font-size:.78rem;color:#7A877F}}
.sp{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));margin-top:.4rem}}
.sp>div{{padding:.2rem 1.1rem;border-left:1px solid {LINE}}}.sp>div:first-child{{border-left:0;padding-left:0}}
.chip{{display:inline-block;padding:.2rem .75rem;border-radius:99px;font-weight:600;font-size:.82rem}}
.legend{{display:flex;flex-wrap:wrap;gap:.4rem 1.1rem;font-size:.8rem;color:{MUTED};margin-top:.3rem}}
.legend i{{display:inline-block;width:.7rem;height:.7rem;border-radius:50%;margin-right:.35rem}}
.rule{{padding:.75rem 0;border-bottom:1px solid {LINE}}}
div[class*=st-key-panel]{{background:#fff;border:1px solid {LINE};border-radius:18px;padding:1.5rem 1.7rem;box-shadow:0 1px 2px rgba(21,60,50,.04)}}
.stButton>button{{border-radius:10px;border:1px solid #D5DCD3}}
.stButton>button[kind=primary]{{background:{GREEN};border:0;color:#10261F;font-weight:700}}
.stButton>button[kind=primary]:hover{{background:#74B374;color:#10261F}}
.stTabs [data-baseweb=tab-list]{{gap:1.4rem;border-bottom:1px solid {LINE}}}
.stTabs [aria-selected=true]{{color:{SIDE};font-weight:600}}
.stTabs [data-baseweb=tab-highlight]{{background:{GREEN}}}
:focus-visible{{outline:2px solid {TEAL};outline-offset:2px}}
@media(max-width:700px){{.block-container{{padding:1.5rem 1rem}}}}
</style>"""

ICONS = {
 "dashboard": "<rect width='7' height='9' x='3' y='3' rx='1'/><rect width='7' height='5' x='14' y='3' rx='1'/><rect width='7' height='9' x='14' y='12' rx='1'/><rect width='7' height='5' x='3' y='16' rx='1'/>",
 "wind": "<path d='M12.8 19.6A2 2 0 1 0 14 16H2'/><path d='M17.5 8a2.5 2.5 0 1 1 2 4H2'/><path d='M9.8 4.4A2 2 0 1 1 11 8H2'/>",
 "scatter": "<circle cx='7.5' cy='7.5' r='.5'/><circle cx='18.5' cy='5.5' r='.5'/><circle cx='11.5' cy='11.5' r='.5'/><circle cx='7.5' cy='16.5' r='.5'/><circle cx='17.5' cy='14.5' r='.5'/><path d='M3 3v16a2 2 0 0 0 2 2h16'/>",
 "share": "<circle cx='18' cy='5' r='3'/><circle cx='6' cy='12' r='3'/><circle cx='18' cy='19' r='3'/><line x1='8.59' x2='15.42' y1='13.51' y2='17.49'/><line x1='15.41' x2='8.59' y1='6.51' y2='10.49'/>",
 "activity": "<path d='M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2'/>",
 "database": "<ellipse cx='12' cy='5' rx='9' ry='3'/><path d='M3 5V19A9 3 0 0 0 21 19V5'/><path d='M3 12A9 3 0 0 0 21 12'/>",
 "arrow": "<path d='M5 12h14'/><path d='m12 5 7 7-7 7'/>"}


def datauri(inner, vb="0 0 24 24", stroke="black", sw=2):
    t = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='{vb}' fill='none' stroke='{stroke}' stroke-width='{sw}' "
         f"stroke-linecap='round' stroke-linejoin='round'>{inner}</svg>")
    return 'url("data:image/svg+xml,' + t.replace("#", "%23").replace("<", "%3C").replace(">", "%3E") + '")'


def svg(n, s=16):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{s}' height='{s}' viewBox='0 0 24 24' fill='none' stroke='currentColor' "
            f"stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>{ICONS[n]}</svg>")


NAVCSS = "".join(f"[data-testid=stSidebarNavItems] li:nth-child({i + 1}) [data-testid=stSidebarNavLink]{{--ic:{datauri(ICONS[n])}}}"
                 for i, n in enumerate(["dashboard", "wind", "scatter", "share", "activity"]))
CONT = datauri("<g stroke-opacity='.4'>" + "".join(f"<ellipse cx='260' cy='150' rx='{r}' ry='{int(r * .62)}' transform='rotate(-18 260 150)'/>" for r in range(30, 260, 26)) + "</g>",
               vb="0 0 400 300", stroke=LIME, sw=.8)
BRAND = datauri(ICONS["wind"], stroke=LIME)
CSS2 = f"""<style>
{NAVCSS}
[data-testid=stSidebar]{{background:linear-gradient(180deg,#0F2A22,#12332A)}}
[data-testid=stSidebarNav]::before{{padding-left:3rem;background:{BRAND} no-repeat .85rem 1.45rem/22px;border-bottom:1px solid rgba(255,255,255,.1);margin-bottom:.8rem}}
[data-testid=stSidebarNavLink]{{transition:background .18s,transform .18s}}
[data-testid=stSidebarNavLink]:hover{{transform:translateX(2px)}}
[data-testid=stSidebarNavLink][aria-current=page]{{background:rgba(184,243,74,.1);box-shadow:inset 3px 0 0 {LIME}}}
[data-testid=stSidebarNavLink] [data-testid=stIconMaterial]{{font-size:0!important;width:18px;height:18px;background:currentColor;-webkit-mask:var(--ic) center/contain no-repeat;mask:var(--ic) center/contain no-repeat}}
.side p.note{{color:#8FA99B;font-size:.74rem;line-height:1.4;margin:.6rem 0 0}}
.ph{{display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap}}
.pills{{display:flex;gap:.5rem;flex-wrap:wrap}}
.pill{{display:flex;align-items:center;gap:.45rem;border:1px solid #D5DCD3;background:#fff;border-radius:99px;padding:.35rem .8rem;font-size:.78rem;color:{MUTED};white-space:nowrap}}
.pill svg{{color:{TEAL}}}
[class*=st-key-panel-dark]{{background:radial-gradient(120% 140% at 85% 0%,#1E4D40 0%,#0F2A22 62%);border:0;position:relative;overflow:hidden;z-index:0}}
[class*=st-key-panel-dark]::before{{content:"";position:absolute;inset:0;background:{CONT} right center/75% no-repeat;z-index:-1;pointer-events:none}}
[class*=st-key-panel-dark] [data-testid=stMarkdownContainer]{{color:#fff}}
[class*=st-key-panel-dark] .sub,[class*=st-key-panel-dark] .legend,[class*=st-key-panel-dark] .strip span,[class*=st-key-panel-dark] .strip i,[class*=st-key-panel-dark] small{{color:#9DB5A9}}
[class*=st-key-panel-dark] .strip b{{color:#fff}}
[class*=st-key-panel-dark] .strip,[class*=st-key-panel-dark] .strip>div,[class*=st-key-panel-dark] .sp>div{{border-color:rgba(255,255,255,.14)}}
.stButton>button{{transition:transform .12s,box-shadow .12s,background .15s}}
.stButton>button:hover{{transform:translateY(-1px);box-shadow:0 4px 12px rgba(15,42,34,.12)}}
.stButton>button:active{{transform:translateY(0)}}
.stButton>button[kind=primary]{{background:{LIME};color:#10261F}}
.stButton>button[kind=primary]:hover{{background:#C8F76A}}
.empty{{border:1px dashed #C9D3C7;border-radius:16px;padding:2.6rem 1.5rem;text-align:center;color:{MUTED}}}
.empty svg{{color:{TEAL}}}.empty h4{{margin:.6rem 0 .2rem}}
.rule{{display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;transition:background .15s}}
.rule:hover{{background:#fff}}
.rl{{display:flex;align-items:center;gap:.4rem;flex-wrap:wrap}}
.rc{{padding:.2rem .6rem;border-radius:8px;font-size:.82rem;font-weight:500}}
.rc.a{{background:#E7ECE5;color:{INK}}}.rc.c{{background:#0F2A22;color:{LIME}}}
.ar{{color:#7A877F;display:flex}}
.mt{{display:flex;gap:1rem}}
.mini{{display:flex;flex-direction:column;gap:.15rem;min-width:84px;font-size:.72rem;color:{MUTED}}}
.mini div{{height:4px;background:#E7ECE5;border-radius:4px}}.mini i{{display:block;height:4px;background:{TEAL};border-radius:4px}}.mini b{{color:{INK};font-size:.82rem}}
</style>"""

pio.templates["atmos"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, Segoe UI, sans-serif", color=INK, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[TEAL, GREEN, "#E0A93B", "#D9783A", "#8A6FB0", "#3E7CB1"],
    xaxis=dict(showgrid=False, linecolor=LINE, tickcolor=LINE, zeroline=False, automargin=True),
    yaxis=dict(gridcolor="#E7ECE5", zeroline=False, linecolor="rgba(0,0,0,0)", automargin=True),
    hoverlabel=dict(bgcolor=SIDE, bordercolor=SIDE, font=dict(color="#fff", family="Inter, sans-serif")),
    legend=dict(orientation="h", y=1.1, x=0, title=dict(text="")),
    margin=dict(l=4, r=4, t=24, b=4)))
pio.templates.default = "atmos"
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(CSS2, unsafe_allow_html=True)


# ───────────────────────── UI helpers ─────────────────────────
def H(s):
    st.markdown("".join(l.strip() for l in s.splitlines()), unsafe_allow_html=True)


def head(t, s):
    H(f'<div class="ph"><div><h1>{t}</h1><p>{s}</p></div><div class="pills"><div class="pill">{svg("database", 14)}<span>Demonstration data</span></div>'
      f'<div class="pill">{svg("activity", 14)}<span>CPCB-style AQI scale</span></div></div></div>')


def sec(t, s=""):
    H(f'<h3 class="sec">{t}</h3><p class="sub">{s}</p>')


def strip(items):
    H('<div class="strip">' + "".join(f"<div><span>{l}</span><b>{v}</b><i>{n}</i></div>" for l, v, n in items) + "</div>")


def chip(c):
    return f'<span class="chip" style="background:{CCOL[c]};color:{TXT[c]}">{c}</span>'


def legend():
    return '<div class="legend"><b>CPCB-style AQI scale</b>' + "".join(f'<span><i style="background:{c}"></i>{n} {lo}–{hi}</span>' for n, lo, hi, c in BANDS) + "</div>"


def show(fig, h=340):
    fig.update_layout(height=h)
    try:
        st.plotly_chart(fig, theme=None, config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, config={"displayModeBar": False})


def table(d, **kw):
    try:
        st.dataframe(d, width="stretch", hide_index=True, **kw)
    except Exception:
        st.dataframe(d, use_container_width=True, hide_index=True, **kw)


# ───────────────────────── Data (replace load_data with your real dataset) ─────────────────────────
def sub_indices(d):
    return pd.DataFrame({p: np.interp(d[p], BP[p], [0, 50, 100, 200, 300, 400, 500]) for p in POLL}, index=d.index)


def categorize(a):
    return pd.cut(a, [-1, 50, 100, 200, 300, 400, 1e9], labels=CATS)


def cat_of(x):
    return str(categorize(pd.Series([x])).iloc[0])


@st.cache_data(show_spinner="Preparing dataset…")
def load_data():
    """Synthetic sample. To use real data return a frame with Date, Station, PM2.5, PM10, NO2, SO2, CO, O3, AQI, Category."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", "2025-12-31")
    n, doy = len(dates), dates.dayofyear.values
    stations = {"Central Plaza": (55, 9, 22, 1.1, 48), "Industrial Belt": (78, 22, 40, 1.6, 40),
                "Riverside": (48, 7, 18, 0.9, 52), "Ring Road": (66, 11, 28, 1.4, 44),
                "Coastal Edge": (30, 5, 14, 0.7, 58), "University Campus": (38, 6, 16, 0.8, 54)}
    out = []
    for name, (b25, bn, bs, bc, bo) in stations.items():
        z = np.zeros(n)
        for i in range(1, n):
            z[i] = 0.8 * z[i - 1] + rng.normal(0, 0.6)
        lat = 0.55 * np.cos(2 * np.pi * (doy - 10) / 365.25) + 0.5 * z
        ep = np.ones(n)
        for s in rng.integers(0, n - 7, 24):
            ep[s:s + rng.integers(3, 7)] *= rng.uniform(1.5, 2.3)
        e = lambda s_: np.exp(s_ * rng.normal(size=n))
        pm25 = b25 * np.exp(0.6 * lat) * e(0.12) * ep
        out.append(pd.DataFrame({"Date": dates, "Station": name, "PM2.5": pm25,
                                 "PM10": pm25 * rng.uniform(1.5, 1.9, n) * e(0.08),
                                 "NO2": bn * np.exp(0.4 * lat) * e(0.15), "SO2": bs * np.exp(0.3 * lat) * e(0.2) / 4,
                                 "CO": bc * np.exp(0.45 * lat) * e(0.15), "O3": bo * np.exp(-0.25 * lat) * e(0.12)}))
    d = pd.concat(out, ignore_index=True).round(2)
    d["AQI"] = sub_indices(d).max(axis=1).round(0)
    d["Category"] = categorize(d["AQI"])
    return d


df = load_data()


# ───────────────────────── Models ─────────────────────────
@st.cache_resource(show_spinner="Training Random Forest…")
def train_rf(d, test=0.2, trees=200):
    Xtr, Xte, ytr, yte = train_test_split(d[POLL], d["AQI"], test_size=test, random_state=42)
    m = RandomForestRegressor(n_estimators=trees, min_samples_leaf=2, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    p = m.predict(Xte)
    mse = mean_squared_error(yte, p)
    return dict(model=m, y=yte.values, p=p, imp=pd.Series(m.feature_importances_, POLL), MAE=mean_absolute_error(yte, p),
                MSE=mse, RMSE=mse ** .5, R2=r2_score(yte, p), trainR2=m.score(Xtr, ytr))


@st.cache_data(show_spinner="Clustering…")
def run_kmeans(d, k):
    sc = StandardScaler()
    X = sc.fit_transform(d[POLL])
    km = KMeans(k, n_init=10, random_state=42).fit(X)
    order = pd.Series(d["AQI"].values).groupby(km.labels_).mean().sort_values().index
    remap = {old: i + 1 for i, old in enumerate(order)}  # cluster 1 = lowest mean AQI
    lab = np.array([remap[l] for l in km.labels_])
    pca = PCA(2, random_state=42).fit(X)
    idx = [remap[i] for i in range(k)]
    cen = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=POLL, index=idx).sort_index()
    cxy = pd.DataFrame(pca.transform(km.cluster_centers_), index=idx).sort_index().values
    return dict(lab=lab, xy=pca.transform(X), cen=cen, cxy=cxy, var=pca.explained_variance_ratio_,
                sil=silhouette_score(X, lab, sample_size=min(3000, len(X)), random_state=42),
                dbi=davies_bouldin_score(X, lab), ch=calinski_harabasz_score(X, lab), inertia=km.inertia_)


@st.cache_data(show_spinner="Comparing cluster counts…")
def sweep(d):
    X = StandardScaler().fit_transform(d[POLL])
    r = []
    for k in range(2, 10):
        km = KMeans(k, n_init=5, random_state=42).fit(X)
        r.append((k, km.inertia_, silhouette_score(X, km.labels_, sample_size=min(2500, len(X)), random_state=42)))
    return pd.DataFrame(r, columns=["k", "Inertia", "Silhouette"])


@st.cache_data(show_spinner="Mining rules…")
def mine(d, levels, season, minsup, maxlen):
    names = {3: ["Low", "Medium", "High"], 4: ["Low", "Moderate", "High", "Very high"]}[levels]
    t = pd.DataFrame(index=d.index)
    for p in POLL:
        t[p] = LABEL[p] + ": " + pd.qcut(d[p], levels, labels=names).astype(str)
    t["AQI"] = "AQI: " + d["Category"].astype(str)
    if season:
        t["Season"] = "Season: " + d["Date"].dt.month.map(SEASON)
    oh = pd.get_dummies(t, prefix="", prefix_sep="").astype(bool)
    fi = apriori(oh, min_support=minsup, use_colnames=True, max_len=maxlen)
    if fi.empty:
        return pd.DataFrame(), pd.DataFrame()
    try:
        try:
            r = association_rules(fi, metric="confidence", min_threshold=0.3)
        except TypeError:
            r = association_rules(fi, num_itemsets=len(oh), metric="confidence", min_threshold=0.3)
    except ValueError:
        r = pd.DataFrame()
    fi["Size"] = fi["itemsets"].map(len)
    fi["Itemset"] = fi["itemsets"].map(lambda s: ", ".join(sorted(s)))
    fi = fi.rename(columns={"support": "Support"})[["Itemset", "Size", "Support"]]
    if r.empty:
        return fi, r
    r["Antecedents"] = r["antecedents"].map(lambda s: ", ".join(sorted(s)))
    r["Consequents"] = r["consequents"].map(lambda s: ", ".join(sorted(s)))
    r = r.rename(columns={"support": "Support", "confidence": "Confidence", "lift": "Lift"})
    return fi, r[["Antecedents", "Consequents", "Support", "Confidence", "Lift"]]


# ───────────────────────── Page: Overview ─────────────────────────
def ribbon(avg, p10, p90):
    f = go.Figure()
    for n, lo, hi, c in BANDS:
        f.add_shape(type="rect", x0=lo, x1=hi, y0=0, y1=1, fillcolor=c, line=dict(color="#0F2A22", width=2))
        f.add_trace(go.Scatter(x=[(lo + hi) / 2], y=[.5], mode="markers", marker=dict(opacity=0, size=30), showlegend=False,
                               hovertemplate=f"<b>{n}</b><br>AQI {lo} to {hi}<extra></extra>"))
    f.add_trace(go.Scatter(x=[p10, p90], y=[-.3, -.3], mode="lines+markers", showlegend=False, line=dict(color="#fff", width=3),
                           marker=dict(symbol="line-ns", size=12, line=dict(width=2, color="#fff")),
                           hovertemplate="Typical range edge: AQI %{x:.0f}<extra></extra>"))
    f.add_trace(go.Scatter(x=[avg], y=[1.2], mode="markers+text", text=[f"Average {avg:.0f}"], textposition="top center",
                           marker=dict(symbol="triangle-down", size=16, color=LIME), textfont=dict(color="#fff"), showlegend=False,
                           hovertemplate="Average AQI %{x:.0f}<extra></extra>"))
    f.add_annotation(x=(p10 + p90) / 2, y=-.3, yshift=-18, showarrow=False, text="Typical day (10th to 90th percentile)",
                     font=dict(size=11, color="#9DB5A9"))
    f.update_xaxes(range=[0, 500], tickvals=[0, 50, 100, 200, 300, 400, 500], title="AQI", color="#C9D8CF")
    f.update_yaxes(visible=False, range=[-.9, 1.9])
    return f


def spark(s):
    v = s.values
    if len(v) < 3:
        return ""
    x, y = np.linspace(0, 100, len(v)), 28 - (v - v.min()) / (np.ptp(v) + 1e-9) * 24
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    return (f'<svg viewBox="0 0 100 30" preserveAspectRatio="none" style="width:100%;height:34px">'
            f'<polyline points="{pts}" fill="none" stroke="{TEAL}" stroke-width="1.8" vector-effect="non-scaling-stroke"/></svg>')


def overview():
    head("Air quality overview", "How AQI and its pollutants have moved across monitoring stations, and what drives the poor days.")
    c1, c2 = st.columns([1, 2])
    stn = c1.selectbox("Station", ["All stations"] + sorted(df.Station.unique()))
    lo_, hi_ = df.Date.min().date(), df.Date.max().date()
    rg = c2.date_input("Period", (lo_, hi_), min_value=lo_, max_value=hi_)
    if len(rg) != 2:
        st.info("Choose an end date to continue.")
        return
    d = df[(df.Date >= pd.Timestamp(rg[0])) & (df.Date <= pd.Timestamp(rg[1]))]
    if stn != "All stations":
        d = d[d.Station == stn]
    if d.empty:
        st.warning("No records for this selection. Widen the period.")
        return

    avg = d.AQI.mean()
    p10, p90 = d.AQI.quantile(.1), d.AQI.quantile(.9)
    with st.container(key="panel-dark-hero"):
        a, b = st.columns([1, 2.2], gap="large")
        with a:
            H(f'<p class="sub">Average AQI, {stn.lower()}</p><div style="font:800 4.4rem/1 Manrope">{avg:.0f}</div>'
              f'<p style="margin:.7rem 0">{chip(cat_of(avg))}</p><p class="sub">{ADVICE[cat_of(avg)]}</p>')
        with b:
            show(ribbon(avg, p10, p90), 240)
            H(legend())
        strip([("Cleanest day", f"{d.AQI.min():.0f}", "lowest daily AQI"), ("Worst day", f"{d.AQI.max():.0f}", "highest daily AQI"),
               ("Poor or worse", f"{(d.AQI > 200).mean():.0%}", "share of records"), ("Records", f"{len(d):,}", f"{d.Date.nunique():,} days")])

    sec("Pollutants", "Period average, monthly trend, and where the average sits on the AQI scale.")
    means = d[POLL].mean()
    si = sub_indices(means.to_frame().T).iloc[0]
    cells = ""
    for p in POLL:
        c = cat_of(si[p])
        cells += (f'<div><span style="color:{MUTED};font-size:.85rem">{LABEL[p]}</span>'
                  f'<div style="font:800 1.7rem Manrope">{means[p]:.1f}<small style="font:500 .75rem Inter;color:#7A877F"> {UNIT[p]}</small></div>'
                  f'{spark(d.set_index("Date")[p].resample("MS").mean())}'
                  f'<div style="height:4px;background:#E7ECE5;border-radius:4px;margin:.4rem 0 .25rem"><div style="height:4px;width:{min(si[p] / 5, 100):.0f}%;background:{CCOL[c]};border-radius:4px"></div></div>'
                  f'<small style="color:#7A877F">Sub-index {si[p]:.0f}, {c}</small></div>')
    H(f'<div class="sp">{cells}</div>')

    sec("AQI over time", "Shaded bands show the AQI category at each level.")
    f1, f2 = st.columns([1, 2])
    res = f1.radio("Resolution", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True)
    split = f2.toggle("Split by station", False) if stn == "All stations" else False
    g = d.groupby([pd.Grouper(key="Date", freq={"Daily": "D", "Weekly": "W", "Monthly": "MS"}[res])] + (["Station"] if split else []))["AQI"].mean().reset_index()
    top = max(g.AQI.max() * 1.1, 120)
    fig = go.Figure()
    for n, lo, hi, c in BANDS:
        if lo < top:
            fig.add_hrect(y0=lo, y1=min(hi, top), fillcolor=c, opacity=.1, line_width=0)
    if split:
        for s, gg in g.groupby("Station"):
            fig.add_scatter(x=gg.Date, y=gg.AQI, name=s, mode="lines", hovertemplate="%{x|%d %b %Y}<br>AQI %{y:.0f}<extra>" + s + "</extra>")
    else:
        fig.add_scatter(x=g.Date, y=g.AQI, name="AQI", mode="lines", line=dict(color=TEAL, width=2), hovertemplate="%{x|%d %b %Y}<br>AQI %{y:.0f}<extra></extra>")
        if res == "Daily":
            fig.add_scatter(x=g.Date, y=g.AQI.rolling(30, min_periods=7).mean(), name="30-day average", line=dict(color=INK, width=2.5),
                            hovertemplate="%{x|%d %b %Y}<br>30-day avg %{y:.0f}<extra></extra>")
    fig.update_yaxes(title="AQI", range=[0, top])
    show(fig, 380)

    l, r = st.columns([1.5, 1], gap="large")
    with l:
        sec("Days by category", "Share of records in each AQI category, per station.")
        ct = (pd.crosstab(d.Station, d.Category, normalize="index") * 100).reindex(columns=CATS, fill_value=0)
        ct = ct.loc[ct[["Poor", "Very Poor", "Severe"]].sum(axis=1).sort_values().index]
        f = go.Figure()
        for c in CATS:
            f.add_bar(y=ct.index, x=ct[c], name=c, orientation="h", marker_color=CCOL[c], hovertemplate="%{y}<br>" + c + ": %{x:.1f}% of records<extra></extra>")
        f.update_layout(barmode="stack")
        f.update_xaxes(title="% of records", range=[0, 100])
        show(f, 320)
    with r:
        sec("What drives AQI", "Pollutant with the highest sub-index.")
        dom = sub_indices(d).idxmax(axis=1).value_counts(normalize=True).reindex(POLL, fill_value=0) * 100
        f = go.Figure(go.Bar(x=dom.values, y=[LABEL[p] for p in dom.index], orientation="h", marker_color=TEAL,
                             hovertemplate="%{y}: %{x:.1f}% of records<extra></extra>"))
        f.update_yaxes(autorange="reversed")
        f.update_xaxes(title="% of records")
        show(f, 320)

    sec("Pollution patterns")
    t1, t2, t3 = st.tabs(["Seasonal pattern", "Pollutant relationships", "Station comparison"])
    with t1:
        m = d.groupby(d.Date.dt.month)[POLL].mean()
        z = (m / m.mean()).T
        f = go.Figure(go.Heatmap(z=z.values, x=[MONTHS[i - 1] for i in z.columns], y=[LABEL[p] for p in z.index],
                                 colorscale=[[0, TEAL], [.5, BG], [1, "#D9783A"]], zmid=1, text=z.values.round(2), texttemplate="%{text}",
                                 hovertemplate="%{y}, %{x}: %{z:.2f}× the average month<extra></extra>", colorbar=dict(title="× mean", thickness=10)))
        f.update_yaxes(autorange="reversed")
        show(f, 330)
    with t2:
        cm = d[POLL + ["AQI"]].corr()
        lb = [LABEL.get(c, c) for c in cm.columns]
        f = go.Figure(go.Heatmap(z=cm.values, x=lb, y=lb, zmin=-1, zmax=1, colorscale=[[0, "#D9783A"], [.5, "#fff"], [1, TEAL]],
                                 text=cm.values.round(2), texttemplate="%{text}", hovertemplate="%{y} vs %{x}: r = %{z:.2f}<extra></extra>",
                                 colorbar=dict(title="r", thickness=10)))
        f.update_yaxes(autorange="reversed")
        show(f, 400)
    with t3:
        p = st.selectbox("Pollutant", POLL, format_func=LABEL.get)
        f = go.Figure()
        for s in sorted(d.Station.unique()):
            f.add_box(y=d[d.Station == s][p], name=s, marker_color=TEAL, line_color=TEAL, boxpoints=False)
        f.update_yaxes(title=f"{LABEL[p]} ({UNIT[p]})")
        f.update_layout(showlegend=False)
        show(f, 360)


# ───────────────────────── Page: Prediction ─────────────────────────
STEP = {p: 0.5 for p in POLL}
STEP["CO"] = 0.05


def snap(p, v, mx):
    return float(np.clip(np.round(round(v / STEP[p]) * STEP[p], 2), 0, mx))


GROUPS = {"Particulate matter (µg/m³)": ["PM2.5", "PM10"], "Gases (µg/m³, CO in mg/m³)": ["NO2", "SO2", "CO", "O3"]}


def gauge(v):
    return go.Figure(go.Indicator(mode="gauge+number", value=v, number=dict(font=dict(size=58, family="Manrope", color="#fff")),
                                  gauge=dict(axis=dict(range=[0, 500], tickvals=[0, 100, 200, 300, 400, 500], tickcolor="#9DB5A9", tickfont=dict(color="#9DB5A9")),
                                             bar=dict(color="#fff", thickness=.2), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                                             steps=[dict(range=[lo, hi], color=c) for _, lo, hi, c in BANDS])))


def prediction():
    head("AQI prediction", "Enter pollutant concentrations and run the Random Forest. Nothing is estimated until you press Run Prediction.")
    R = train_rf(df)
    mdl = R["model"]
    mx = {p: float(np.ceil(df[p].quantile(.999) * 1.25)) for p in POLL}
    PRE = {"Typical day": df[POLL].median(), "Clean day": df[POLL].quantile(.05), "Heavy smog": df[POLL].quantile(.97)}
    saved = st.session_state.get("saved", {})
    for p in POLL:
        st.session_state.setdefault("in_" + p, snap(p, saved.get(p, PRE["Typical day"][p]), mx[p]))

    def preset(n):
        for p in POLL:
            st.session_state["in_" + p] = snap(p, PRE[n][p], mx[p])

    L, Rt = st.columns([1, 1.1], gap="large")
    with L:
        sec("Presets", "Start from a quantile of the loaded dataset, then adjust.")
        for c, n in zip(st.columns(3), PRE):
            c.button(n, on_click=preset, args=(n,), key="pre_" + n)
        vals = {}
        for g, ps in GROUPS.items():
            sec(g)
            cols = st.columns(2)
            for i, p in enumerate(ps):
                vals[p] = cols[i % 2].slider(LABEL[p], 0.0, mx[p], step=STEP[p], key="in_" + p)
        if st.button("Run Prediction", type="primary"):
            with st.spinner("Running the Random Forest across all trees…"):
                time.sleep(.6)
                st.session_state["pred"] = dict(vals)
            st.toast("Prediction complete")
        elif "pred" in st.session_state and st.session_state["pred"] != vals:
            st.caption("Inputs changed since the last run. Press Run Prediction to update.")
    st.session_state["saved"] = dict(vals)

    with Rt:
        pr = st.session_state.get("pred")
        if not pr:
            H(f'<div class="empty">{svg("wind", 30)}<h4>No prediction yet</h4><p>Set the six pollutant values and press Run Prediction. '
              'The estimate comes from the Random Forest trained on the loaded dataset.</p></div>')
            return
        x = pd.DataFrame([pr])[POLL]
        tp = np.array([t.predict(x.values)[0] for t in mdl.estimators_])
        pred, (q1, q9) = float(tp.mean()), np.percentile(tp, [10, 90])
        si_ = sub_indices(x).iloc[0]
        cat = cat_of(pred)
        with st.container(key="panel-dark-result"):
            show(gauge(pred), 280)
            H(f'<p style="text-align:center;margin:0">{chip(cat)}</p><p class="sub" style="text-align:center;margin-top:.6rem">{ADVICE[cat]}</p>')
            strip([("Likely range", f"{q1:.0f} to {q9:.0f}", "middle 80% of trees"), ("Higher than", f"{(df.AQI < pred).mean():.0%}", "of dataset records"),
                   ("Main driver", LABEL[si_.idxmax()], f"formula AQI {si_.max():.0f}")])
            H('<div class="sp">' + "".join(f'<div><span style="font-size:.78rem;color:#9DB5A9">{LABEL[p]}</span><div style="font:700 1.1rem Manrope">{pr[p]:g}'
                                           f'<small> {UNIT[p]}</small></div></div>' for p in POLL) + "</div>")

    a, b = st.columns(2, gap="large")
    with a:
        sec("Sub-index by pollutant", "Under the breakpoint method the highest sub-index sets the AQI.")
        f = go.Figure(go.Bar(x=si_.values, y=[LABEL[p] for p in POLL], orientation="h", marker_color=[CCOL[cat_of(v)] for v in si_.values],
                             hovertemplate="%{y}: sub-index %{x:.0f}<extra></extra>"))
        f.add_vline(x=pred, line_dash="dot", line_color=INK, annotation_text="Predicted")
        f.update_yaxes(autorange="reversed")
        f.update_xaxes(title="Sub-index")
        show(f, 320)
    with b:
        sec("Where this sits in the data", "Distribution of AQI across all records.")
        f = go.Figure(go.Histogram(x=df.AQI, nbinsx=50, marker_color=TEAL, opacity=.6, hovertemplate="AQI %{x}<br>%{y} records<extra></extra>"))
        f.add_vline(x=pred, line_dash="dot", line_color=INK, annotation_text="Predicted")
        f.update_xaxes(title="AQI")
        f.update_yaxes(title="Records")
        show(f, 320)


# ───────────────────────── Page: K-Means ─────────────────────────
def kpi(M):
    strip([("Silhouette", f"{M['sil']:.2f}", "higher is better"), ("Davies–Bouldin", f"{M['dbi']:.2f}", "lower is better"),
           ("Calinski–Harabasz", f"{M['ch']:,.0f}", "higher is better"), ("Inertia", f"{M['inertia']:,.0f}", "within-cluster spread")])


def clustering():
    head("K-Means clustering", "Group days by pollutant profile to reveal distinct air quality regimes. Cluster 1 always has the lowest mean AQI.")
    st.session_state.setdefault("w_k", st.session_state.get("k", 4))
    k = st.slider("Number of clusters (k)", 2, 8, key="w_k")
    st.session_state["k"] = k
    M = run_kmeans(df, k)
    lab = M["lab"]
    kpi(M)
    ma = df.groupby(lab).AQI.mean()
    l, r = st.columns([2, 1], gap="large")
    with l:
        sec("Cluster map", f"Each point is one station-day. Choose any two axes; principal components capture {M['var'].sum():.0%} of variance.")
        opts = ["Principal component 1", "Principal component 2"] + POLL
        nm = lambda o: LABEL.get(o, o)
        ttl = lambda o: o if o.startswith("Principal") else f"{LABEL[o]} ({UNIT[o]})"
        ax_, ay_ = st.columns(2)
        xo = ax_.selectbox("X axis", opts, 0, format_func=nm)
        yo = ay_.selectbox("Y axis", opts, 1, format_func=nm)

        def axis(o):
            if o.startswith("Principal"):
                j = int(o[-1]) - 1
                return M["xy"][:, j], M["cxy"][:, j]
            return df[o].values, M["cen"][o].values
        X1, cx = axis(xo)
        Y1, cy = axis(yo)
        f = go.Figure()
        for i in range(1, k + 1):
            m = lab == i
            f.add_trace(go.Scattergl(x=X1[m], y=Y1[m], mode="markers", name=f"Cluster {i}", marker=dict(size=5, color=CL[i - 1], opacity=.55),
                                     hovertext=[f"AQI {a:.0f}, {s}" for a, s in zip(df.AQI.values[m], df.Station.values[m])],
                                     hovertemplate=f"Cluster {i}<br>%{{hovertext}}<extra></extra>"))
        f.add_trace(go.Scatter(x=cx, y=cy, mode="markers+text", name="Centroids", text=[str(i) for i in range(1, k + 1)],
                               textfont=dict(color="#fff", size=11), marker=dict(size=24, color=INK, line=dict(color="#0F2A22", width=2)),
                               hovertemplate="Centroid %{text}<extra></extra>"))
        f.update_xaxes(title=ttl(xo))
        f.update_yaxes(title=ttl(yo))
        show(f, 480)
    with r:
        sec("Cluster sizes", "Records per cluster.")
        sz = pd.Series(lab).value_counts().sort_index()
        f = go.Figure(go.Bar(x=sz.values, y=[f"Cluster {i}" for i in sz.index], orientation="h", marker_color=CL[:k], cliponaxis=False,
                             text=[f"{v / len(df):.0%}" for v in sz.values], textposition="outside", customdata=ma.values,
                             hovertemplate="%{y}: %{x} records<br>Mean AQI %{customdata:.0f}<extra></extra>"))
        f.update_yaxes(autorange="reversed")
        f.update_xaxes(range=[0, sz.max() * 1.25], title="Records")
        show(f, 480)

    cen = M["cen"]
    t1, t2, t3, t4 = st.tabs(["Centroid profile", "Pollutant comparison", "Timing", "Choosing k"])
    with t1:
        a, b = st.columns([1, 1.2], gap="large")
        with a:
            n = (cen - cen.min()) / (cen.max() - cen.min() + 1e-9)
            f = go.Figure()
            th = [LABEL[p] for p in POLL] + [LABEL[POLL[0]]]
            for i in cen.index:
                f.add_trace(go.Scatterpolar(r=list(n.loc[i]) + [n.loc[i].iloc[0]], theta=th, name=f"Cluster {i}", line=dict(color=CL[i - 1]), fill="toself", opacity=.35))
            f.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 1])))
            show(f, 380)
        with b:
            c = cen.copy()
            c.insert(0, "Cluster", [f"Cluster {i}" for i in c.index])
            c["Mean AQI"] = ma.values
            cmap = LinearSegmentedColormap.from_list("g", ["#FFFFFF", "#BFE0C0"])
            cols_ = POLL + ["Mean AQI"]
            table(c.style.format({p: "{:.1f}" for p in cols_}).background_gradient(cmap=cmap, subset=cols_))
            st.caption("Centroids in original units: µg/m³, CO in mg/m³. Radar axes are scaled 0 to 1 across clusters.")
    with t2:
        z = (cen - df[POLL].mean()) / df[POLL].std()
        f = go.Figure()
        for i in z.index:
            f.add_bar(x=[LABEL[p] for p in POLL], y=z.loc[i].values, name=f"Cluster {i}", marker_color=CL[i - 1],
                      hovertemplate="%{x}: %{y:+.2f} SD from dataset mean<extra>Cluster " + str(i) + "</extra>")
        f.update_layout(barmode="group")
        f.update_yaxes(title="Standard deviations from dataset mean")
        show(f, 380)
    with t3:
        mc = pd.crosstab(df.Date.dt.month, lab, normalize="index") * 100
        f = go.Figure()
        for i in mc.columns:
            f.add_bar(x=[MONTHS[m - 1] for m in mc.index], y=mc[i], name=f"Cluster {i}", marker_color=CL[i - 1],
                      hovertemplate="%{x}: %{y:.1f}% of records<extra>Cluster " + str(i) + "</extra>")
        f.update_layout(barmode="stack")
        f.update_yaxes(title="% of records")
        show(f, 380)
    with t4:
        sw = sweep(df)
        a, b = st.columns(2, gap="large")
        for col, key, ttl in ((a, "Inertia", "Inertia (elbow)"), (b, "Silhouette", "Silhouette score")):
            with col:
                f = go.Figure(go.Scatter(x=sw.k, y=sw[key], mode="lines+markers", line=dict(color=TEAL, width=2.5), hovertemplate="k = %{x}<br>" + key + " %{y:.2f}<extra></extra>"))
                f.add_vline(x=k, line_dash="dot", line_color=INK, annotation_text="Selected")
                f.update_xaxes(title="k", dtick=1)
                f.update_yaxes(title=ttl)
                show(f, 320)


# ───────────────────────── Page: Association rules ─────────────────────────
def association():
    head("Association rules", "Find combinations of pollution levels that tend to appear on the same day (Apriori).")
    c = st.columns(4)
    levels = c[0].radio("Levels per pollutant", [3, 4], horizontal=True, format_func=lambda n: {3: "3 levels", 4: "4 levels"}[n])
    season = c[1].toggle("Include season", True)
    minsup = c[2].slider("Min support", .01, .5, .08, .01)
    maxlen = c[3].slider("Max itemset size", 2, 4, 3)
    c = st.columns(4)
    minconf = c[0].slider("Min confidence", .3, 1.0, .6, .05)
    minlift = c[1].slider("Min lift", 1.0, 5.0, 1.2, .1)
    focus = c[2].selectbox("Consequent", ["Any", "AQI category only"])
    find = c[3].text_input("Antecedent contains", "", placeholder="e.g. PM₂.₅: High")

    fi, R = mine(df, levels, season, minsup, maxlen)
    if R.empty:
        st.warning("No rules found at this support. Lower the minimum support.")
        return
    F = R[(R.Confidence >= minconf) & (R.Lift >= minlift)]
    if focus != "Any":
        F = F[F.Consequents.str.contains("AQI:", regex=False)]
    if find:
        F = F[F.Antecedents.str.contains(find, case=False, regex=False)]
    F = F.sort_values(["Lift", "Confidence"], ascending=False).reset_index(drop=True)
    if F.empty:
        st.info("No rules match these filters. Lower the confidence or lift threshold.")
        return
    strip([("Frequent itemsets", f"{len(fi):,}", "at this support"), ("Rules mined", f"{len(R):,}", "confidence 30% or more"),
           ("Rules shown", f"{len(F):,}", "after filters"), ("Strongest lift", f"{F.Lift.max():.2f}", "top rule")])
    sec("Strongest rules", "Grey: conditions observed together. Dark: the outcome that follows.")
    chips = lambda t, c: "".join(f'<span class="rc {c}">{x}</span>' for x in t.split(", "))
    mini = lambda v, m, l: f'<div class="mini"><span>{l}</span><div><i style="width:{min(v / m, 1) * 100:.0f}%"></i></div><b>{v:.2f}</b></div>'
    H("".join(f'<div class="rule"><div class="rl">{chips(r.Antecedents, "a")}<span class="ar">{svg("arrow")}</span>{chips(r.Consequents, "c")}</div>'
              f'<div class="mt">{mini(r.Support, 1, "Support")}{mini(r.Confidence, 1, "Confidence")}{mini(r.Lift, 5, "Lift")}</div></div>' for r in F.head(8).itertuples()))
    t1, t2, t3 = st.tabs(["Rules", "Support and confidence", "Frequent itemsets"])
    with t1:
        table(F, height=420, column_config={
            "Support": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "Lift": st.column_config.NumberColumn(format="%.2f")})
        st.download_button("Download rules (CSV)", F.to_csv(index=False), "association_rules.csv", "text/csv")
    with t2:
        f = go.Figure(go.Scatter(x=F.Support, y=F.Confidence, mode="markers", text=F.Antecedents + " → " + F.Consequents,
                                 marker=dict(size=np.clip(F.Lift * 6, 6, 22), color=F.Lift, colorscale=[[0, "#CFE6DF"], [1, SIDE]], opacity=.8,
                                             colorbar=dict(title="Lift", thickness=10)),
                                 hovertemplate="%{text}<br>Support %{x:.3f}<br>Confidence %{y:.2f}<extra></extra>"))
        f.update_xaxes(title="Support")
        f.update_yaxes(title="Confidence")
        show(f, 420)
    with t3:
        table(fi.sort_values("Support", ascending=False), height=420,
              column_config={"Support": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})


# ───────────────────────── Page: Evaluation ─────────────────────────
def evaluation():
    head("Model evaluation", "Held-out accuracy for the Random Forest, and cluster quality for K-Means.")
    c1, c2 = st.columns(2)
    test = c1.slider("Test share", .1, .4, .2, .05)
    trees = c2.select_slider("Trees", [100, 200, 300], 200)
    R = train_rf(df, test, trees)
    y, p = R["y"], R["p"]
    res = y - p
    cats = categorize(pd.Series(y))
    sec("Regression: Random Forest", "Errors are in AQI points on records the model did not see during training.")
    strip([("MAE", f"{R['MAE']:.2f}", "mean absolute error"), ("MSE", f"{R['MSE']:.2f}", "mean squared error"),
           ("RMSE", f"{R['RMSE']:.2f}", "AQI points"), ("R²", f"{R['R2']:.3f}", f"train {R['trainR2']:.3f}")])
    a, b = st.columns([1.3, 1], gap="large")
    with a:
        f = go.Figure()
        for n, _, _, c in BANDS:
            m = (cats == n).values
            if m.any():
                f.add_trace(go.Scattergl(x=y[m], y=p[m], mode="markers", name=n, marker=dict(size=5, color=c, opacity=.7),
                                         hovertemplate="Actual %{x:.0f}<br>Predicted %{y:.0f}<extra></extra>"))
        mx = max(y.max(), p.max())
        f.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect fit", line=dict(color=INK, dash="dot"), hoverinfo="skip"))
        f.update_xaxes(title="Actual AQI")
        f.update_yaxes(title="Predicted AQI")
        show(f, 400)
    with b:
        f = go.Figure(go.Histogram(x=res, nbinsx=40, marker_color=TEAL, opacity=.7, hovertemplate="Residual %{x:.1f}<br>%{y} records<extra></extra>"))
        f.add_vline(x=0, line_dash="dot", line_color=INK)
        f.update_xaxes(title="Residual (actual − predicted)")
        f.update_yaxes(title="Records")
        show(f, 400)
    a, b = st.columns(2, gap="large")
    with a:
        f = go.Figure(go.Scattergl(x=p, y=res, mode="markers", marker=dict(size=5, color=TEAL, opacity=.55), hovertemplate="Predicted %{x:.0f}<br>Residual %{y:.1f}<extra></extra>"))
        f.add_hline(y=0, line_dash="dot", line_color=INK)
        f.update_xaxes(title="Predicted AQI")
        f.update_yaxes(title="Residual")
        show(f, 330)
    with b:
        imp = R["imp"].sort_values()
        f = go.Figure(go.Bar(x=imp.values, y=[LABEL[i] for i in imp.index], orientation="h", marker_color=TEAL, hovertemplate="%{y}: %{x:.1%} importance<extra></extra>"))
        f.update_xaxes(title="Feature importance", tickformat=".0%")
        show(f, 330)
    e = pd.DataFrame({"c": cats, "e": np.abs(res)}).groupby("c", observed=False).e.mean().reindex(CATS)
    sec("Error by AQI category", "Mean absolute error for each category of the actual AQI.")
    f = go.Figure(go.Bar(x=e.index, y=e.values, marker_color=[CCOL[c] for c in e.index], hovertemplate="%{x}: MAE %{y:.2f}<extra></extra>"))
    f.update_yaxes(title="MAE (AQI points)")
    show(f, 280)
    st.caption("If AQI was calculated from these same pollutants, as in the sample data, scores will be close to perfect. Expect lower R² on measured data.")

    k = st.session_state.get("k", 4)
    M = run_kmeans(df, k)
    sec("Clustering: K-Means", f"Evaluated at k = {k}, set on the K-Means page. Silhouette suggests {'strong' if M['sil'] > .5 else 'reasonable' if M['sil'] > .25 else 'weak'} structure.")
    kpi(M)
    sw = sweep(df)
    f = go.Figure(go.Scatter(x=sw.k, y=sw.Silhouette, mode="lines+markers", line=dict(color=TEAL, width=2.5), hovertemplate="k = %{x}<br>Silhouette %{y:.2f}<extra></extra>"))
    f.add_vline(x=k, line_dash="dot", line_color=INK, annotation_text="Selected")
    f.update_xaxes(title="k", dtick=1)
    f.update_yaxes(title="Silhouette score")
    show(f, 300)


# ───────────────────────── Navigation and sidebar ─────────────────────────
pg = st.navigation([
    st.Page(overview, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page(prediction, title="AQI prediction", icon=":material/air:", url_path="prediction"),
    st.Page(clustering, title="K-Means clusters", icon=":material/bubble_chart:", url_path="clusters"),
    st.Page(association, title="Association rules", icon=":material/hub:", url_path="rules"),
    st.Page(evaluation, title="Model evaluation", icon=":material/monitoring:", url_path="evaluation"),
])
with st.sidebar:
    train_rf(df)
    H(f'<div class="side"><h6><span class="dot" style="background:#E0A93B"></span>Demonstration data</h6>'
      f'<div>Source<b>Synthetic sample</b></div><div>Records<b>{len(df):,}</b></div><div>Stations<b>{df.Station.nunique()}</b></div>'
      f'<div>Period<b>{df.Date.min():%b %Y} to {df.Date.max():%b %Y}</b></div><div>AQI standard<b>CPCB-style</b></div>'
      f'<p class="note">Values on every page come from generated sample data, not measured readings. Replace load_data() with your dataset.</p></div>')
pg.run()
