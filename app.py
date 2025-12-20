# app.py
import math
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# =============================
# Page + Style (iPhone friendly)
# =============================
st.set_page_config(page_title="Trade Signal (Personal)", page_icon="📈", layout="centered")

CUSTOM_CSS = """
<style>
:root{
  --bg:#0b0f16;
  --card:#0f1724;
  --muted:#93a4bf;
  --line:#1c2a42;
  --good:#22c55e;
  --warn:#f59e0b;
  --bad:#ef4444;
  --neon:#40e0ff;
  --pink:#ff4fd8;
  --gray:#a7b2c5;
}
html, body, [class*="css"]  { background-color: var(--bg) !important; color: #e7eefc !important; }
.block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 560px; }
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.00));
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 14px 14px 12px 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.subtle { color: var(--muted); font-size: 12px; }
.small { font-size: 12px; color: var(--muted); }
.kv { display:flex; justify-content:space-between; gap:10px; align-items:center; }
.k { color: var(--muted); font-size: 12px; }
.v { font-weight: 650; }
hr { border: none; border-top: 1px solid var(--line); margin: 12px 0; }
.neon { color: var(--neon); }
.pink { color: var(--pink); }
.good { color: var(--good); }
.warn { color: var(--warn); }
.bad { color: var(--bad); }
.gray { color: var(--gray); }
.bigscore {
  font-size: 78px; font-weight: 900; line-height: 1; letter-spacing:-0.04em;
}
.footer { color: var(--muted); font-size: 11px; opacity: 0.92; margin-top: 6px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================
# Timeframe config
# =============================
TF_OPTIONS = {
    "스윙 (1D)": {"interval": "1d",  "period": "1y",   "min_bars": 120},
    "단타 (1H)": {"interval": "1h",  "period": "180d", "min_bars": 220},
    "단타 (15m)": {"interval": "15m","period": "60d",  "min_bars": 320},  # yfinance 제한상 60d가 안전
}

# =============================
# Indicators (pure pandas)
# =============================
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.clip(0, 100)

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).abs(),
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1/n, adjust=False).mean()

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"].fillna(0)).cumsum()

# ✅ FIXED MFI (0~100 보장, 음수/이상치 제거)
def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    mf = tp * df["Volume"].fillna(0)

    pos = mf.where(tp.diff() > 0, 0.0)
    neg = mf.where(tp.diff() < 0, 0.0)

    pmf = pos.rolling(n).sum()
    nmf = neg.abs().rolling(n).sum()

    mfr = pmf / (nmf.replace(0, np.nan))
    out = 100 - (100 / (1 + mfr))
    return out.clip(0, 100)

def bbands(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return lower, mid, upper

def pivot_levels(df: pd.DataFrame, lookback: int = 60) -> Tuple[float, float]:
    recent = df.tail(lookback)
    return float(recent["Low"].min()), float(recent["High"].max())

# =============================
# Helpers
# =============================
def clamp_int(x, lo=0, hi=100):
    return int(max(lo, min(hi, round(x))))

def money(x: float) -> str:
    return f"${x:,.2f}"

def grade_from_score(score: int) -> str:
    # 공격적(상위 등급 좁힘)
    if score >= 92: return "SSS"
    if score >= 84: return "SS"
    if score >= 75: return "S"
    if score >= 63: return "A"
    if score >= 52: return "B"
    if score >= 40: return "C"
    return "D"

# ✅ 더 공격적인 색 기준
def score_class_for_ui(score: int) -> str:
    if score >= 88: return "pink"   # SS~SSS
    if score >= 72: return "good"   # S~A
    if score >= 55: return "warn"   # B
    return "gray"                   # 관망

def vix_warning(vix: Optional[float]) -> Optional[str]:
    if vix is None or (isinstance(vix, float) and math.isnan(vix)):
        return None
    if vix >= 30:
        return f"VIX 경고 ({vix:.1f}): 변동성 매우 큼 — 포지션 축소/현금 비중 권장"
    if vix >= 25:
        return f"VIX 경고 ({vix:.1f}): 변동성 큼 — 손절 엄수/분할 진입 권장"
    if vix >= 20:
        return f"VIX 주의 ({vix:.1f}): 변동성 상승 — 무리한 추격매수 금지"
    return f"VIX 안정 ({vix:.1f})"

# =============================
# Data fetch
# =============================
@st.cache_data(ttl=60*10, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename_axis("Date").reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    keep = ["Date","Open","High","Low","Close","Volume"]
    df = df[keep].dropna(subset=["Close"]).copy()
    df.set_index("Date", inplace=True)
    return df

@st.cache_data(ttl=60*10, show_spinner=False)
def fetch_vix() -> Optional[float]:
    v = yf.download("^VIX", period="10d", interval="1d", progress=False)
    if v is None or v.empty:
        return None
    try:
        return float(v["Close"].dropna().iloc[-1])
    except Exception:
        return None

# =============================
# Chart
# =============================
def sparkline_figure(df: pd.DataFrame, title: str):
    d = df.tail(160)
    close = d["Close"]
    ma = sma(close, 10)
    vol = d["Volume"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=close, mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=d.index, y=ma, mode="lines", name="MA10"))
    fig.add_trace(go.Bar(x=d.index, y=vol, name="Volume", opacity=0.35, yaxis="y2"))

    fig.update_layout(
        height=230,
        margin=dict(l=10,r=10,t=30,b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(text=title, x=0.02, y=0.95, font=dict(size=14)),
        xaxis=dict(showgrid=False, zeroline=False, showline=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10)),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig

# =============================
# 실전형 스코어 (핵심만)
# =============================
def compute_score(df: pd.DataFrame) -> Tuple[int, dict]:
    close = df["Close"]
    last = float(close.iloc[-1])

    ma20 = float(sma(close, 20).iloc[-1])
    ma50 = float(sma(close, 50).iloc[-1])
    ma200 = float(sma(close, 200).iloc[-1]) if len(df) >= 200 else float(sma(close, 100).iloc[-1])

    r = float(rsi(close, 14).iloc[-1])
    mf = float(mfi(df, 14).iloc[-1])
    a = float(atr(df, 14).iloc[-1])
    atr_pct = (a / last) if last else 0.0

    vol_now = float(df["Volume"].tail(20).mean())
    vol_base = float(df["Volume"].tail(80).mean()) if len(df) >= 80 else float(df["Volume"].mean())
    vol_ratio = (vol_now / vol_base) if vol_base else 1.0

    # Trend
    trend = 0
    trend += 12 if last > ma20 else 0
    trend += 14 if last > ma50 else 0
    trend += 16 if last > ma200 else 0
    trend += 8 if ma20 > ma50 else 0
    trend += 10 if ma50 > ma200 else 0

    # Momentum (RSI)
    mom = 0
    if r < 30: mom += 20
    elif r < 40: mom += 14
    elif r < 55: mom += 10
    elif r < 65: mom += 7
    elif r < 75: mom += 4
    else: mom += 1

    # Money flow (MFI)
    flow = 0
    if mf < 20: flow += 12
    elif mf < 40: flow += 9
    elif mf < 60: flow += 7
    elif mf < 80: flow += 4
    else: flow += 1

    # Volatility filter
    vol = 0
    if atr_pct < 0.012: vol += 12
    elif atr_pct < 0.025: vol += 9
    elif atr_pct < 0.045: vol += 5
    else: vol += 2

    # Volume confirmation
    vol_conf = 0
    if vol_ratio >= 1.3: vol_conf += 10
    elif vol_ratio >= 1.1: vol_conf += 7
    elif vol_ratio >= 0.9: vol_conf += 5
    else: vol_conf += 3

    score = clamp_int(trend + mom + flow + vol + vol_conf, 0, 100)

    explain = dict(
        last=last, ma20=ma20, ma50=ma50, ma200=ma200,
        rsi=r, mfi=mf, atr=a, atr_pct=atr_pct, vol_ratio=vol_ratio,
        trend=trend, mom=mom, flow=flow, vol=vol, vol_conf=vol_conf
    )
    return score, explain

# =============================
# Labels
# =============================
def label_wave(df: pd.DataFrame) -> str:
    close = df["Close"]
    if len(df) < 60:
        return "파동: 데이터 부족"
    ma20 = sma(close, 20)
    ma50 = sma(close, 50)
    slope20 = float(ma20.diff().tail(5).mean())
    slope50 = float(ma50.diff().tail(5).mean())
    r = float(rsi(close, 14).iloc[-1])

    if slope20 > 0 and slope50 > 0 and r > 55:
        return "파동: 상승 파동 (추세 지속)"
    if slope20 < 0 and r < 45:
        return "파동: 조정/횡보 파동"
    if r < 35:
        return "파동: 반등 준비 (과매도)"
    return "파동: 혼합"

def label_energy(df: pd.DataFrame) -> Tuple[str, Optional[float]]:
    o = obv(df)
    if len(o) < 80:
        return "에너지: 보통", None
    recent = o.diff().tail(10).mean()
    base = o.diff().tail(60).mean()
    if base == 0 or math.isnan(base):
        return "에너지: 보통", None
    ratio = float(recent / base)
    if ratio > 1.15:
        return "에너지: 매수세 증가 (강함)", ratio
    if ratio > 0.95:
        return "에너지: 매수/매도 균형", ratio
    return "에너지: 매도세 우위", ratio

def label_pattern(df: pd.DataFrame) -> str:
    close = df["Close"]
    lower, mid, upper = bbands(close, 20, 2.0)
    last = float(close.iloc[-1])
    lb = float(lower.iloc[-1]); mb = float(mid.iloc[-1]); ub = float(upper.iloc[-1])

    if last < lb:
        return "복합 패턴: 반등 후보 (BB 하단 이탈)"
    if last > ub:
        return "복합 패턴: 과열/추격 주의 (BB 상단 돌파)"
    if last > mb:
        return "복합 패턴: 상승 흐름 유지"
    return "복합 패턴: 조정/관망 구간"

# =============================
# Target / Stop (TF + style 반영)
# =============================
def calc_target_stop(df: pd.DataFrame, style: str, tf_choice: str) -> Tuple[float, float]:
    last = float(df["Close"].iloc[-1])
    a = float(atr(df, 14).iloc[-1])
    support, resistance = pivot_levels(df, 60)

    interval = TF_OPTIONS[tf_choice]["interval"]
    if interval == "15m":
        tf_stop_mul, tf_tgt_mul = 0.9, 1.4
    elif interval == "1h":
        tf_stop_mul, tf_tgt_mul = 1.0, 1.6
    else:  # 1d
        tf_stop_mul, tf_tgt_mul = 1.2, 2.2

    if style == "단타":
        stop = max(support, last - (1.0 * tf_stop_mul) * a)
        target = min(resistance, last + (1.7 * tf_tgt_mul) * a)
    else:  # 스윙
        stop = max(support, last - (1.4 * tf_stop_mul) * a)
        target = min(resistance, last + (2.6 * tf_tgt_mul) * a)

    stop = min(stop, last * 0.999)
    target = max(target, last * 1.001)
    return float(stop), float(target)

# =============================
# 최종 전략 라인
# =============================
def final_action_line(score: int, bias: str, rsi_val: float, vix: Optional[float], last_price: float, ma20: float, tf_label: str) -> str:
    trend_up = ("상승" in bias)
    vix_high = (vix is not None and vix >= 25)

    if score >= 84 and trend_up and last_price >= ma20 and rsi_val <= 68 and not vix_high:
        return f"▶ 전략: 추세 추종 진입(분할) / 손절 엄수 · {tf_label}"

    if rsi_val < 35 and score >= 63:
        return f"▶ 전략: 단기 반등 노림(분할) / 빠른 익절 우선 · {tf_label}"

    if vix_high:
        return f"▶ 전략: 변동성 주의(포지션 축소) / 무리한 추격금지 · {tf_label}"

    if score < 52 or (last_price < ma20 and rsi_val < 45):
        return f"▶ 전략: 관망(추격금지) / 지지 확인 후 접근 · {tf_label}"

    return f"▶ 전략: 눌림 대기 후 분할매수 / 손절 엄수 · {tf_label}"

# =============================
# 점수 이유(한 줄) — “왜 이 점수인지”
# =============================
def score_reason_one_line(ex: dict) -> str:
    last = ex["last"]; ma20 = ex["ma20"]; ma50 = ex["ma50"]; ma200 = ex["ma200"]
    r = ex["rsi"]; mf = ex["mfi"]; atr_pct = ex["atr_pct"]; vr = ex["vol_ratio"]

    parts = []

    # Trend 요약
    if last > ma20 and last > ma50 and last > ma200:
        parts.append("추세 강함(주요 MA 상단)")
    elif last < ma20 and last < ma50:
        parts.append("추세 약함(MA 하단)")
    else:
        parts.append("추세 혼합(경계 구간)")

    # Momentum 요약
    if r < 35:
        parts.append("RSI 과매도(반등 여지)")
    elif r < 55:
        parts.append("RSI 중립(확신 낮음)")
    elif r < 70:
        parts.append("RSI 양호(모멘텀 유지)")
    else:
        parts.append("RSI 과열(추격 주의)")

    # Flow 요약
    if mf < 30:
        parts.append("MFI 낮음(자금 유입 약)")
    elif mf < 70:
        parts.append("MFI 보통(수급 중립)")
    else:
        parts.append("MFI 높음(수급 강)")

    # Volatility 요약
    if atr_pct >= 0.045:
        parts.append("변동성 큼(신호 신뢰↓)")
    elif atr_pct >= 0.025:
        parts.append("변동성 보통")
    else:
        parts.append("변동성 낮음(세팅 유리)")

    # Volume confirmation
    if vr >= 1.3:
        parts.append("거래량 확인(강)")
    elif vr >= 1.1:
        parts.append("거래량 확인(보통)")
    else:
        parts.append("거래량 약함")

    # 너무 길지 않게 3~4개로 제한
    return "점수 이유: " + " · ".join(parts[:4])

# =============================
# Signal model
# =============================
@dataclass
class Signal:
    score: int
    grade: str
    score_reason: str
    bias: str
    wave: str
    energy: str
    pattern: str
    obv_ratio: Optional[float]
    rsi: float
    mfi: float
    weekly_perf: float
    target: float
    stop: float
    vix: Optional[float]
    vix_text: Optional[str]
    asof: str

def build_signal(ticker: str, style: str, tf_choice: str) -> Optional[Tuple[Signal, pd.DataFrame, dict]]:
    tf = TF_OPTIONS[tf_choice]
    df = fetch_ohlcv(ticker, period=tf["period"], interval=tf["interval"])
    if df is None or df.empty or len(df) < tf["min_bars"]:
        return None

    score, ex = compute_score(df)
    grade = grade_from_score(score)
    reason = score_reason_one_line(ex)

    close = float(df["Close"].iloc[-1])
    ma50 = float(sma(df["Close"], 50).iloc[-1])
    ma200 = float(sma(df["Close"], 200).iloc[-1]) if len(df) >= 200 else float(sma(df["Close"], 100).iloc[-1])

    if close > ma50 and ma50 > ma200:
        bias = "추세: 상승장 (강함)"
    elif close < ma50 and ma50 < ma200:
        bias = "추세: 하락장 (주의)"
    else:
        bias = "추세: 횡보장"

    wave = label_wave(df)
    energy, obv_ratio = label_energy(df)
    pattern = label_pattern(df)

    interval = tf["interval"]
    if interval == "1d":
        steps = 5
    elif interval == "1h":
        steps = 6 * 5
    else:
        steps = 26 * 5

    if len(df) > steps:
        weekly_perf = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-(steps+1)]) - 1.0) * 100
    else:
        weekly_perf = float("nan")

    stop, target = calc_target_stop(df, style, tf_choice)

    vix = fetch_vix()
    vix_text = vix_warning(vix)
    asof = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sig = Signal(
        score=score,
        grade=grade,
        score_reason=reason,
        bias=bias,
        wave=wave,
        energy=energy,
        pattern=pattern,
        obv_ratio=obv_ratio,
        rsi=float(ex["rsi"]),
        mfi=float(ex["mfi"]),
        weekly_perf=float(weekly_perf),
        target=float(target),
        stop=float(stop),
        vix=vix,
        vix_text=vix_text,
        asof=asof
    )
    return sig, df, ex

# =============================
# UI
# =============================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>개인용 해외주식 시그널 · 타임프레임/스타일 분리</div>", unsafe_allow_html=True)

c1, c2 = st.columns([1.1, 0.9], vertical_alignment="center")
with c1:
    ticker = st.text_input("티커", value="AAPL", help="예: AAPL, NVDA, TSLA, SMR, IREN, PGY, GOOG")
with c2:
    style = st.selectbox("스타일", ["단타", "스윙"], index=0)

tf_default = "스윙 (1D)" if style == "스윙" else "단타 (1H)"
tf_choice = st.selectbox("타임프레임", list(TF_OPTIONS.keys()), index=list(TF_OPTIONS.keys()).index(tf_default))

st.markdown("</div>", unsafe_allow_html=True)

if not ticker:
    st.stop()

ticker = ticker.strip().upper()

result = build_signal(ticker, style, tf_choice)
if result is None:
    st.error(f"데이터가 부족하거나 티커를 불러오지 못했어요. ({tf_choice}) 다른 타임프레임으로 바꿔보거나 잠시 후 다시 시도해줘.")
    st.stop()

sig, df, ex = result
last_price = float(df["Close"].iloc[-1])
ma20_ui = float(sma(df["Close"], 20).iloc[-1])

# VIX strip
strip = ""
if sig.vix_text:
    if "경고" in sig.vix_text:
        strip = f"<span class='bad'>⚠ {sig.vix_text}</span>"
    elif "주의" in sig.vix_text:
        strip = f"<span class='warn'>⚠ {sig.vix_text}</span>"
    else:
        strip = f"<span class='good'>✓ {sig.vix_text}</span>"

st.markdown(f"<div class='card'>{strip}<div class='small' style='margin-top:6px;'>TF: {tf_choice} · 스타일: {style}</div></div>", unsafe_allow_html=True)

# Header card
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center; font-size:40px; font-weight:900;' class='neon'>{ticker}</div>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center; font-size:20px; font-weight:750;'>{ticker} <span class='subtle'> {money(last_price)}</span></div>", unsafe_allow_html=True)
st.plotly_chart(sparkline_figure(df, f"Price · {tf_choice} · MA10 · Volume"), use_container_width=True)

# ✅ 점수는 “하나만”
cls = score_class_for_ui(sig.score)
st.markdown(
    f"""
<div style="text-align:center; margin-top:6px;">
  <div class="subtle">AI 추천 점수</div>
  <div class="bigscore {cls}">{sig.score}</div>
  <div class="subtle">등급 [{sig.grade}]</div>
</div>
""",
    unsafe_allow_html=True,
)

# ✅ 점수 이유 한 줄
st.markdown(
    f"<div class='small' style='text-align:center; margin-top:6px;'>{sig.score_reason}</div>",
    unsafe_allow_html=True
)

# 전략 한 줄
action = final_action_line(
    score=sig.score,
    bias=sig.bias,
    rsi_val=sig.rsi,
    vix=sig.vix,
    last_price=last_price,
    ma20=ma20_ui,
    tf_label=tf_choice,
)
st.markdown(f"<div style='text-align:center; margin-top:10px; font-weight:800;' class='neon'>{action}</div>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

def line(k, v, cls=""):
    return f"<div class='kv'><div class='k'>{k}</div><div class='v {cls}'>{v}</div></div>"

details = []
details.append(line("출력 시간", sig.asof))
details.append(line("추세", sig.bias.replace("추세: ",""), "good" if "상승" in sig.bias else ("bad" if "하락" in sig.bias else "warn")))
details.append(line("주간 성과 (≈1W)", f"{sig.weekly_perf:+.2f}%", "good" if sig.weekly_perf >= 0 else "bad"))
details.append(line("파동", sig.wave.replace("파동: ","")))
details.append(line("에너지", sig.energy.replace("에너지: ",""), "good" if "매수세" in sig.energy else ("bad" if "매도세" in sig.energy else "warn")))
if sig.obv_ratio is not None and not math.isnan(sig.obv_ratio):
    details.append(line("OBV 잔존율", f"{sig.obv_ratio:.2f}x", "good" if sig.obv_ratio >= 1 else "warn"))
details.append(line("복합 패턴", sig.pattern.replace("복합 패턴: ","")))
details.append(line("신호", f"RSI {sig.rsi:.0f} / MFI {sig.mfi:.0f}"))
details.append(line("MA20", money(ma20_ui), "gray"))

st.markdown("<div>" + "".join(details) + "</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)  # close header card

# Target / Stop card
st.markdown("<div class='card'>", unsafe_allow_html=True)
up_pct = (sig.target / last_price - 1) * 100
dn_pct = (sig.stop / last_price - 1) * 100

st.markdown(
    f"""
<div style="display:flex; gap:12px;">
  <div style="flex:1; border:1px solid var(--line); border-radius:14px; padding:12px;">
    <div class="k">목표가 (TARGET)</div>
    <div class="v good" style="font-size:22px;">{money(sig.target)} ({up_pct:+.1f}%)</div>
    <div class="small">1차저항(추정): {money(sig.target)}</div>
  </div>
  <div style="flex:1; border:1px solid var(--line); border-radius:14px; padding:12px;">
    <div class="k">손절가 (STOP)</div>
    <div class="v bad" style="font-size:22px;">{money(sig.stop)} ({dn_pct:+.1f}%)</div>
    <div class="small">1차지지(추정): {money(sig.stop)}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

rr = abs(up_pct / dn_pct) if dn_pct != 0 else float("inf")
st.markdown(
    f"<div class='small' style='margin-top:10px;'>리스크/리워드(%) ≈ {rr:.2f} · *체결/슬리피지 고려 필요*</div>",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# Quick scan
with st.expander("여러 티커 빠른 스캔(옵션)"):
    tickers_raw = st.text_area("티커 목록 (쉼표/줄바꿈)", value="NVDA,TSLA,SMR,IREN,PGY,GOOG")
    tickers = [t.strip().upper() for t in tickers_raw.replace("\n", ",").split(",") if t.strip()]
    if st.button("스캔 실행"):
        rows = []
        tf = TF_OPTIONS[tf_choice]
        for t in tickers[:30]:
            res = build_signal(t, style, tf_choice)
            if not res:
                continue
            s, d, _ = res
            last = float(d["Close"].iloc[-1])
            up = (s.target/last - 1)*100
            dn = (s.stop/last - 1)*100
            rows.append([t, last, s.score, s.grade, up, dn])
        if rows:
            out = pd.DataFrame(rows, columns=["Ticker","Last","Score","Grade","Target%","Stop%"])
            out = out.sort_values("Score", ascending=False)
            st.dataframe(out, use_container_width=True, hide_index=True)
        else:
            st.info("스캔 결과가 없어요. (데이터 부족/티커 확인)")

st.markdown("<div class='footer'>주의: 이 앱은 투자 조언이 아니며, 개인 학습/참고용입니다.</div>", unsafe_allow_html=True)
