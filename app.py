
import math
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# -----------------------------
# Styling (iPhone-friendly, clean dark UI)
# -----------------------------
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
}
html, body, [class*="css"]  { background-color: var(--bg) !important; color: #e7eefc !important; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 560px; }
h1,h2,h3 { letter-spacing: -0.02em; }
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.00));
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 14px 14px 12px 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.pill {
  display:inline-block; padding:6px 10px; border-radius: 999px;
  border:1px solid var(--line); color: var(--muted); font-size: 12px;
}
.bigscore {
  font-size: 72px; font-weight: 800; line-height: 1; letter-spacing:-0.04em;
}
.scoreArrow { font-size: 56px; font-weight: 800; line-height:1; color: var(--muted); padding: 0 10px; }
.subtle { color: var(--muted); font-size: 12px; }
.kv { display:flex; justify-content:space-between; gap: 10px; align-items:center; }
.k { color: var(--muted); font-size: 12px; }
.v { font-weight: 650; }
hr { border: none; border-top: 1px solid var(--line); margin: 12px 0; }
.small { font-size: 12px; color: var(--muted); }
.neon { color: var(--neon); }
.pink { color: var(--pink); }
.good { color: var(--good); }
.warn { color: var(--warn); }
.bad { color: var(--bad); }
.footer { color: var(--muted); font-size: 11px; opacity: 0.9; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Indicators
# -----------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]).abs(),
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1/n, adjust=False).mean()

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0)
    return (direction * df["Volume"].fillna(0)).cumsum()

def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    mf = tp * df["Volume"].fillna(0)
    pos = mf.where(tp.diff() > 0, 0.0)
    neg = mf.where(tp.diff() < 0, 0.0)
    pmf = pos.rolling(n).sum()
    nmf = (-neg).rolling(n).sum()
    mfr = pmf / (nmf.replace(0, np.nan))
    return 100 - (100 / (1 + mfr))

def bbands(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return lower, mid, upper

def pivot_levels(df: pd.DataFrame, lookback: int = 60) -> Tuple[float, float]:
    recent = df.tail(lookback)
    return float(recent["Low"].min()), float(recent["High"].max())

# -----------------------------
# Signal model
# -----------------------------
@dataclass
class Signal:
    score_now: int
    score_next: int
    grade: str
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

def grade_from_score(score: int) -> str:
    if score >= 90: return "SSS"
    if score >= 80: return "SS"
    if score >= 70: return "S"
    if score >= 60: return "A"
    if score >= 50: return "B"
    if score >= 40: return "C"
    return "D"

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

def clamp_int(x, lo=0, hi=100):
    return int(max(lo, min(hi, round(x))))

def compute_score(df: pd.DataFrame) -> Tuple[int, dict]:
    """
    Deterministic "rules + indicators" score (0~100).
    Focus: trend, momentum, volatility, volume confirmation.
    """
    close = df["Close"]
    last = float(close.iloc[-1])

    ma20 = sma(close, 20).iloc[-1]
    ma50 = sma(close, 50).iloc[-1]
    ma200 = sma(close, 200).iloc[-1] if len(df) >= 200 else sma(close, 100).iloc[-1]

    r = float(rsi(close, 14).iloc[-1])
    mf = float(mfi(df, 14).iloc[-1])
    a = float(atr(df, 14).iloc[-1])

    # Trend points
    trend = 0
    if last > ma20: trend += 8
    if last > ma50: trend += 10
    if last > ma200: trend += 12
    if ma20 > ma50: trend += 6
    if ma50 > ma200: trend += 8

    # Momentum points (RSI)
    mom = 0
    if r < 30: mom += 18   # oversold rebound potential
    elif r < 40: mom += 10
    elif r < 60: mom += 8
    elif r < 70: mom += 6
    else: mom += 2         # overbought -> lower score

    # Money flow (MFI)
    flow = 0
    if mf < 20: flow += 12
    elif mf < 40: flow += 8
    elif mf < 60: flow += 6
    elif mf < 80: flow += 4
    else: flow += 1

    # Bollinger context
    lower, mid, upper = bbands(close, 20, 2.0)
    lb = float(lower.iloc[-1]); mb = float(mid.iloc[-1]); ub = float(upper.iloc[-1])
    bb = 0
    if last < lb: bb += 14
    elif last < mb: bb += 8
    elif last < ub: bb += 6
    else: bb += 2

    # Volatility penalty (too wild for precision entries)
    vol = 0
    atr_pct = a / last if last else 0
    if atr_pct < 0.015: vol += 10
    elif atr_pct < 0.03: vol += 7
    elif atr_pct < 0.05: vol += 4
    else: vol += 1

    # OBV confirmation (recent slope)
    obv_series = obv(df)
    obv_slope = float(obv_series.diff().tail(10).mean())
    obv_pts = 0
    if obv_slope > 0: obv_pts += 10
    else: obv_pts += 4

    score = trend + mom + flow + bb + vol + obv_pts
    score = clamp_int(score, 0, 100)

    explain = dict(
        last=last, ma20=float(ma20), ma50=float(ma50), ma200=float(ma200),
        rsi=r, mfi=mf, atr=a, atr_pct=atr_pct,
        bb_lower=lb, bb_mid=mb, bb_upper=ub,
        obv_slope=obv_slope
    )
    return score, explain

def next_score_projection(df: pd.DataFrame, score_now: int) -> int:
    """
    Light 'projection' to show score drift (like 86 -> 69):
    - penalize if momentum deteriorating (RSI falling, price below MA20)
    - boost if recovering (RSI rising, OBV rising)
    """
    close = df["Close"]
    if len(df) < 30:
        return score_now

    r = rsi(close, 14)
    r_now = float(r.iloc[-1])
    r_prev = float(r.iloc[-6])  # ~1 week ago
    r_chg = r_now - r_prev

    ma20 = sma(close, 20)
    below20 = float(close.iloc[-1]) < float(ma20.iloc[-1])

    obv_series = obv(df)
    obv_chg = float(obv_series.iloc[-1] - obv_series.iloc[-6])

    drift = 0
    if r_chg < -6: drift -= 18
    elif r_chg < -2: drift -= 10
    elif r_chg > 6: drift += 10
    elif r_chg > 2: drift += 6

    if below20: drift -= 6
    else: drift += 3

    if obv_chg > 0: drift += 4
    else: drift -= 2

    return clamp_int(score_now + drift, 0, 100)

def label_wave(df: pd.DataFrame) -> str:
    """
    Simple wave-ish label using MA cross + momentum:
    (Not real Elliott wave detection; practical proxy.)
    """
    close = df["Close"]
    ma20 = sma(close, 20)
    ma50 = sma(close, 50)
    if len(df) < 60:
        return "파동: 데이터 부족"

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
    """
    OBV ratio-ish: recent OBV momentum vs 60d baseline
    """
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
        return "복합 패턴: 모멘텀 반등 신호 (BB 하단 이탈)"
    if last > ub:
        return "복합 패턴: 과열/추격 주의 (BB 상단 돌파)"
    if last > mb:
        return "복합 패턴: 상승 흐름 유지"
    return "복합 패턴: 조정/관망 구간"

def calc_target_stop(df: pd.DataFrame, style: str) -> Tuple[float, float]:
    """
    Target/Stop derived from:
    - support/resistance (60d pivots)
    - ATR buffers
    style: '단타' or '스윙'
    """
    last = float(df["Close"].iloc[-1])
    a = float(atr(df, 14).iloc[-1])
    support, resistance = pivot_levels(df, 60)

    if style == "단타":
        stop = max(support, last - 1.0 * a)
        target = min(resistance, last + 1.6 * a)
    else:
        stop = max(support, last - 1.5 * a)
        target = min(resistance, last + 2.6 * a)

    # Ensure sane ordering
    stop = min(stop, last * 0.999)
    target = max(target, last * 1.001)
    return float(stop), float(target)

@st.cache_data(ttl=60*10, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename_axis("Date").reset_index()
    # yfinance can return multi-index columns sometimes; normalize
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # Ensure columns
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

def sparkline_figure(df: pd.DataFrame, title: str):
    d = df.tail(80)
    close = d["Close"]
    ma = sma(close, 10)
    vol = d["Volume"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=close, mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=d.index, y=ma, mode="lines", name="MA10"))
    fig.add_trace(go.Bar(x=d.index, y=vol, name="Volume", opacity=0.35, yaxis="y2"))

    fig.update_layout(
        height=220,
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

def build_signal(ticker: str, style: str) -> Optional[Signal]:
    df = fetch_ohlcv(ticker, period="9mo", interval="1d")
    if df is None or df.empty or len(df) < 60:
        return None

    score_now, ex = compute_score(df)
    score_next = next_score_projection(df, score_now)
    grade = grade_from_score(score_next)

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

    # Weekly performance (5 trading days)
    if len(df) >= 6:
        weekly_perf = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-6]) - 1.0) * 100
    else:
        weekly_perf = float("nan")

    stop, target = calc_target_stop(df, style)

    vix = fetch_vix()
    vix_text = vix_warning(vix)

    asof = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return Signal(
        score_now=score_now,
        score_next=score_next,
        grade=grade,
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

def score_color(score: int) -> str:
    if score >= 85: return "pink"
    if score >= 70: return "good"
    if score >= 55: return "warn"
    return "bad"

def money(x: float) -> str:
    return f"${x:,.2f}"

# -----------------------------
# UI
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='subtle'>개인용 해외주식 시그널 · 로컬/자가호스팅 전용</div>", unsafe_allow_html=True)

colA, colB = st.columns([1.2, 0.8], vertical_alignment="center")
with colA:
    ticker = st.text_input("티커", value="AAPL", help="예: AAPL, NVDA, TSLA, SMR, IREN, PGY, GOOG")
with colB:
    style = st.selectbox("스타일", ["단타", "스윙"], index=0)

st.markdown("</div>", unsafe_allow_html=True)

if not ticker:
    st.stop()

ticker = ticker.strip().upper()

sig = build_signal(ticker, style)
if sig is None:
    st.error("데이터를 불러오지 못했어요. 티커를 확인하거나 잠시 후 다시 시도해줘.")
    st.stop()

df = fetch_ohlcv(ticker, period="9mo", interval="1d")
last_price = float(df["Close"].iloc[-1])

# Top VIX strip (like screenshot)
strip = ""
if sig.vix_text:
    if "경고" in sig.vix_text:
        strip = f"<span class='bad'>⚠ {sig.vix_text}</span>"
    elif "주의" in sig.vix_text:
        strip = f"<span class='warn'>⚠ {sig.vix_text}</span>"
    else:
        strip = f"<span class='good'>✓ {sig.vix_text}</span>"

st.markdown(f"<div class='card'>{strip}</div>", unsafe_allow_html=True)

# Ticker header card
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center; font-size:40px; font-weight:800;' class='neon'>{ticker}</div>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center; font-size:20px; font-weight:700;'>{ticker} <span class='subtle'> {money(last_price)}</span></div>", unsafe_allow_html=True)
st.plotly_chart(sparkline_figure(df, "Price (Close) · MA10 · Volume"), use_container_width=True)

# Score block
c = score_color(sig.score_next)
st.markdown(f"""
<div style="text-align:center; margin-top:4px;">
  <div class="subtle">AI 추천 점수</div>
  <div style="display:flex; justify-content:center; align-items:baseline; gap:10px;">
    <div class="bigscore {score_color(sig.score_now)}">{sig.score_now}</div>
    <div class="scoreArrow">→</div>
    <div class="bigscore {score_color(sig.score_next)}">{sig.score_next}</div>
  </div>
  <div class="subtle">등급 [{sig.grade}] · {style}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# Details (compact)
def line(k, v, cls=""):
    return f"<div class='kv'><div class='k'>{k}</div><div class='v {cls}'>{v}</div></div>"

details = []
details.append(line("출력 시간", sig.asof))
details.append(line("추세", sig.bias.replace("추세: ",""), "good" if "상승" in sig.bias else ("bad" if "하락" in sig.bias else "warn")))
details.append(line("주간 성과 (1W)", f"{sig.weekly_perf:+.2f}%",
                    "good" if sig.weekly_perf >= 0 else "bad"))
details.append(line("파동", sig.wave.replace("파동: ","")))
details.append(line("에너지", sig.energy.replace("에너지: ",""), "good" if "매수세" in sig.energy else ("bad" if "매도세" in sig.energy else "warn")))
if sig.obv_ratio is not None and not math.isnan(sig.obv_ratio):
    details.append(line("OBV 잔존율", f"{sig.obv_ratio:.2f}x", "good" if sig.obv_ratio >= 1 else "warn"))
details.append(line("복합 패턴", sig.pattern.replace("복합 패턴: ","")))
details.append(line("신호", f"RSI {sig.rsi:.0f} / MFI {sig.mfi:.0f}"))

st.markdown("<div>" + "".join(details) + "</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)  # close card

# Target / Stop card
st.markdown("<div class='card'>", unsafe_allow_html=True)
up_pct = (sig.target / last_price - 1) * 100
dn_pct = (sig.stop / last_price - 1) * 100

st.markdown(f"""
<div style="display:flex; gap:12px;">
  <div style="flex:1; border:1px solid var(--line); border-radius:14px; padding:12px;">
    <div class="k">목표가 (TARGET)</div>
    <div class="v good" style="font-size:22px;">{money(sig.target)} ({up_pct:+.1f}%)</div>
    <div class="small">1차저항: {money(sig.target)}</div>
  </div>
  <div style="flex:1; border:1px solid var(--line); border-radius:14px; padding:12px;">
    <div class="k">손절가 (STOP)</div>
    <div class="v bad" style="font-size:22px;">{money(sig.stop)} ({dn_pct:+.1f}%)</div>
    <div class="small">1차지지: {money(sig.stop)}</div>
  </div>
</div>
""", unsafe_allow_html=True)

rr = abs(up_pct / dn_pct) if dn_pct != 0 else float("inf")
st.markdown(f"<div class='small' style='margin-top:10px;'>리스크/리워드(%) ≈ {rr:.2f} · *이 계산은 단순 참고용(실전은 체결/슬리피지 고려)*</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Scan list
with st.expander("여러 티커 빠른 스캔(옵션)"):
    tickers_raw = st.text_area("티커 목록 (쉼표/줄바꿈)", value="NVDA,TSLA,SMR,IREN,PGY,GOOG")
    tickers = [t.strip().upper() for t in tickers_raw.replace("\n", ",").split(",") if t.strip()]
    if st.button("스캔 실행"):
        rows = []
        for t in tickers[:30]:
            s = build_signal(t, style)
            if not s: 
                continue
            d = fetch_ohlcv(t, period="9mo", interval="1d")
            last = float(d["Close"].iloc[-1])
            up = (s.target/last - 1)*100
            dn = (s.stop/last - 1)*100
            rows.append([t, last, s.score_now, s.score_next, s.grade, up, dn])
        if rows:
            out = pd.DataFrame(rows, columns=["Ticker","Last","ScoreNow","ScoreNext","Grade","Target%","Stop%"])
            out = out.sort_values("ScoreNext", ascending=False)
            st.dataframe(out, use_container_width=True, hide_index=True)
        else:
            st.info("스캔 결과가 없어요.")

st.markdown("<div class='footer'>주의: 이 앱은 투자 조언이 아니며, 개인 학습/참고용입니다.</div>", unsafe_allow_html=True)
