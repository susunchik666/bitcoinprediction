from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from io import BytesIO
import base64
from typing import Iterable, Optional

import pandas as pd
import mplfinance as mpf

from models import Candle


class InvestError(RuntimeError):
    pass


# --- Импорты SDK (поддерживаем и новый t-tech-investments, и старый tinkoff-investments) ---
try:
    # Новый SDK (как в документации T-Bank)
    from t_tech.invest import Client, CandleInterval  # type: ignore
    from t_tech.invest.utils import now  # type: ignore
    _SDK_NAME = "t-tech-investments"
except Exception:  # pragma: no cover
    try:
        # Старый SDK (PyPI)
        from tinkoff.invest import Client, CandleInterval  # type: ignore
        from tinkoff.invest.utils import now  # type: ignore
        _SDK_NAME = "tinkoff-investments"
    except Exception as e:  # pragma: no cover
        Client = None  # type: ignore
        CandleInterval = None  # type: ignore
        now = None  # type: ignore
        _SDK_NAME = "not-installed"


@dataclass(frozen=True, slots=True)
class CandleRequest:
    instrument_id: str
    days_back: int = 10
    interval: str = "4h"  # '1m', '5m', '15m', '1h', '4h', '1d'


def _quotation_to_float(q) -> float:
    """В SDK цены обычно приходят как Quotation(units, nano)."""
    # Часто у объекта есть units/nano (protobuf)
    units = getattr(q, "units", 0)
    nano = getattr(q, "nano", 0)
    try:
        return float(units) + float(nano) / 1_000_000_000.0
    except Exception:
        # На всякий случай
        return float(units)


def _interval_from_str(interval: str):
    if CandleInterval is None:
        raise InvestError("SDK не установлен")

    m = {
        "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
        "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
        "15m": CandleInterval.CANDLE_INTERVAL_15_MIN,
        "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
        "4h": CandleInterval.CANDLE_INTERVAL_4_HOUR,
        "1d": CandleInterval.CANDLE_INTERVAL_DAY,
    }
    if interval not in m:
        raise InvestError(f"Неизвестный интервал: {interval}. Пример: 4h, 1h, 15m")
    return m[interval]


def fetch_candles(token: str, req: CandleRequest) -> list[Candle]:
    """Получаем свечи и возвращаем список dataclass Candle (Model)."""
    if Client is None or now is None:
        raise InvestError(
            "SDK T-Invest не установлен.\n"
            "Установите: pip install t-tech-investments --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple"
        )

    interval_enum = _interval_from_str(req.interval)
    from_dt = now() - timedelta(days=req.days_back)

    candles: list[Candle] = []

    try:
        with Client(token) as client:
            # В разных версиях параметр может называться instrument_id или figi
            try:
                it = client.get_all_candles(
                    instrument_id=req.instrument_id,
                    interval=interval_enum,
                    from_=from_dt,
                )
            except TypeError:
                it = client.get_all_candles(
                    figi=req.instrument_id,
                    interval=interval_enum,
                    from_=from_dt,
                )

            for c in it:
                candles.append(
                    Candle(
                        time=c.time,
                        open=_quotation_to_float(c.open),
                        high=_quotation_to_float(c.high),
                        low=_quotation_to_float(c.low),
                        close=_quotation_to_float(c.close),
                        volume=int(getattr(c, "volume", 0)),
                    )
                )
    except Exception as e:
        raise InvestError(str(e)) from e

    if not candles:
        raise InvestError("Свечи не найдены (проверьте instrument_id/FIGI и период)")

    return candles


def candles_to_dataframe(candles: list[Candle]) -> pd.DataFrame:
    df = pd.DataFrame([c.as_dict() for c in candles])
    df.set_index("time", inplace=True)
    return df


def plot_candles_base64(df: pd.DataFrame) -> str:
    """Строим свечной график и возвращаем PNG как base64 строку."""
    buf = BytesIO()
    mpf.plot(df, type="candle", volume=True, style="charles", savefig=buf)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def sdk_name() -> str:
    return _SDK_NAME


import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

from models import PDTClassifier, PDTRegressor


def ichimoku_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ichimoku components.
    For ML features we avoid "future-shifted" spans to prevent leakage.
    For plotting we can still use the classic shifts later.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2.0
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2.0

    span_a_raw = (tenkan + kijun) / 2.0
    span_b_raw = (high.rolling(52).max() + low.rolling(52).min()) / 2.0

    out = pd.DataFrame(
        {
            "tenkan": tenkan,
            "kijun": kijun,
            "span_a_raw": span_a_raw,
            "span_b_raw": span_b_raw,
            "close": close,
        },
        index=df.index,
    )
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic feature set: returns, volatility, volume change + Ichimoku (no leakage).
    """
    dfi = df.copy()

    dfi["ret_1"] = dfi["close"].pct_change()
    dfi["ret_3"] = dfi["close"].pct_change(3)
    dfi["vol_10"] = dfi["ret_1"].rolling(10).std()
    dfi["ma_10"] = dfi["close"].rolling(10).mean()
    dfi["ma_20"] = dfi["close"].rolling(20).mean()
    dfi["vol_chg"] = dfi["volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    ichi = ichimoku_components(dfi)
    dfi["tenkan"] = ichi["tenkan"]
    dfi["kijun"] = ichi["kijun"]
    dfi["cloud_thickness"] = (ichi["span_a_raw"] - ichi["span_b_raw"]).abs()

    # Drop raw OHLC columns? Keep close/volume for signal is ok.
    feat_cols = [
        "ret_1", "ret_3", "vol_10", "ma_10", "ma_20",
        "vol_chg", "tenkan", "kijun", "cloud_thickness",
    ]
    return dfi[feat_cols]


def make_dataset(df: pd.DataFrame, horizon: int, task: str):
    Xdf = build_features(df)
    if task == "cls":
        y = (df["close"].shift(-horizon) > df["close"]).astype(int)
    else:
        y = df["close"].shift(-horizon)

    data = Xdf.copy()
    data["y"] = y
    data = data.dropna()

    X = data.drop(columns=["y"]).to_numpy(dtype=float)
    y = data["y"].to_numpy()
    return X, y, data.index


def train_and_predict(df: pd.DataFrame, horizon: int, task: str, model_name: str, train_ratio: float):
    X, y, idx = make_dataset(df, horizon=horizon, task=task)

    n = len(X)
    split = max(10, int(n * train_ratio))
    split = min(split, n - 1)

    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    if task == "cls":
        if model_name == "tree":
            model = DecisionTreeClassifier(max_depth=5, random_state=42)
        else:
            model = PDTClassifier(n_estimators=35, max_depth=5, random_state=42)

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics = {
            "Accuracy": float(accuracy_score(y_test, pred)),
            "F1": float(f1_score(y_test, pred)),
        }

        # Forecast for the last available row
        last_proba = float(model.predict_proba(X[-1:].copy())[:, 1][0])
        direction = "UP" if last_proba >= 0.5 else "DOWN"
        return {
            "task": "cls",
            "direction": direction,
            "proba_up": last_proba,
            "metrics": metrics,
        }

    else:
        if model_name == "tree":
            model = DecisionTreeRegressor(max_depth=5, random_state=42)
        else:
            model = PDTRegressor(n_estimators=35, max_depth=5, random_state=42)

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rmse = mean_squared_error(y_test, pred, squared=False)
        mae = mean_absolute_error(y_test, pred)

        metrics = {"MAE": float(mae), "RMSE": float(rmse)}

        y_hat = float(model.predict(X[-1:].copy())[0])
        return {
            "task": "reg",
            "y_hat": y_hat,
            "metrics": metrics,
        }


def plot_candles_with_ichimoku_base64(df: pd.DataFrame, forecast: dict, horizon: int) -> str:
    """
    Prettier plot:
    - candles + volume
    - Ichimoku (tenkan/kijun + cloud)
    - marker for forecasted point
    """
    dfi = df.copy()

    ichi = ichimoku_components(dfi)
    tenkan = ichi["tenkan"]
    kijun = ichi["kijun"]
    span_a = ((tenkan + kijun) / 2.0).shift(26)
    span_b = ((dfi["high"].rolling(52).max() + dfi["low"].rolling(52).min()) / 2.0).shift(26)

    addplots = []
    addplots.append(mpf.make_addplot(tenkan, color="deepskyblue", width=1))
    addplots.append(mpf.make_addplot(kijun, color="orange", width=1))
    addplots.append(mpf.make_addplot(span_a, color="lime", width=1, alpha=0.6))
    addplots.append(mpf.make_addplot(span_b, color="red", width=1, alpha=0.6))

    # Marker: predicted next point (use last close + horizon shift as a visual anchor)
    marker_series = pd.Series(index=dfi.index, dtype=float)
    last_idx = dfi.index[-1]
    if forecast.get("task") == "reg":
        marker_series.loc[last_idx] = forecast["y_hat"]
    else:
        # for classification just mark last close
        marker_series.loc[last_idx] = float(dfi["close"].iloc[-1])

    addplots.append(
        mpf.make_addplot(marker_series, type="scatter", markersize=80, marker="*", color="gold")
    )

    buf = BytesIO()

    mpf.plot(
        dfi,
        type="candle",
        volume=True,
        style="yahoo",
        addplot=addplots,
        fill_between=dict(y1=span_a.values, y2=span_b.values, alpha=0.08, color="gray"),
        savefig=dict(fname=buf, dpi=160, bbox_inches="tight"),
    )

    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"