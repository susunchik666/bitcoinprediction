from __future__ import annotations

from flask import Flask, render_template, request

from services import (
    CandleRequest,
    InvestError,
    fetch_candles,
    candles_to_dataframe,
    sdk_name,
    train_and_predict,
    plot_candles_with_ichimoku_base64,
)

try:
    from app_secrets import TINKOFF_TOKEN
except Exception:
    TINKOFF_TOKEN = ""

app = Flask(__name__)


@app.get("/")
def index():
    return render_template(
        "index.html",
        default_instrument_id="BBG004730N88",
        default_days=120,
        default_interval="4h",
        default_horizon=1,
        default_task="cls",
        default_model="pdt",
        default_train_ratio=0.8,
        sdk=sdk_name(),
    )


@app.post("/run")
def run():
    instrument_id = (request.form.get("instrument_id") or "").strip()
    days_back = int(request.form.get("days_back") or "120")
    interval = (request.form.get("interval") or "4h").strip()

    horizon = int(request.form.get("horizon") or "1")
    task = (request.form.get("task") or "cls").strip()          # "cls" | "reg"
    model = (request.form.get("model") or "pdt").strip()        # "pdt" | "tree"
    train_ratio = float(request.form.get("train_ratio") or "0.8")

    if not TINKOFF_TOKEN:
        return render_template(
            "error.html",
            message="Token not found. Create app_secrets.py next to app.py and set TINKOFF_TOKEN.",
            sdk=sdk_name(),
        ), 500

    try:
        candles = fetch_candles(
            TINKOFF_TOKEN,
            CandleRequest(instrument_id=instrument_id, days_back=days_back, interval=interval),
        )
        df = candles_to_dataframe(candles)

        forecast = train_and_predict(
            df=df,
            horizon=horizon,
            task=task,
            model_name=("tree" if model == "tree" else "pdt"),
            train_ratio=train_ratio,
        )

        chart_uri = plot_candles_with_ichimoku_base64(df, forecast=forecast, horizon=horizon)

        table_html = df.tail(30).to_html(classes="table table-sm", border=0)

        return render_template(
            "result.html",
            instrument_id=instrument_id,
            days_back=days_back,
            interval=interval,
            horizon=horizon,
            task=task,
            model=model,
            chart_uri=chart_uri,
            table_html=table_html,
            sdk=sdk_name(),
            n=len(df),
            forecast=forecast,
        )

    except InvestError as e:
        return render_template("error.html", message=str(e), sdk=sdk_name()), 400


if __name__ == "__main__":
    app.run(debug=True)