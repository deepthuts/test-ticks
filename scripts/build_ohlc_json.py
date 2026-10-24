#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np


def resolve_files(files_file: str, repo_root: str):
    with open(files_file, "r", encoding="utf-8") as f:
        entries = [line.strip() for line in f if line.strip()]

    resolved = []
    for entry in entries:
        candidate = os.path.join(repo_root, entry)
        if os.path.exists(candidate):
            resolved.append(candidate)
    return resolved


def load_tick_file(path: str) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict):
            if "data" in obj and isinstance(obj["data"], list):
                df = pd.DataFrame(obj["data"])
            else:
                df = pd.DataFrame([obj])
        elif isinstance(obj, list):
            df = pd.DataFrame(obj)
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
    else:
        raise ValueError(f"Unsupported file type: {path}")

    if df.empty:
        return df

    timestamp_candidates = ["timestamp", "time", "datetime", "date", "ts", "event_time"]
    ts_col = next((c for c in timestamp_candidates if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"Missing timestamp column in {path}")

    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    # Spread and pricing
    if "bid" in df.columns and "ask" in df.columns:
        df["mid_price"] = (pd.to_numeric(df["bid"], errors="coerce") + pd.to_numeric(df["ask"], errors="coerce")) / 2.0
        df["spread"] = pd.to_numeric(df["ask"], errors="coerce") - pd.to_numeric(df["bid"], errors="coerce")
    elif "mid" in df.columns:
        df["mid_price"] = pd.to_numeric(df["mid"], errors="coerce")
        df["spread"] = np.nan
    elif "price" in df.columns:
        df["mid_price"] = pd.to_numeric(df["price"], errors="coerce")
        df["spread"] = np.nan
    elif "last" in df.columns:
        df["mid_price"] = pd.to_numeric(df["last"], errors="coerce")
        df["spread"] = np.nan
    else:
        raise ValueError(f"No usable price columns found in {path}")

    # Volume logic
    if "tick_volume" in df.columns:
        df["tick_volume"] = pd.to_numeric(df["tick_volume"], errors="coerce").fillna(0)
    elif "volume" in df.columns:
        df["tick_volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    elif "qty" in df.columns:
        df["tick_volume"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)
    else:
        df["tick_volume"] = 1

    return df[["timestamp", "mid_price", "spread", "tick_volume"]]


def aggregate_ohlc(df: pd.DataFrame, interval_seconds: int, symbol: str):
    df = df.sort_values("timestamp").copy()
    df["bucket"] = df["timestamp"].dt.floor(f"{interval_seconds}s")

    # One spread value per bar: last valid spread in the bucket
    spread_last = (
        df.groupby("bucket")["spread"]
        .apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
        .reset_index(name="spread")
    )

    ohlc = (
        df.groupby("bucket", as_index=False)
        .agg(
            open_price=("mid_price", "first"),
            high_price=("mid_price", "max"),
            low_price=("mid_price", "min"),
            close_price=("mid_price", "last"),
            volume=("tick_volume", "sum"),
        )
    )

    ohlc = ohlc.merge(spread_last, on="bucket", how="left")

    ohlc["timestamp"] = pd.to_datetime(ohlc["bucket"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    ohlc["symbol"] = symbol
    ohlc["interval_seconds"] = interval_seconds

    result = ohlc[
        [
            "timestamp",
            "symbol",
            "interval_seconds",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
            "spread",
        ]
    ].rename(columns={
        "open_price": "open",
        "high_price": "high",
        "low_price": "low",
        "close_price": "close",
    })

    return result


def main():
    parser = argparse.ArgumentParser(description="Convert tick files to 30s OHLC JSON")
    parser.add_argument("--files-file", required=True, help="Manifest with changed tick files")
    parser.add_argument("--symbol", required=True, help="Symbol name, e.g. EURUSD")
    parser.add_argument("--interval", type=int, default=30, help="OHLC interval in seconds")
    parser.add_argument("--output-dir", required=True, help="Folder for per-day JSON files")
    args = parser.parse_args()

    repo_root = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
    files = resolve_files(args.files_file, repo_root)

    if not files:
        raise FileNotFoundError(f"No valid tick files found in manifest: {args.files_file}")

    data_frames = []
    for file in files:
        try:
            df = load_tick_file(file)
            if not df.empty:
                data_frames.append(df)
        except Exception as e:
            print(f"Skipping {file}: {e}")

    if not data_frames:
        raise ValueError("No valid tick data found in repo folder.")

    combined = pd.concat(data_frames, ignore_index=True)
       out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    result["day"] = result["timestamp"].str[:10]

    for day, day_df in result.groupby("day"):
        rows = day_df.drop(columns=["day", "symbol", "interval_seconds"])
        rows = rows.astype(object).where(rows.notna(), None)
        payload = {
            "symbol": args.symbol,
            "interval_seconds": args.interval,
            "date": day,
            "generated_at_utc": generated,
            "rows": rows.to_dict(orient="records"),
        }
        with open(out_dir / f"{day}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"), default=str)
        print(f"Saved {day}.json with {len(rows)} candles")

if __name__ == "__main__":
    main()