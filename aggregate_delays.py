"""
Verspätungsmarkt – historische Verspätungsstatistik pro Route+Uhrzeit
-----------------------------------------------------------------
Lädt die letzten paar Monate der öffentlichen, echten DB-Verspätungsdaten
von HuggingFace (piebro/deutsche-bahn-data, CC BY 4.0) herunter und
berechnet für jede Kombination aus:
    Startbahnhof (einer der ORIGIN_STATIONS) x Zielbahnhof x Abfahrtsstunde
eine echte historische Quote: wie oft kam eine Fahrt, die zu dieser Stunde
in diesem Bahnhof abfuhr und dort auch hielt, am Zielbahnhof >=20 Min zu
spät an oder fiel aus.

Das misst die Verspätung tatsächlich am Ziel (nicht über alle Zwischenhalte
gemittelt), gematcht über die Fahrt-ID (train_line_ride_id) und die
Reihenfolge der Halte (train_line_station_num).

Zusätzlich (als Fallback für Fälle ohne genug Route-Daten): dieselbe Quote
grob pro Zugkategorie+Nummer, unabhängig von der genauen Route.

Nutzung:
    pip install pandas pyarrow requests
    python aggregate_delays.py
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://huggingface.co/datasets/piebro/deutsche-bahn-data/resolve/main/monthly_processed_data/data-{month}.parquet"
ONLY_CATEGORIES = {"ICE", "IC"}
MONTHS_BACK = 3
MIN_SAMPLES_ROUTE = 20     # Mindestbeobachtungen für eine Route+Stunde-Quote
MIN_SAMPLES_TRAIN = 30     # Mindestbeobachtungen für den Zugnummer-Fallback
OUTPUT_PATH = Path("delay_stats.json")

# Startbahnhöfe, für die Route-Statistiken berechnet werden (identisch zur
# STATIONS-Liste in der App). Ziel-Bahnhöfe werden dynamisch aus den Daten
# übernommen, müssen hier nicht gelistet werden.
ORIGIN_STATIONS = [
    "Bremen Hbf", "Osnabrück Hbf", "Hannover Hbf", "Hamburg Hbf", "Berlin Hbf",
    "Köln Hbf", "Frankfurt (Main) Hbf", "Stuttgart Hbf", "München Hbf",
    "Leipzig Hbf", "Dresden Hbf", "Düsseldorf Hbf",
]

NEEDED_COLUMNS = [
    "station_name", "train_type", "train_name", "train_line_ride_id",
    "train_line_station_num", "departure_planned_time", "arrival_planned_time",
    "delay_in_min", "is_canceled",
]


def last_n_completed_months(n):
    today = date.today()
    y, m = today.year, today.month
    months = []
    for _ in range(n):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
        months.append(f"{y:04d}-{m:02d}")
    return months


def download_month(month):
    url = BASE_URL.format(month=month)
    print(f"Lade {url} ...")
    r = requests.get(url, timeout=180)
    if r.status_code != 200:
        print(f"  -> nicht verfügbar ({r.status_code}), überspringe")
        return None
    tmp_path = Path(f"_tmp_{month}.parquet")
    tmp_path.write_bytes(r.content)
    return tmp_path


def load_month(month):
    path = download_month(month)
    if path is None:
        return None
    df = pd.read_parquet(path, columns=NEEDED_COLUMNS)
    path.unlink()
    df = df[df["train_type"].isin(ONLY_CATEGORIES)]
    return df


def build_route_stats(all_df):
    """Route+Stunde -> Quote, gemessen am tatsächlichen Zielhalt derselben Fahrt."""
    origin_df = all_df[all_df["station_name"].isin(ORIGIN_STATIONS)].copy()
    origin_df = origin_df.dropna(subset=["departure_planned_time"])
    origin_df["hour"] = pd.to_datetime(origin_df["departure_planned_time"]).dt.hour
    origin_df = origin_df[[
        "train_line_ride_id", "train_line_station_num", "station_name", "hour",
    ]].rename(columns={
        "station_name": "origin_name",
        "train_line_station_num": "origin_num",
    })

    dest_df = all_df.dropna(subset=["station_name"]).copy()
    dest_df["is_problem"] = dest_df["is_canceled"] | (dest_df["delay_in_min"] >= 20)
    dest_df = dest_df[[
        "train_line_ride_id", "train_line_station_num", "station_name", "is_problem", "delay_in_min",
    ]].rename(columns={
        "station_name": "dest_name",
        "train_line_station_num": "dest_num",
    })

    merged = origin_df.merge(dest_df, on="train_line_ride_id")
    # nur Halte, die NACH dem Startbahnhof in derselben Fahrt kommen
    merged = merged[merged["dest_num"] > merged["origin_num"]]

    grouped = (
        merged.groupby(["origin_name", "dest_name", "hour"])
        .agg(
            samples=("is_problem", "size"),
            problem_rate=("is_problem", "mean"),
            avg_delay=("delay_in_min", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["samples"] >= MIN_SAMPLES_ROUTE]

    routes = {}
    for _, row in grouped.iterrows():
        key = f"{row['origin_name']}|{row['dest_name']}|{int(row['hour'])}"
        routes[key] = {
            "samples": int(row["samples"]),
            "pct": round(float(row["problem_rate"]) * 100, 1),
            "avgDelay": round(float(row["avg_delay"]), 1),
        }
    return routes


def build_train_number_stats(all_df):
    """Fallback: grobe Quote pro Zugkategorie+Nummer, egal auf welcher Teilstrecke."""
    df = all_df.copy()
    df["is_problem"] = df["is_canceled"] | (df["delay_in_min"] >= 20)
    grouped = (
        df.groupby(["train_type", "train_name"])
        .agg(
            samples=("is_problem", "size"),
            problem_rate=("is_problem", "mean"),
            avg_delay=("delay_in_min", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["samples"] >= MIN_SAMPLES_TRAIN]

    trains = {}
    for _, row in grouped.iterrows():
        key = f"{row['train_type']} {row['train_name']}"
        trains[key] = {
            "samples": int(row["samples"]),
            "pct": round(float(row["problem_rate"]) * 100, 1),
            "avgDelay": round(float(row["avg_delay"]), 1),
        }
    return trains


def main():
    months = last_n_completed_months(MONTHS_BACK)
    frames = []
    for month in months:
        df = load_month(month)
        if df is not None:
            frames.append(df)

    if not frames:
        print("Keine Daten gefunden — breche ab, ohne die bestehende Datei zu überschreiben.")
        sys.exit(1)

    all_df = pd.concat(frames, ignore_index=True)

    routes = build_route_stats(all_df)
    trains = build_train_number_stats(all_df)

    output = {
        "generatedAt": date.today().isoformat(),
        "monthsIncluded": months,
        "onlyCategories": sorted(ONLY_CATEGORIES),
        "minSamplesRoute": MIN_SAMPLES_ROUTE,
        "minSamplesTrain": MIN_SAMPLES_TRAIN,
        "note": (
            "pct = Anteil der Fahrten mit Ausfall oder >=20 Min Verspätung. "
            "'routes' misst die Verspätung am tatsächlichen Zielbahnhof derselben Fahrt, "
            "gebündelt nach Startbahnhof, Zielbahnhof und Abfahrtsstunde. "
            "'trains' ist ein gröberer Fallback pro Zugnummer über alle Teilstrecken. "
            "Echte historische Daten von Deutsche Bahn (CC BY 4.0) via "
            "huggingface.co/datasets/piebro/deutsche-bahn-data"
        ),
        "routes": routes,
        "trains": trains,
    }

    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Fertig: {len(routes)} Route+Stunde-Kombinationen, {len(trains)} Zugnummern in {OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
