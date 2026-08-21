"""
Verspätungsmarkt – historische Verspätungsstatistik pro Route+Uhrzeit,
inkl. Tagesverlauf (für "Goldpick"-Karte im Stil von delaybahn.com)
-----------------------------------------------------------------
Lädt echte DB-Verspätungsdaten von HuggingFace (piebro/deutsche-bahn-data,
CC BY 4.0) und berechnet pro Route (Start -> Ziel, gebündelt nach
Abfahrtsstunde):
    - Gesamt-Kennzahlen: samples, pct (Anteil >=20 Min oder Ausfall), avgDelay
    - "days": Tagesreihe der letzten 30 Tage (Verspätung in Min pro Tag)
    - cancelledCount: Anzahl (Teil-)Ausfälle in den letzten 30 Tagen

Spaltenschema (Stand: bestätigt live via GitHub-Actions-Fehlermeldung, 2026-08):
    station_name, xml_station_name, eva, train_number, line_number,
    final_destination_station, delay_in_min, time,
    arrival_is_canceled, departure_is_canceled, train_type,
    train_line_ride_id, train_line_station_num,
    arrival_planned_time, arrival_change_time,
    departure_planned_time, departure_change_time, id

Nutzung:
    pip install pandas pyarrow requests
    python aggregate_delays.py
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://huggingface.co/datasets/piebro/deutsche-bahn-data/resolve/main/monthly_processed_data/data-{month}.parquet"
ONLY_CATEGORIES = {"ICE", "IC"}
MONTHS_BACK = 2
DAYS_WINDOW = 30
MIN_SAMPLES_ROUTE = 15
MIN_SAMPLES_TRAIN = 20
OUTPUT_PATH = Path("delay_stats.json")

ORIGIN_STATIONS = [
    "Bremen Hbf", "Osnabrück Hbf", "Hannover Hbf", "Hamburg Hbf", "Berlin Hbf",
    "Köln Hbf", "Frankfurt (Main) Hbf", "Stuttgart Hbf", "München Hbf",
    "Leipzig Hbf", "Dresden Hbf", "Düsseldorf Hbf",
]
# Berlin Hbf hat zwei Ebenen mit unterschiedlichen IRIS-Namen: Fernverkehr
# (ICE/IC) fährt fast nur an den unterirdischen Gleisen 1-8, die separat als
# "Berlin Hbf (tief)" geführt werden. Beide Schreibweisen werden hier auf
# denselben Anzeigenamen "Berlin Hbf" normalisiert.
STATION_NAME_ALIASES = {
    "Berlin Hbf (tief)": "Berlin Hbf",
    "Berlin Hbf (S-Bahn)": "Berlin Hbf",
}
# Ziele werden auf dieselbe Liste großer Hauptbahnhöfe beschränkt — sonst
# entstehen tausende Kombinationen mit kleinen/seltenen Zielbahnhöfen, die
# die Datei unnötig aufblähen und praktisch kaum gesucht werden.
DEST_STATIONS = set(ORIGIN_STATIONS)

NEEDED_COLUMNS = [
    "station_name", "train_type", "train_number", "line_number",
    "train_line_ride_id", "train_line_station_num",
    "departure_planned_time", "departure_change_time",
    "arrival_planned_time", "arrival_change_time",
    "arrival_is_canceled", "departure_is_canceled",
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

    df["arrival_delay_min"] = (
        (df["arrival_change_time"] - df["arrival_planned_time"]).dt.total_seconds() / 60
    ).fillna(0).round().astype(int)
    df["departure_delay_min"] = (
        (df["departure_change_time"] - df["departure_planned_time"]).dt.total_seconds() / 60
    ).fillna(0).round().astype(int)
    df["is_canceled"] = df["arrival_is_canceled"].fillna(False) | df["departure_is_canceled"].fillna(False)

    berlin_variants = sorted(df.loc[df["station_name"].str.contains("erlin", na=False), "station_name"].unique())
    if berlin_variants:
        print(f"  Gefundene Berlin-Schreibweisen in {month}: {berlin_variants}")

    df["station_name"] = df["station_name"].replace(STATION_NAME_ALIASES)

    return df


def matched_route_legs(all_df):
    origin_df = all_df[all_df["station_name"].isin(ORIGIN_STATIONS)].copy()
    origin_df = origin_df.dropna(subset=["departure_planned_time"])
    origin_df["hour"] = origin_df["departure_planned_time"].dt.hour
    origin_df["day"] = origin_df["departure_planned_time"].dt.date
    origin_df["time_str"] = origin_df["departure_planned_time"].dt.strftime("%H:%M")
    origin_df = origin_df[[
        "train_line_ride_id", "train_line_station_num", "station_name", "hour", "day", "time_str",
    ]].rename(columns={"station_name": "origin_name", "train_line_station_num": "origin_num"})
    # Das Datenset wird per Snapshot-Polling erzeugt — derselbe Halt kann
    # mehrfach auftauchen (aus verschiedenen Abfragen). Pro Fahrt+Bahnhof nur
    # eine Zeile behalten, sonst tauchen Tage im Chart mehrfach identisch auf.
    origin_df = origin_df.drop_duplicates(subset=["train_line_ride_id", "origin_name"])

    dest_df = all_df[all_df["station_name"].isin(DEST_STATIONS)].copy()
    dest_df = dest_df[[
        "train_line_ride_id", "train_line_station_num", "station_name",
        "arrival_delay_min", "is_canceled",
    ]].rename(columns={"station_name": "dest_name", "train_line_station_num": "dest_num"})
    dest_df = dest_df.drop_duplicates(subset=["train_line_ride_id", "dest_name"], keep="last")

    merged = origin_df.merge(dest_df, on="train_line_ride_id")
    merged = merged[merged["dest_num"] > merged["origin_num"]]
    return merged


def build_route_stats(all_df):
    merged = matched_route_legs(all_df)
    merged["is_problem"] = merged["is_canceled"] | (merged["arrival_delay_min"] >= 20)

    cutoff = date.today() - timedelta(days=DAYS_WINDOW)

    routes = {}
    # Gruppierung nach EXAKTER Abfahrtszeit (nicht nur Stunde) — sonst werden auf
    # belebten Strecken mehrere verschiedene Züge derselben Stunde vermischt,
    # was zu falschen "ca."-Zeiten und mehreren Balken pro Tag im Chart führt.
    for (origin_name, dest_name, time_str), grp in merged.groupby(["origin_name", "dest_name", "time_str"]):
        samples = len(grp)
        if samples < MIN_SAMPLES_ROUTE:
            continue

        pct = round(float(grp["is_problem"].mean()) * 100, 1)
        avg_delay = round(float(grp["arrival_delay_min"].mean()), 1)

        recent = grp[grp["day"] >= cutoff].drop_duplicates(subset=["day"], keep="last").sort_values("day")
        cancelled_count = int(recent["is_canceled"].sum())

        days = None
        if avg_delay >= 10 or pct >= 20:
            days = []
            for _, row in recent.iterrows():
                if bool(row["is_canceled"]):
                    days.append({"date": row["day"].isoformat(), "delay": None, "cancelled": True})
                else:
                    days.append({"date": row["day"].isoformat(), "delay": int(row["arrival_delay_min"]), "cancelled": False})

        key = f"{origin_name}|{dest_name}|{time_str}"
        routes[key] = {
            "samples": int(samples),
            "pct": pct,
            "avgDelay": avg_delay,
            "typicalTime": time_str,
            "cancelledCount": cancelled_count,
            "days": days,
        }
    return routes


def build_train_number_stats(all_df):
    df = all_df.copy()
    df["is_problem"] = df["is_canceled"] | (df["arrival_delay_min"] >= 20)
    grouped = (
        df.groupby(["train_type", "train_number"])
        .agg(
            samples=("is_problem", "size"),
            problem_rate=("is_problem", "mean"),
            avg_delay=("arrival_delay_min", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["samples"] >= MIN_SAMPLES_TRAIN]

    trains = {}
    for _, row in grouped.iterrows():
        key = f"{row['train_type']} {row['train_number']}"
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

    # Sicherung: eine (fast) leere Datei würde stillschweigend gute alte Daten
    # überschreiben. Lieber der Job schlägt sichtbar fehl, als dass die App
    # danach plötzlich "0 Startbahnhöfe" zeigt.
    if len(routes) < 50 or len(trains) < 20:
        print(f"WARNUNG: nur {len(routes)} Routen / {len(trains)} Zugnummern gefunden — das ist verdächtig wenig.")
        print("Breche ab, ohne delay_stats.json zu überschreiben. Prüfe Spaltennamen/Stationsnamen im Datenset.")
        sys.exit(1)

    output = {
        "generatedAt": date.today().isoformat(),
        "monthsIncluded": months,
        "daysWindow": DAYS_WINDOW,
        "onlyCategories": sorted(ONLY_CATEGORIES),
        "minSamplesRoute": MIN_SAMPLES_ROUTE,
        "minSamplesTrain": MIN_SAMPLES_TRAIN,
        "note": (
            "pct = Anteil der Fahrten mit Ausfall oder >=20 Min Ankunftsverspätung. "
            "'routes' misst die Verspätung am tatsächlichen Zielbahnhof derselben Fahrt, "
            "inkl. Tagesreihe 'days' der letzten 30 Tage für Goldpick-Charts. "
            "'trains' ist ein gröberer Fallback pro Zugnummer über alle Teilstrecken. "
            "Echte historische Daten von Deutsche Bahn (CC BY 4.0) via "
            "huggingface.co/datasets/piebro/deutsche-bahn-data"
        ),
        "routes": routes,
        "trains": trains,
    }

    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Fertig: {len(routes)} Route+Stunde-Kombinationen, {len(trains)} Zugnummern in {OUTPUT_PATH} ({size_mb:.1f} MB).")
    if size_mb > 90:
        print("WARNUNG: Datei ist nahe am GitHub-Limit von 100 MB — Push könnte fehlschlagen.")
        sys.exit(1)


if __name__ == "__main__":
    main()
