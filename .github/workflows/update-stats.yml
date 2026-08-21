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
    "Berlin Hauptbahnhof": "Berlin Hbf",
}
# Ziele werden auf dieselbe Liste großer Hauptbahnhöfe beschränkt — sonst
# entstehen tausende Kombinationen mit kleinen/seltenen Zielbahnhöfen, die
# die Datei unnötig aufblähen und praktisch kaum gesucht werden.
DEST_STATIONS = set(ORIGIN_STATIONS)

# Für die Diagnose: ein Stichwort pro Stadt, um alle Schreibweisen im
# Datenset zu finden, unabhängig davon, wie ORIGIN_STATIONS sie benennt.
DIAGNOSTIC_KEYWORDS = [
    "Bremen", "Osnabrück", "Hannover", "Hamburg", "Berlin", "Köln",
    "Frankfurt", "Stuttgart", "München", "Leipzig", "Dresden", "Düsseldorf",
]

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

    all_names = df["station_name"].dropna().unique()
    for kw in DIAGNOSTIC_KEYWORDS:
        kw_matches = [n for n in all_names if kw.lower() in n.lower()]
        hbf_matches = [n for n in kw_matches if "hbf" in n.lower() or "hauptbahnhof" in n.lower()]
        shown = sorted(hbf_matches) if hbf_matches else sorted(kw_matches)
        if shown:
            print(f"  [{kw}] {shown[:8]}")

    df["station_name"] = df["station_name"].replace(STATION_NAME_ALIASES)

    return df


def matched_route_legs(all_df):
    origin_df = all_df[all_df["station_name"].isin(ORIGIN_STATIONS)].copy()
    print(f"    origin_df nach Stationsfilter: {len(origin_df)} Zeilen")
    origin_df = origin_df.dropna(subset=["departure_planned_time"])
    print(f"    origin_df nach dropna(departure_planned_time): {len(origin_df)} Zeilen")
    origin_df["hour"] = origin_df["departure_planned_time"].dt.hour
    origin_df["day"] = origin_df["departure_planned_time"].dt.date
    origin_df["time_str"] = origin_df["departure_planned_time"].dt.strftime("%H:%M")
    origin_df = origin_df[[
        "train_line_ride_id", "train_line_station_num", "station_name", "hour", "day", "time_str",
    ]].rename(columns={"station_name": "origin_name", "train_line_station_num": "origin_num"})
    origin_df = origin_df.drop_duplicates(subset=["train_line_ride_id", "origin_name"])
    print(f"    origin_df nach Dedup: {len(origin_df)} Zeilen")

    dest_df = all_df[all_df["station_name"].isin(DEST_STATIONS)].copy()
    print(f"    dest_df nach Stationsfilter: {len(dest_df)} Zeilen")
    dest_df = dest_df[[
        "train_line_ride_id", "train_line_station_num", "station_name",
        "arrival_delay_min", "is_canceled",
    ]].rename(columns={"station_name": "dest_name", "train_line_station_num": "dest_num"})
    dest_df = dest_df.drop_duplicates(subset=["train_line_ride_id", "dest_name"], keep="last")
    print(f"    dest_df nach Dedup: {len(dest_df)} Zeilen")

    merged = origin_df.merge(dest_df, on="train_line_ride_id")
    print(f"    merged nach Join: {len(merged)} Zeilen")
    merged = merged[merged["dest_num"] > merged["origin_num"]]
    print(f"    merged nach dest_num > origin_num: {len(merged)} Zeilen")
    return merged


def build_route_stats(all_df):
    merged = matched_route_legs(all_df)
    merged["is_problem"] = merged["is_canceled"] | (merged["arrival_delay_min"] >= 20)

    cutoff = date.today() - timedelta(days=DAYS_WINDOW)

    routes = {}
    total_groups = 0
    kept_groups = 0
    # Stunden-Gruppierung (nicht exakte Minute) — direkte Verbindungen zwischen
    # zwei großen Bahnhöfen ohne Umstieg sind seltener als gedacht, exakte
    # Minute wäre zu fein für genug Stichprobengröße. Mehrere Züge derselben
    # Stunde werden im Tages-Chart pro Kalendertag gemittelt (ein Balken/Tag).
    for (origin_name, dest_name, hour), grp in merged.groupby(["origin_name", "dest_name", "hour"]):
        total_groups += 1
        samples = len(grp)
        if samples < MIN_SAMPLES_ROUTE:
            continue
        kept_groups += 1

        pct = round(float(grp["is_problem"].mean()) * 100, 1)
        avg_delay = round(float(grp["arrival_delay_min"].mean()), 1)
        typical_time = grp["time_str"].mode().iloc[0] if not grp["time_str"].mode().empty else f"{int(hour):02d}:00"

        recent = grp[grp["day"] >= cutoff]
        cancelled_count = 0
        days = None
        if avg_delay >= 10 or pct >= 20:
            days = []
            for day_value, day_grp in recent.groupby("day"):
                day_cancelled = bool(day_grp["is_canceled"].any())
                if day_cancelled:
                    cancelled_count += 1
                    days.append({"date": day_value.isoformat(), "delay": None, "cancelled": True})
                else:
                    days.append({"date": day_value.isoformat(), "delay": round(float(day_grp["arrival_delay_min"].mean())), "cancelled": False})
            days.sort(key=lambda d: d["date"])
        else:
            cancelled_count = int(recent.groupby("day")["is_canceled"].any().sum())

        key = f"{origin_name}|{dest_name}|{int(hour)}"
        routes[key] = {
            "samples": int(samples),
            "pct": pct,
            "avgDelay": avg_delay,
            "typicalTime": typical_time,
            "cancelledCount": cancelled_count,
            "days": days,
        }
    print(f"    Gruppen gesamt: {total_groups}, davon >= {MIN_SAMPLES_ROUTE} Beobachtungen: {kept_groups}")
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
