"""Descarga de la curva de tipos del BCE (bonos soberanos AAA de la zona euro)."""

import re

import pandas as pd

from ccr_engine.curves import YieldCurve

ECB_URL = "https://data-api.ecb.europa.eu/service/data/YC/{key}"
SERIES_PREFIX = "B.U2.EUR.4F.G_N_A.SV_C_YM."
DEFAULT_TENORS = ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"]


def parse_tenor(tenor):
    """Convierte un plazo de texto en años: '6M' -> 0.5, '1Y6M' -> 1.5, '10Y' -> 10."""
    match = re.fullmatch(r"(?:(\d+)Y)?(?:(\d+)M)?", tenor.strip().upper())
    if not tenor.strip() or match is None:
        raise ValueError(f"Plazo no válido: {tenor!r}")
    years = int(match.group(1) or 0)
    months = int(match.group(2) or 0)
    return years + months / 12


def build_ecb_url(tenors=DEFAULT_TENORS, last_n=1):
    """URL de la API del BCE para descargar los plazos pedidos en CSV."""
    key = SERIES_PREFIX + "+".join(f"SR_{t}" for t in tenors)
    return ECB_URL.format(key=key) + f"?lastNObservations={last_n}&format=csvdata"


def fetch_ecb_spot_rates(tenors=DEFAULT_TENORS, last_n=1, source=None):
    """Lee los tipos cupón cero del BCE.

    source puede ser la ruta a un CSV descargado a mano desde el navegador.
    Si es None, se descarga directamente de la API del BCE.
    Devuelve un DataFrame con columnas: date, tenor, years, rate (en decimal).
    """
    raw = pd.read_csv(build_ecb_url(tenors, last_n) if source is None else source)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["TIME_PERIOD"]),
            "tenor": raw["DATA_TYPE_FM"].str.removeprefix("SR_"),
            "rate": raw["OBS_VALUE"] / 100,  # el BCE publica en %
        }
    )
    df["years"] = df["tenor"].map(parse_tenor)
    return df.sort_values(["date", "years"]).reset_index(drop=True)


def curve_from_dataframe(df, date=None):
    """Construye una YieldCurve a partir del DataFrame de fetch_ecb_spot_rates.

    Si no se indica fecha, usa la más reciente.
    """
    date = df["date"].max() if date is None else pd.Timestamp(date)
    day = df[df["date"] == date].sort_values("years")
    if day.empty:
        raise ValueError(f"No hay datos para la fecha {date.date()}")
    return YieldCurve(day["years"].to_numpy(), day["rate"].to_numpy())