"""
Docstring for downloadFRDData

Usage:
python download_intermagnet.py \
  --obs FRD \
  --start 2024-12-11 \
  --end 2024-12-20 \
  --cadence second \
  --publication "Best available" \
  --orientation native \
  --format Iaga2002

python ./downloadFRDData.py --obs FRD --start 2024-11-30 --end 2025-06-30 --cadence second --publication "Best available" --orientation
 native --format Iaga2002
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import requests

BASE_URL = "https://imag-data.bgs.ac.uk/GIN_V1/GINServices"

def daterange(d0: date, d1: date):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)

def download_one_day(
    session: requests.Session,
    out_dir: Path,
    obs: str,
    pub_state: str,
    orientation: str,
    cadence: str,   # "second" or "minute" (matches the web form parameter)
    fmt: str,       # "Iaga2002", "json", "xml", "wdc", "imagcdf", ...
    day: date,
    retries: int = 5,
    timeout_s: int = 120,
):
    params = {
        "Request": "GetData",
        "observatoryIagaCode": obs,
        "publicationState": pub_state,
        "orientation": orientation,
        "samplesPerDay": cadence,
        "dataStartDate": day.isoformat(),  # YYYY-MM-DD
        "dataDuration": 1,                 # 1 day
        "format": fmt,
        "testObsys": 0,
    }

    # nice filename
    safe_pub = pub_state.replace(" ", "_").replace("+", "_")
    ext = fmt.lower()
    # out_path = out_dir / f"{obs}_{day.isoformat()}_{cadence}_{safe_pub}_{orientation}.{ext}"
    station = obs.lower()              # frd
    datestr = day.strftime("%Y%m%d")   # 20241130

    out_path = out_dir / f"{station}{datestr}psec.sec"

    for attempt in range(1, retries + 1):
        try:
            with session.get(BASE_URL, params=params, stream=True, timeout=timeout_s) as r:
                if r.status_code == 404:
                    print(f"[{day}] 404 (no data / embargo / unavailable). Skipping.")
                    return None
                r.raise_for_status()

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            print(f"[{day}] saved -> {out_path}")
            return out_path

        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == retries:
                raise
            backoff = min(60, 2 ** (attempt - 1))
            print(f"[{day}] network issue ({e}). retry {attempt}/{retries} in {backoff}s...")
            time.sleep(backoff)

        except requests.HTTPError as e:
            # 400 usually means a bad parameter combination; 500 means server error
            if attempt == retries or (e.response is not None and e.response.status_code == 400):
                raise
            backoff = min(60, 2 ** (attempt - 1))
            code = e.response.status_code if e.response is not None else "?"
            print(f"[{day}] HTTP {code}. retry {attempt}/{retries} in {backoff}s...")
            time.sleep(backoff)

def main():
    ap = argparse.ArgumentParser(description="Download INTERMAGNET data from BGS GIN over a date range.")
    ap.add_argument("--obs", default="FRD", help="3-letter IAGA observatory code (e.g., FRD).")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive).")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (inclusive).")
    ap.add_argument("--out", default="frd_data", help="Output directory.")
    ap.add_argument("--publication", default="Best available", help="e.g., Best available / Definitive / Quasi-definitive / Provisional / Variation")
    ap.add_argument("--orientation", default="native", help="e.g., native, XYZF, HDZF, DIFF, ...")
    ap.add_argument("--cadence", default="second", choices=["second", "minute"], help="Second or minute data.")
    ap.add_argument("--format", default="Iaga2002",
                    help="Data format (e.g., Iaga2002, json, xml, wdc, imagcdf, covJson).")
    args = ap.parse_args()

    d0 = date.fromisoformat(args.start)
    d1 = date.fromisoformat(args.end)
    out_dir = Path(args.out)

    with requests.Session() as s:
        s.headers.update({"User-Agent": "intermagnet-downloader/1.0 (+https://imag-data.bgs.ac.uk/)"})
        for day in daterange(d0, d1):
            download_one_day(
                session=s,
                out_dir=out_dir,
                obs=args.obs,
                pub_state=args.publication,
                orientation=args.orientation,
                cadence=args.cadence,
                fmt=args.format,
                day=day,
            )

if __name__ == "__main__":
    main()
