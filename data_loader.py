import numpy as np
import openpyxl

EXCEL_PATH = str(__import__("pathlib").Path(__file__).parent / "data" / "real_light_source_spds.xlsx")
WL_OUT = np.arange(380, 781, 1, dtype=float)  # 401 points, 380–780 nm @ 1 nm


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Load SPDs and CCTs from the dataset Excel file.

    Returns
    -------
    ccts : np.ndarray, shape (N,)
    spds : np.ndarray, shape (N, 401)  — normalized to max=1, resampled to 380–780 nm @ 1 nm
    """
    wb = openpyxl.load_workbook(EXCEL_PATH, read_only=True)

    ws_perf = wb["Performance Data"]
    rows_perf = list(ws_perf.iter_rows(values_only=True))
    cct_by_number: dict[int, float] = {}
    for row in rows_perf[2:]:
        num, cct = row[0], row[11]
        if num is not None and cct is not None and float(cct) > 0:
            cct_by_number[int(num)] = float(cct)

    ws_spds = wb["SPDs"]
    rows_spds = list(ws_spds.iter_rows(values_only=True))

    ccts, spds = [], []
    for row in rows_spds[2:]:
        num = row[0]
        if num is None or int(num) not in cct_by_number:
            continue

        start_wl, end_wl, interval = row[8], row[9], row[10]
        if any(v is None for v in (start_wl, end_wl, interval)):
            continue

        n_points = int(round((end_wl - start_wl) / interval)) + 1
        vals = [row[11 + i] for i in range(n_points)]
        if any(v is None for v in vals):
            continue

        spd_raw = np.array(vals, dtype=float)
        wl_source = np.linspace(start_wl, end_wl, n_points)
        spd_resampled = np.interp(WL_OUT, wl_source, spd_raw)

        peak = spd_resampled.max()
        if peak <= 0:
            continue
        spd_resampled /= peak

        ccts.append(cct_by_number[int(num)])
        spds.append(spd_resampled)

    wb.close()
    return np.array(ccts, dtype=float), np.array(spds, dtype=float)
