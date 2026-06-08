"""
Shared download utility for ENTSO-E scripts.
"""

import pandas as pd

YEARS = [
    ('2022-01-01', '2023-01-01'),
    ('2023-01-01', '2024-01-01'),
    ('2024-01-01', '2025-01-06'),
]


def download_in_chunks(query_func, *args):
    """Call any ENTSO-E query function year by year and concatenate results."""
    chunks = []
    for s, e in YEARS:
        start = pd.Timestamp(s, tz='Europe/Berlin')
        end = pd.Timestamp(e, tz='Europe/Berlin')
        chunk = query_func(*args, start=start, end=end)
        chunks.append(chunk)
        print(f"  Downloaded {s} to {e}")
    return pd.concat(chunks)
