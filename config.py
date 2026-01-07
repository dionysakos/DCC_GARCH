from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class Config:
    tickers: List[str] = ("AAPL", "NVDA")
    start_date: str = "2020-01-01"
    end_date: str = "2024-06-01"
    scale: float = 100.0  # returns in percent
    dcc_init: Tuple[float, float] = (0.02, 0.97)
