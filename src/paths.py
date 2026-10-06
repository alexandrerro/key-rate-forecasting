"""
Общие пути проекта. Кладётся в src/ рядом со скриптами.

В каждом скрипте вместо строк вида
    DATA_PATH = r"C:\\Users\\...\\dataset_monthly.csv"
    FIG_DIR = "figures000"
пишем:
    from paths import DATA_DIR, FIG_DIR, RESULTS_DIR
    DATA_PATH = DATA_DIR / "dataset_monthly.csv"

Пути считаются от расположения этого файла, поэтому скрипты работают
независимо от того, из какой папки их запускают.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

for _d in (RESULTS_DIR, FIG_DIR):
    _d.mkdir(exist_ok=True)

# Ключи берём из переменных окружения; если есть файл .env и установлен
# python-dotenv, он подхватится автоматически.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass
