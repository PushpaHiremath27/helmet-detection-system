"""
Entrypoint for the Helmet Detection app.

Launches the Streamlit frontend (frontend/app.py), which reads trained
weights from runs/detect/*/weights/best.pt and delegates all model/inference
logic to backend/.

Usage:
    python main.py
    # equivalent to: streamlit run frontend/app.py
"""
import sys
from pathlib import Path

from streamlit.web import cli as stcli

ROOT = Path(__file__).resolve().parent
FRONTEND_APP = ROOT / "frontend" / "app.py"


def main():
    sys.argv = ["streamlit", "run", str(FRONTEND_APP), *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
