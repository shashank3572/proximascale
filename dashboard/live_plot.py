"""
dashboard/live_plot.py — Streamlit live dashboard for ProximaScale.
Person D owns this file.

STATUS: Semester 2 deliverable — not yet implemented.

Planned features:
  - Live line chart: actual CPU% vs predicted CPU%
  - Scaling event markers (scale_up / scale_down timestamps)
  - Anomaly flag highlights (red markers)
  - Current replica count display

Run (Semester 2):
  streamlit run dashboard/live_plot.py
"""

# ── Semester 2 placeholder ────────────────────────────────────────────────────
# Remove this block and implement below once Semester 2 begins.

import sys

def main():
    print("Dashboard is a Semester 2 deliverable.")
    print("Run the system with: python main.py --simulate")
    sys.exit(0)

if __name__ == "__main__":
    main()

# ── Semester 2 implementation (skeleton) ──────────────────────────────────────
# Uncomment and build out in Semester 2.
#
# import streamlit as st
# import pandas as pd
# import time
# from monitoring.storage import read_last_n
# from model.predict import predict
#
# st.set_page_config(page_title="ProximaScale Dashboard", layout="wide")
# st.title("ProximaScale — Live Workload Dashboard")
#
# placeholder = st.empty()
# while True:
#     records = read_last_n(50)
#     df = pd.DataFrame([r.to_dict() for r in records])
#     with placeholder.container():
#         st.line_chart(df.set_index("timestamp")[["cpu_percent"]])
#     time.sleep(10)
