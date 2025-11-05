#!/usr/bin/env python3
"""Run Streamlit app"""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])

