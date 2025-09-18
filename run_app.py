#!/usr/bin/env python3
"""
Quick start script for the Spotify Mood Recommender.

This script provides easy commands to run different parts of the application.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_pipeline():
    """Run the complete ML pipeline."""
    print("🚀 Running complete Spotify Mood Recommender pipeline...")
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
        print("✅ Pipeline completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)


def run_streamlit():
    """Run the Streamlit web application."""
    print("🌐 Starting Streamlit web application...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "src/app/ui.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit failed to start: {e}")
        sys.exit(1)


def run_tests():
    """Run the test suite."""
    print("🧪 Running test suite...")
    try:
        subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=True)
        print("✅ All tests passed!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        sys.exit(1)


def run_notebooks():
    """Launch Jupyter notebooks."""
    print("📓 Starting Jupyter notebooks...")
    try:
        subprocess.run([sys.executable, "-m", "jupyter", "notebook", "notebooks/"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Jupyter failed to start: {e}")
        sys.exit(1)


def check_requirements():
    """Check if all requirements are installed."""
    print("🔍 Checking requirements...")
    try:
        import pandas
        import numpy
        import sklearn
        import matplotlib
        import plotly
        import streamlit
        import umap
        import joblib
        import tqdm
        print("✅ All requirements are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Spotify Mood Recommender - Quick Start")
    parser.add_argument(
        "command",
        choices=["pipeline", "app", "test", "notebooks", "check"],
        help="Command to run: pipeline (run ML pipeline), app (start Streamlit), test (run tests), notebooks (start Jupyter), check (check requirements)"
    )
    
    args = parser.parse_args()
    
    if args.command == "pipeline":
        run_pipeline()
    elif args.command == "app":
        run_streamlit()
    elif args.command == "test":
        run_tests()
    elif args.command == "notebooks":
        run_notebooks()
    elif args.command == "check":
        check_requirements()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🎧 Spotify Mood Recommender - Quick Start")
        print("=" * 50)
        print("Available commands:")
        print("  python run_app.py pipeline  - Run complete ML pipeline")
        print("  python run_app.py app       - Start Streamlit web app")
        print("  python run_app.py test      - Run test suite")
        print("  python run_app.py notebooks - Start Jupyter notebooks")
        print("  python run_app.py check     - Check requirements")
        print("=" * 50)
        sys.exit(0)
    
    main()
