from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    st.set_page_config(page_title="ChurnCube", page_icon="📊", layout="wide")
    st.title("ChurnCube")
    st.caption(f"Workspace: `{ROOT.name}`")
    st.write("Dashboard scaffolding is in place. Feature tabs will be added in follow-up commits.")


if __name__ == "__main__":
    main()
