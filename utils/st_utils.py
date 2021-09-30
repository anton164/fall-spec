import streamlit as st
import os

def st_select_file(
    label,
    data_dir, 
    only_files=True, 
    only_dirs=True,
    extension=""
) -> str:

    def filter_files(filename: str):
        if filename.endswith(extension):
            if only_dirs and os.path.isdir(os.path.join(data_dir, filename)):
                return True
            elif only_files and not os.path.isdir(os.path.join(data_dir, filename)):
                return True
        return False

    files = [f for f in os.listdir(data_dir) if filter_files(f)]
    selected_file = st.selectbox(
        label, options=files
    )
    return os.path.join(data_dir, selected_file)
