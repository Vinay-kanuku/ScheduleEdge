import pandas as pd
import streamlit as st
import re

class DataProcessor:
    """Processes input Excel files to extract subject and faculty data"""

    def __init__(self, uploaded_file):
        """Initialize with the uploaded file"""
        self.uploaded_file = uploaded_file

    def process_file(self):
        """Process the uploaded Excel file and extract data blocks"""
        xls = pd.ExcelFile(self.uploaded_file)
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0], header=None)

        # Clean and normalize
        df = df.map(lambda x: str(x).strip().replace('\n', ' ') if pd.notna(x) else x)

        # Detect semesters
        semesters = {}
        semester_pattern = r'([IVX]+)-Year\s*I-Semester'
        for idx, row in df.iterrows():
            cell = str(row[0]).strip()
            match = re.search(semester_pattern, cell, re.IGNORECASE)
            if match:
                sem = match.group(1)
                semesters[sem] = cell

        if not semesters:
            st.error("No semester headers like 'II-Year I-Semester' found.")
            return {}

        print("Detected Semesters:", semesters)

        blocks = {}

        for sem, marker in semesters.items():
            start_matches = df.index[df[0].astype(str).str.contains(marker, na=False)]
            if start_matches.empty:
                continue
            start_idx = start_matches[0] + 1

            # Next marker = next semester
            next_markers = [i for i in df.index if i > start_idx and any(m in str(df.at[i, 0]) for m in semesters.values())]
            end_idx = min(next_markers) if next_markers else len(df)
            table = df.iloc[start_idx:end_idx].reset_index(drop=True)

            if table.empty or 0 not in table.columns:
                continue

            header_row_index = table.index[table[0].astype(str).str.contains("S.No", case=False, na=False)]
            if header_row_index.empty:
                continue

            header_idx = header_row_index[0]
            table.columns = table.iloc[header_idx]
            table = table[header_idx + 1:].dropna(subset=['S.No'])

            # Find all section columns dynamically
            possible_section_cols = [col for col in table.columns if isinstance(col, str) and col.strip().startswith("Section")]

            for sec_col in possible_section_cols:
                section_name = sec_col.replace("Section-", "").strip()
                section_key = f"{sem}-{section_name}"

                try:
                    # Clean faculty names — extract first word (remove C1/A1 etc.)
                    cleaned = table[['Subject Code', 'Subject Name', sec_col]].copy()
                    cleaned.columns = ['Subject Code', 'Subject Name', 'Faculty Name']
                    cleaned['Faculty Name'] = cleaned['Faculty Name'].apply(self.clean_faculty)
                    blocks[section_key] = cleaned
                except Exception as e:
                    st.warning(f"Skipping {section_key} due to error: {e}")

        print("Extracted Blocks:", blocks.keys())
        return blocks

    def clean_faculty(self, cell):
        """Extract the first meaningful faculty name"""
        if pd.isna(cell):
            return ""
        text = str(cell)
        if ':' in text:
            text = text.split(':')[-1]
        if '/' in text:
            text = text.split('/')[0]
        return text.strip()