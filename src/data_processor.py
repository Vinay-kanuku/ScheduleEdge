# modules/data_processor.py
import pandas as pd
import streamlit as st

class DataProcessor:
    """Processes input Excel files to extract subject and faculty data"""
    
    def __init__(self, uploaded_file):
        """Initialize with the uploaded file"""
        self.uploaded_file = uploaded_file
    
    def process_file(self):
        """Process the uploaded Excel file and extract data blocks"""
        xls = pd.ExcelFile(self.uploaded_file)
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0], header=None)
        
        semesters = {'II': 'II-Year I-Semester', 'III': 'III-Year I-Semester', 'IV': 'IV-Year I-Semester'}
        section_names = ['A', 'B', 'C', 'CSIT']
        blocks = {}
        
        # Enhanced block detection logic
        for sem, marker in semesters.items():
            start_matches = df.index[df[0].astype(str).str.contains(marker, na=False)]
            
            if start_matches.empty:
                # Try looser matching
                alt_marker = f"{sem}-Year"
                start_matches = df.index[df[0].astype(str).str.contains(alt_marker, na=False)]
                if start_matches.empty:
                    st.error(f"Could not find data for {sem} year.")
                    continue
            
            start_idx = start_matches[0] + 1
            
            # Find the next section marker or the end of the file
            next_markers = []
            for s, m in semesters.items():
                if s != sem:  # Look for other markers
                    matches = df.index[(df.index > start_idx) & df[0].astype(str).str.contains(m, na=False)]
                    if not matches.empty:
                        next_markers.append(matches[0])
            
            # Also consider rows with many NaN values as potential end markers
            end_matches = df.index[(df.index > start_idx) & df[0].isna()]
            if not end_matches.empty:
                next_markers.append(end_matches[0])
            
            # If we're at the last section, use the end of the dataframe
            if not next_markers:
                end_idx = len(df)
            else:
                end_idx = min(next_markers)
            
            table = df.iloc[start_idx:end_idx].reset_index(drop=True)
            
            # Find header row containing "S.No"
            header_rows = table.index[table[0].astype(str).str.contains('S.No', na=False)]
            if header_rows.empty:
                st.error(f"Could not find 'S.No' header in {sem} year data.")
                continue
                
            header_row = header_rows[0]
            table.columns = table.iloc[header_row]
            table = table[header_row+1:].dropna(subset=['S.No'])
            
            # Process each section
            for i, section in enumerate(section_names, start=4):
                if i >= len(table.columns):
                    continue
                    
                col_name = table.columns[i]
                if pd.isna(col_name):
                    continue
                    
                section_key = f"{sem}-{section}"
                
                # Check if the required columns exist
                if 'Subject Code' not in table.columns or 'Subject Name' not in table.columns:
                    st.error(f"Required columns missing in {sem} year data.")
                    continue
                    
                blocks[section_key] = table[['Subject Code', 'Subject Name', col_name]].rename(
                    columns={col_name: 'Faculty Name'}
                )
        
        return blocks