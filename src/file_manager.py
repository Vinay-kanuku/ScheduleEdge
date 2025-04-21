# modules/file_manager.py
import pandas as pd
from pandas import ExcelWriter
from io import BytesIO
import re

class FileManager:
    """Manages file operations for timetable data"""
    
    def __init__(self, timetables, faculty_timetables):
        """Initialize with timetables and faculty timetables"""
        self.timetables = timetables
        self.faculty_timetables = faculty_timetables
    
    def sanitize_sheet_name(self, name):
        """Sanitize Excel sheet names to avoid errors"""
        if not isinstance(name, str):
            name = str(name)
        # Remove invalid characters and truncate to Excel's limit
        sanitized = re.sub(r'[\[\]\*\?\:/\\]', '', name)[:31]
        return sanitized
    
    def create_complete_timetable_excel(self):
        """Create an Excel file with all timetables"""
        output = BytesIO()
        
        with ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write Section Timetables
            for sec, df in self.timetables.items():
                df.to_excel(writer, sheet_name=self.sanitize_sheet_name(sec))
            
            # Write Faculty Timetables
            for faculty, timetable in self.faculty_timetables.items():
                faculty_name = self.sanitize_sheet_name(faculty)
                if faculty_name == "":
                    faculty_name = "Unknown_Faculty"
                timetable.to_excel(writer, sheet_name=faculty_name)
        
        # Reset pointer to beginning of file
        output.seek(0)
        return output
    
    def create_year_timetable_excel(self, year):
        """Create an Excel file with timetables for a specific year"""
        output = BytesIO()
        
        with ExcelWriter(output, engine='xlsxwriter') as writer:
            for sec, df in self.timetables.items():
                if sec.startswith(year):
                    df.to_excel(writer, sheet_name=self.sanitize_sheet_name(sec))
        
        # Reset pointer to beginning of file
        output.seek(0)
        return output
    
    def create_faculty_timetable_excel(self):
        """Create an Excel file with all faculty timetables"""
        output = BytesIO()
        
        with ExcelWriter(output, engine='xlsxwriter') as writer:
            for faculty, timetable in self.faculty_timetables.items():
                faculty_name = self.sanitize_sheet_name(faculty)
                if faculty_name == "":
                    faculty_name = "Unknown_Faculty"
                timetable.to_excel(writer, sheet_name=faculty_name)
        
        # Reset pointer to beginning of file
        output.seek(0)
        return output