import pandas as pd
from pandas import ExcelWriter
from io import BytesIO
import zipfile
import re

class FileManager:
    """Manages file operations for timetable data"""
    
    def __init__(self, timetables, faculty_timetables, validation_results=None):
        """Initialize with timetables, faculty timetables, and optional validation results"""
        self.timetables = timetables
        self.faculty_timetables = faculty_timetables
        self.validation_results = validation_results or {}
    
    def sanitize_sheet_name(self, name):
        """Sanitize Excel sheet names to avoid errors"""
        if not isinstance(name, str):
            name = str(name)
        # Remove invalid characters and truncate to Excel's limit
        sanitized = re.sub(r'[\[\]\*\?\:/\\]', '', name)[:31]
        return sanitized
    
    def generate_excel(self):
        """Create an Excel file with all timetables"""
        output = BytesIO()
        
        with ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write Section Timetables
            for sec, df in self.timetables.items():
                df.to_excel(writer, sheet_name=self.sanitize_sheet_name(f"Sec_{sec}"))
            
            # Write Faculty Timetables
            for faculty, timetable in self.faculty_timetables.items():
                faculty_name = self.sanitize_sheet_name(f"F_{faculty}")
                if faculty_name == "F_":
                    faculty_name = "Unknown_Faculty"
                timetable.to_excel(writer, sheet_name=faculty_name)
                
            # Add validation summary if available
            if self.validation_results:
                # Create summary sheet
                summary_data = {
                    "Metric": ["Total Violations", "Quality Score"],
                    "Value": [
                        sum(self.validation_results.get("violations", {}).values()),
                        self.validation_results.get("score", "N/A")
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Summary")
                
                # Create violations sheet if violations exist
                violations = self.validation_results.get("violations", {})
                if violations:
                    violations_data = []
                    for constraint, count in violations.items():
                        if count > 0:
                            violations_data.append({
                                "Constraint": constraint,
                                "Count": count
                            })
                    if violations_data:
                        violations_df = pd.DataFrame(violations_data)
                        violations_df.to_excel(writer, sheet_name="Violations")
        
        # Reset pointer to beginning of file
        output.seek(0)
        return output
    
    def generate_csv(self):
        """Create a zip file with CSV files for each timetable"""
        output = BytesIO()
        
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add section timetables
            for sec, df in self.timetables.items():
                csv_data = df.to_csv()
                zipf.writestr(f"sections/{sec}.csv", csv_data)
            
            # Add faculty timetables
            for faculty, timetable in self.faculty_timetables.items():
                safe_name = re.sub(r'[^\w\s-]', '', faculty).strip() or "Unknown_Faculty"
                csv_data = timetable.to_csv()
                zipf.writestr(f"faculty/{safe_name}.csv", csv_data)
                
            # Add summary file if validation results are available
            if self.validation_results:
                summary = f"Timetable Validation Summary\n"
                summary += f"Total Violations: {sum(self.validation_results.get('violations', {}).values())}\n"
                summary += f"Quality Score: {self.validation_results.get('score', 'N/A')}\n\n"
                
                # Add violations summary
                summary += "Constraint Violations:\n"
                for constraint, count in self.validation_results.get("violations", {}).items():
                    if count > 0:
                        summary += f"- {constraint}: {count}\n"
                        
                zipf.writestr("summary.txt", summary)
        
        # Reset pointer to beginning of file
        output.seek(0)
        return output
    
    def generate_pdf(self):
        """Create PDFs for timetables (implemented as a placeholder)"""
        # In a real implementation, this would use a PDF generation library
        # For now, we'll just create a text file explaining this is a placeholder
        output = BytesIO()
        
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zipf:
            placeholder = "This is a placeholder for PDF generation functionality.\n"
            placeholder += "In a production environment, this would generate formatted PDF timetables."
            zipf.writestr("pdf_placeholder.txt", placeholder)
            
            # Add simple text versions as placeholders
            for sec, df in self.timetables.items():
                text_data = f"Timetable for {sec}\n\n"
                text_data += df.to_string()
                zipf.writestr(f"{sec}.txt", text_data)
        
        output.seek(0)
        return output
    
    def generate_validation_report(self):
        """Create a validation report"""
        # In a real implementation, this would create a PDF report
        # For now, we'll create a text summary
        output = BytesIO()
        
        report = "Timetable Validation Report\n"
        report += "==========================\n\n"
        
        if self.validation_results:
            # Add overall metrics
            report += "Overall Metrics:\n"
            report += f"- Total Violations: {sum(self.validation_results.get('violations', {}).values())}\n"
            report += f"- Quality Score: {self.validation_results.get('score', 'N/A')}\n\n"
            
            # Add constraint violations
            violations = self.validation_results.get("violations", {})
            if violations:
                report += "Constraint Violations:\n"
                for constraint, count in violations.items():
                    if count > 0:
                        report += f"- {constraint}: {count}\n"
                
                # Add violation details if available
                violation_details = self.validation_results.get("violation_details", {})
                if violation_details:
                    report += "\nViolation Details:\n"
                    for constraint, details in violation_details.items():
                        if details:
                            report += f"\n{constraint}:\n"
                            for detail in details:
                                report += f"- {detail}\n"
            
            # Add suggestions if available
            suggestions = self.validation_results.get("suggestions", [])
            if suggestions:
                report += "\nImprovement Suggestions:\n"
                for i, suggestion in enumerate(suggestions, 1):
                    report += f"{i}. {suggestion}\n"
        else:
            report += "No validation results available."
        
        output.write(report.encode('utf-8'))
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
        
    def generate_custom_excel(self, selected_sections=None, selected_faculty=None):
        """Create a custom Excel file with selected sections and faculty"""
        output = BytesIO()
        
        with ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write selected section timetables
            if selected_sections:
                for sec in selected_sections:
                    if sec in self.timetables:
                        self.timetables[sec].to_excel(writer, sheet_name=self.sanitize_sheet_name(f"Sec_{sec}"))
            
            # Write selected faculty timetables
            if selected_faculty:
                for faculty in selected_faculty:
                    if faculty in self.faculty_timetables:
                        faculty_name = self.sanitize_sheet_name(f"F_{faculty}")
                        if faculty_name == "F_":
                            faculty_name = "Unknown_Faculty"
                        self.faculty_timetables[faculty].to_excel(writer, sheet_name=faculty_name)
        
        # Reset pointer to beginning of file
        output.seek(0)
        return output