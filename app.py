# app.py
import streamlit as st
from io import BytesIO
import pandas as pd
import time

# Fix import paths - use relative imports
from src.timetable_validator import TimetableValidator
from src.ui_components import setup_ui, create_tabs
from src.state_manager import StateManager
from src.data_processor import DataProcessor
from src.timetable_generator import TimetableGenerator
from src.file_manager import FileManager

def main():
    """Main application entry point"""
    # Setup UI components
    setup_ui()
    
    # Create the state manager instance
    state_manager = StateManager()
    
    # Initialize generate_button
    generate_button = False
    
    # Create tabs for the main sections
    tab1, tab2, tab3, tab4, tab5 = create_tabs([
        "📤 Upload & Generate", 
        "📊 Year-wise Timetables", 
        "👩‍🏫 Faculty Timetables", 
        "📥 Download",
        "🔍 Validation"
    ])
    
    # Handle upload & generate tab
    with tab1:
        st.markdown('<div class="tab-subheader">Upload Subject Allocation File</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Subject Allocation Excel File", type=["xlsx"])
        
        if uploaded_file:
            with st.spinner("Processing data..."):
                try:
                    data_processor = DataProcessor(uploaded_file)
                    blocks = data_processor.process_file()
                    
                    if not blocks:
                        st.error("No valid data blocks were extracted from the uploaded file. Please check the file format.")
                    else:
                        state_manager.set_blocks(blocks)
                        st.success(f"File uploaded successfully! Found {len(blocks)} section blocks.")
                        
                        # Show a preview of the extracted data
                        with st.expander("Preview Extracted Data"):
                            for section, data in blocks.items():
                                st.subheader(f"Section: {section}")
                                st.dataframe(data)
                        
                        # Settings for timetable generation
                        st.markdown("### Generation Settings")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            max_attempts = st.slider("Maximum Generation Attempts", 
                                                   min_value=1, max_value=50, value=10,
                                                   help="Higher values might produce better results but will take longer")
                        
                        with col2:
                            consecutive_hours = st.checkbox("Enforce Consecutive Hours for Classes", value=True,
                                                         help="When checked, multi-hour classes will be scheduled consecutively on the same day")
                        
                        # Button to start generation
                        generate_button = st.button("🔄 Generate Timetables", type="primary", use_container_width=True)
                        
                        if not generate_button:
                            st.info("Configure settings and click 'Generate Timetables' to process the data.")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.exception(e)
                    generate_button = False
        else:
            st.info("Please upload an Excel file with subject allocation data.")
            generate_button = False
    
    # Main generation logic
    if uploaded_file and generate_button and state_manager.get_blocks():
        with tab1:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Start timer
            start_time = time.time()
            
            try:
                # Phase 1: Initial processing
                status_text.text("Phase 1/3: Processing input data...")
                progress_bar.progress(10)
                time.sleep(0.5)  # Simulate processing time
                
                # Phase 2: Generating timetables
                status_text.text("Phase 2/3: Generating timetables (this may take a moment)...")
                progress_bar.progress(30)
                
                # Get blocks from state manager
                blocks = state_manager.get_blocks()
                
                # Debug output
                st.write(f"Blocks found: {len(blocks)}")
                
                # Actual generation
                timetable_generator = TimetableGenerator(blocks)
                timetables, faculty_timetables, validation_summary = timetable_generator.generate(max_attempts)
                
                progress_bar.progress(70)
                status_text.text("Phase 3/3: Validating and finalizing results...")
                
                if timetables and faculty_timetables:
                    # Phase 3: Validation
                    validator = TimetableValidator(timetables, faculty_timetables, blocks)
                    validation_results = validator.validate()
                    
                    # Store results in session state
                    state_manager.set_timetables(timetables)
                    state_manager.set_faculty_timetables(faculty_timetables)
                    state_manager.set_validation_results(validation_results)
                    state_manager.set_generation_summary(validation_summary)
                    state_manager.set_generation_complete(True)
                    
                    # Complete progress bar
                    progress_bar.progress(100)
                    
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    
                    # Display completion message
                    st.success(f"✅ Timetables Generated Successfully in {elapsed_time:.2f} seconds!")
                    
                    # Display generation summary
                    st.markdown("### Generation Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Violations", sum(validation_results["violations"].values()))
                    with col2:
                        st.metric("Generation Attempts", validation_summary["attempts"])
                    with col3:
                        st.metric("Quality Score", f"{validation_results['score']:.2f}", 
                                 help="Lower is better. Factors in violations and faculty workload balance.")
                    
                    # Quick summary of violations
                    if validation_results["violations"]:
                        st.markdown("#### Constraint Violations")
                        for constraint, count in validation_results["violations"].items():
                            if count > 0:
                                st.info(f"**{constraint}**: {count} instances")
                    else:
                        st.success("No constraint violations detected. Perfect timetable!")
                        
                    st.info("Navigate to other tabs to view the results. Check the 'Validation' tab for more details.")
                else:
                    st.error("Timetable generation failed. No timetables were produced.")
                    status_text.text("Generation failed. Please check your input data.")
                    progress_bar.progress(100)
            except Exception as e:
                st.error(f"Error generating timetables: {str(e)}")
                st.exception(e)
                status_text.text("Generation failed due to an error.")
                progress_bar.progress(100)
    
    # Year-wise Timetables Tab
    with tab2:
        st.markdown('<div class="tab-subheader">Year-wise Timetables</div>', unsafe_allow_html=True)
        
        if state_manager.is_generation_complete():
            timetables = state_manager.get_timetables()
            
            # Group timetables by year
            year_groups = {}
            for sec in timetables:
                year = sec.split('-')[0]
                if year not in year_groups:
                    year_groups[year] = []
                year_groups[year].append(sec)
            
            # Create tabs for each year
            if year_groups:
                year_tabs = st.tabs([f"Year {year}" for year in sorted(year_groups.keys())])
                
                for i, year in enumerate(sorted(year_groups.keys())):
                    with year_tabs[i]:
                        for sec in sorted(year_groups[year]):
                            st.subheader(f"Section {sec.split('-')[1]}")
                            
                            # Apply color coding for labs and consecutive classes
                            styled_df = timetables[sec].fillna("")
                            
                            # Display with highlighting
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Display any section-specific notes
                            if sec in state_manager.get_validation_results().get("section_notes", {}):
                                for note in state_manager.get_validation_results()["section_notes"][sec]:
                                    st.info(note)
                            
                            st.divider()
            else:
                st.warning("No timetables available to display.")
        else:
            st.info("Generate timetables first to view them here.")
    
    # Faculty Timetables Tab
    with tab3:
        st.markdown('<div class="tab-subheader">Faculty Timetables</div>', unsafe_allow_html=True)
        
        if state_manager.is_generation_complete():
            faculty_timetables = state_manager.get_faculty_timetables()
            
            # Group faculty alphabetically
            faculty_groups = {}
            for faculty in faculty_timetables:
                first_letter = faculty[0].upper() if faculty else "#"
                if first_letter not in faculty_groups:
                    faculty_groups[first_letter] = []
                faculty_groups[first_letter].append(faculty)
            
            if faculty_groups:
                # Create search functionality
                search_query = st.text_input("🔍 Search for faculty", "")
                
                if search_query:
                    filtered_faculty = [f for f in faculty_timetables if search_query.lower() in f.lower()]
                    
                    if filtered_faculty:
                        for faculty in sorted(filtered_faculty):
                            with st.expander(f"{faculty} Timetable", expanded=True):
                                st.dataframe(faculty_timetables[faculty].fillna(""), use_container_width=True)
                                
                                # Display faculty specific notes
                                if faculty in state_manager.get_validation_results().get("faculty_notes", {}):
                                    for note in state_manager.get_validation_results()["faculty_notes"][faculty]:
                                        st.info(note)
                    else:
                        st.warning(f"No faculty found matching '{search_query}'")
                else:
                    # Display alphabetical tabs if no search
                    alpha_tabs = st.tabs(sorted(faculty_groups.keys()))
                    
                    for i, letter in enumerate(sorted(faculty_groups.keys())):
                        with alpha_tabs[i]:
                            for faculty in sorted(faculty_groups[letter]):
                                with st.expander(f"{faculty} Timetable"):
                                    st.dataframe(faculty_timetables[faculty].fillna(""), use_container_width=True)
                                    
                                    # Display faculty specific notes
                                    if faculty in state_manager.get_validation_results().get("faculty_notes", {}):
                                        for note in state_manager.get_validation_results()["faculty_notes"][faculty]:
                                            st.info(note)
            else:
                st.warning("No faculty timetables available to display.")
        else:
            st.info("Generate timetables first to view faculty schedules.")
    
    # Download Tab
    with tab4:
        st.markdown('<div class="tab-subheader">Download Generated Timetables</div>', unsafe_allow_html=True)
        
        if state_manager.is_generation_complete():
            # Create file manager instance
            file_manager = FileManager(
                state_manager.get_timetables(),
                state_manager.get_faculty_timetables(),
                state_manager.get_validation_results()
            )
            
            # Offer different download options
            st.markdown("### Export Options")
            
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                # Download all timetables as Excel
                excel_bytes = file_manager.generate_excel()
                st.download_button(
                    label="📥 Download All Timetables (Excel)",
                    data=excel_bytes,
                    file_name="timetables.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
                # Download validation report
                report_bytes = file_manager.generate_validation_report()
                st.download_button(
                    label="📄 Download Validation Report (PDF)",
                    data=report_bytes,
                    file_name="validation_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            with download_col2:
                # Download section-wise timetables
                csv_bytes = file_manager.generate_csv()
                st.download_button(
                    label="📊 Download as CSV Files (zip)",
                    data=csv_bytes,
                    file_name="timetables_csv.zip",
                    mime="application/zip",
                    use_container_width=True
                )
                
                # Download printable PDFs
                pdf_bytes = file_manager.generate_pdf()
                st.download_button(
                    label="🖨️ Download Printable PDFs",
                    data=pdf_bytes,
                    file_name="timetables_pdf.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            
            # Advanced export options
            with st.expander("Advanced Export Options"):
                st.markdown("#### Custom Export")
                
                # Select specific sections to export
                all_sections = list(state_manager.get_timetables().keys())
                selected_sections = st.multiselect(
                    "Select specific sections to export",
                    options=all_sections,
                    default=all_sections[:min(5, len(all_sections))]
                )
                
                # Select specific faculty to export
                all_faculty = list(state_manager.get_faculty_timetables().keys())
                selected_faculty = st.multiselect(
                    "Select specific faculty to export",
                    options=all_faculty,
                    default=all_faculty[:min(5, len(all_faculty))]
                )
                
                if selected_sections or selected_faculty:
                    custom_excel = file_manager.generate_custom_excel(selected_sections, selected_faculty)
                    st.download_button(
                        label="📥 Download Custom Selection",
                        data=custom_excel,
                        file_name="custom_timetables.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.info("Generate timetables first to download them.")
    
    # Validation Tab
    with tab5:
        st.markdown('<div class="tab-subheader">Timetable Validation</div>', unsafe_allow_html=True)
        
        if state_manager.is_generation_complete():
            validation_results = state_manager.get_validation_results()
            
            # Overview
            st.markdown("### Validation Overview")
            
            # Create metrics
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Total Violations", sum(validation_results["violations"].values()))
            with metric_cols[1]:
                st.metric("Faculty Conflicts", validation_results["violations"].get("faculty_conflict", 0))
            with metric_cols[2]:
                st.metric("Room Conflicts", validation_results["violations"].get("room_conflict", 0))
            with metric_cols[3]:
                st.metric("Quality Score", f"{validation_results['score']:.2f}")
            
            # Constraint violations
            st.markdown("### Constraint Violations")
            
            if sum(validation_results["violations"].values()) > 0:
                # Create a DataFrame for violations
                violations_data = []
                for constraint, count in validation_results["violations"].items():
                    if count > 0:
                        violations_data.append({
                            "Constraint": constraint,
                            "Violations": count,
                            "Severity": validation_results.get("severity", {}).get(constraint, "Medium")
                        })
                
                violations_df = pd.DataFrame(violations_data)
                st.dataframe(violations_df, use_container_width=True)
                
                # Display specific violation details
                st.markdown("### Violation Details")
                for constraint, details in validation_results.get("violation_details", {}).items():
                    if details:
                        with st.expander(f"{constraint} ({len(details)} issues)"):
                            for detail in details:
                                st.info(detail)
            else:
                st.success("No violations detected. All constraints satisfied!")
            
            # Workload analysis
            st.markdown("### Faculty Workload Analysis")
            
            workload_data = validation_results.get("workload_balance", {})
            if workload_data:
                # Convert to DataFrame if it's the correct format
                st.write(f"Standard Deviation: {workload_data.get('std_dev', 0):.2f}")
                st.write(f"Min Hours: {workload_data.get('min', 0)}")
                st.write(f"Max Hours: {workload_data.get('max', 0)}")
                st.write(f"Average Hours: {workload_data.get('avg', 0):.2f}")
            
            # Improvement suggestions
            st.markdown("### Improvement Suggestions")
            suggestions = validation_results.get("suggestions", [])
            
            if suggestions:
                for i, suggestion in enumerate(suggestions, 1):
                    st.info(f"**Suggestion {i}:** {suggestion}")
            else:
                st.success("No improvement suggestions - timetable is optimized!")
        else:
            st.info("Generate timetables first to see validation results.")


if __name__ == "__main__":
    main()