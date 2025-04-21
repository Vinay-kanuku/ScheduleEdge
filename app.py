# app.py
import streamlit as st
from io import BytesIO
import pandas as pd

from modules.ui_components import setup_ui, create_tabs
from modules.data_processor import DataProcessor
from modules.timetable_generator import TimetableGenerator
from modules.file_manager import FileManager
from modules.state_manager import StateManager

def main():
    """Main application entry point"""
    # Setup UI components
    setup_ui()
    
    # Create the state manager instance
    state_manager = StateManager()
    
    # Create tabs for the main sections
    tab1, tab2, tab3, tab4 = create_tabs([
        "📤 Upload & Generate", 
        "📊 Year-wise Timetables", 
        "👩‍🏫 Faculty Timetables", 
        "📥 Download"
    ])
    
    # Handle upload & generate tab
    with tab1:
        st.markdown('<div class="tab-subheader">Upload Subject Allocation File</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Subject Allocation Excel File", type=["xlsx"])
        
        if uploaded_file:
            with st.spinner("Processing data..."):
                data_processor = DataProcessor(uploaded_file)
                blocks = data_processor.process_file()
                state_manager.set_blocks(blocks)
                
                st.success("File uploaded successfully!")
                
                # Button to start generation
                generate_button = st.button("🔄 Generate Timetables", type="primary", use_container_width=True)
                
                if not generate_button:
                    st.info("Click 'Generate Timetables' to process the data and create timetables.")
        else:
            st.info("Please upload an Excel file with subject allocation data.")
            generate_button = False
    
    # Main generation logic
    if uploaded_file and generate_button:
        with st.spinner("Generating timetables..."):
            timetable_generator = TimetableGenerator(state_manager.get_blocks())
            timetables, faculty_timetables = timetable_generator.generate()
            
            state_manager.set_timetables(timetables)
            state_manager.set_faculty_timetables(faculty_timetables)
            state_manager.set_generation_complete(True)
        
        with tab1:
            st.success("✅ Timetables Generated Successfully!")
            st.info("Navigate to other tabs to view the results.")
    
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
                            st.dataframe(timetables[sec].fillna(""), use_container_width=True)
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
                        for faculty in filtered_faculty:
                            st.subheader(f"{faculty}")
                            st.dataframe(faculty_timetables[faculty].fillna(""), use_container_width=True)
                            st.divider()
                    else:
                        st.warning("No faculty found matching your search.")
                else:
                    # Display all faculty grouped alphabetically
                    letter_cols = st.columns(4)
                    for i, letter in enumerate(sorted(faculty_groups.keys())):
                        with letter_cols[i % 4]:
                            if st.button(f"Group {letter}", use_container_width=True):
                                state_manager.set_active_letter(letter)
                    
                    active_letter = state_manager.get_active_letter()
                    if active_letter in faculty_groups:
                        st.subheader(f"Faculty Names Starting with {active_letter}")
                        for faculty in sorted(faculty_groups[active_letter]):
                            with st.expander(f"{faculty}"):
                                st.dataframe(faculty_timetables[faculty].fillna(""), use_container_width=True)
            else:
                st.warning("No faculty timetables available to display.")
        else:
            st.info("Generate timetables first to view faculty schedules.")
    
    # Download Tab
    with tab4:
        st.markdown('<div class="tab-subheader">Download Timetables</div>', unsafe_allow_html=True)
        
        if state_manager.is_generation_complete():
            file_manager = FileManager(
                state_manager.get_timetables(),
                state_manager.get_faculty_timetables()
            )
            
            try:
                complete_output = file_manager.create_complete_timetable_excel()
                
                st.download_button(
                    label="📥 Download Complete Timetable Excel",
                    data=complete_output.getvalue(),
                    file_name="Timetable_With_L1_L2_Labs_in_Single_Slot.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
                # Options for selective downloading
                st.markdown("### Download Options")
                
                download_option = st.radio(
                    "Select what to download:",
                    ["Complete Timetable", "Year-wise Timetables", "Faculty Timetables"]
                )
                
                if download_option == "Year-wise Timetables":
                    year_selection = st.selectbox(
                        "Select Year:",
                        sorted(set([sec.split('-')[0] for sec in state_manager.get_timetables()]))
                    )
                    
                    if year_selection:
                        year_output = file_manager.create_year_timetable_excel(year_selection)
                        
                        st.download_button(
                            label=f"📥 Download Year {year_selection} Timetables",
                            data=year_output.getvalue(),
                            file_name=f"Year_{year_selection}_Timetables.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                
                elif download_option == "Faculty Timetables":
                    faculty_output = file_manager.create_faculty_timetable_excel()
                    
                    st.download_button(
                        label="📥 Download All Faculty Timetables",
                        data=faculty_output.getvalue(),
                        file_name="Faculty_Timetables.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Error generating download file: {str(e)}")
                st.info("Try again with a different file or contact support if the issue persists.")
        else:
            st.info("Generate timetables first to enable download options.")

if __name__ == "__main__":
    main()