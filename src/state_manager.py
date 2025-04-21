# modules/state_manager.py
import streamlit as st

class StateManager:
    """Manages the application state using Streamlit's session state"""
    
    def __init__(self):
        """Initialize session state variables if they don't exist"""
        if 'timetables' not in st.session_state:
            st.session_state.timetables = {}
        if 'faculty_timetables' not in st.session_state:
            st.session_state.faculty_timetables = {}
        if 'blocks' not in st.session_state:
            st.session_state.blocks = {}
        if 'generation_complete' not in st.session_state:
            st.session_state.generation_complete = False
        if 'active_letter' not in st.session_state:
            st.session_state.active_letter = None
    
    def get_timetables(self):
        """Get the generated timetables"""
        return st.session_state.timetables
    
    def set_timetables(self, timetables):
        """Store generated timetables in session state"""
        st.session_state.timetables = timetables
    
    def get_faculty_timetables(self):
        """Get the generated faculty timetables"""
        return st.session_state.faculty_timetables
    
    def set_faculty_timetables(self, faculty_timetables):
        """Store generated faculty timetables in session state"""
        st.session_state.faculty_timetables = faculty_timetables
    
    def get_blocks(self):
        """Get the data blocks"""
        return st.session_state.blocks
    
    def set_blocks(self, blocks):
        """Store data blocks in session state"""
        st.session_state.blocks = blocks
    
    def is_generation_complete(self):
        """Check if timetable generation is complete"""
        return st.session_state.generation_complete
    
    def set_generation_complete(self, status):
        """Set the generation complete status"""
        st.session_state.generation_complete = status
    
    def get_active_letter(self):
        """Get the currently active letter for faculty filtering"""
        return st.session_state.active_letter
    
    def set_active_letter(self, letter):
        """Set the active letter for faculty filtering"""
        st.session_state.active_letter = letter