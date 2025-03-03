import streamlit as st
import pandas as pd
import random
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
from functools import lru_cache
import concurrent.futures

# Configuration
SUBJECT_SHORTCUTS = {
    "Software Engineering": "SE",
    "Artificial Intelligence": "AI",
    "Web Programming": "WP",
    "Data Warehousing and Data Mining": "DWDM",
    "Introduction to Data Science": "IDS",
    "Internet of Things and Applications Lab": "IOT Lab",
    "Web Programming Lab": "WP Lab",
    "Artificial Intelligence Lab Using Python": "AI Lab",
    "Constitution of India": "COI"
}

SUBJECTS = [
    {"name": "Software Engineering", "weekly_hours": 4, "faculty": ["PKA"], "is_lab": False},
    {"name": "Artificial Intelligence", "weekly_hours": 5, "faculty": ["Dr. GP"], "is_lab": False},
    {"name": "Web Programming", "weekly_hours": 4, "faculty": ["KS"], "is_lab": False},
    {"name": "Data Warehousing and Data Mining", "weekly_hours": 4, "faculty": ["Dr. NR"], "is_lab": False},
    {"name": "Introduction to Data Science", "weekly_hours": 3, "faculty": ["CHS"], "is_lab": False},
    {"name": "Internet of Things and Applications Lab", "weekly_hours": 2.5, "faculty": ["Dr. KRK", "LIB"], "is_lab": True},
    {"name": "Web Programming Lab", "weekly_hours": 2.5, "faculty": ["KS", "KB", "Dr. KP"], "is_lab": True},
    {"name": "Artificial Intelligence Lab Using Python", "weekly_hours": 2.5, "faculty": ["BHKD", "Dr. GP"], "is_lab": True},
    {"name": "Constitution of India", "weekly_hours": 2, "faculty": ["PPS"], "is_lab": False}
]

SECTIONS = ["A", "B", "C"]
LAB_BATCHES = ["1", "2"]
DAYS = ["Monday", "Tuesday", "Thursday", "Friday", "Saturday"]

# Modified time slots with lunch break
MORNING_SLOTS = [
    "9:00-9:50", 
    "9:50-10:40", 
    "10:40-11:30", 
    "11:30-12:20"
]

LUNCH_BREAK = "12:20-1:00"

AFTERNOON_SLOTS = [
    "1:00-1:50",
    "1:50-2:40",
    "2:40-3:30"
]

TIME_SLOTS = MORNING_SLOTS + [LUNCH_BREAK] + AFTERNOON_SLOTS

# Pre-compute available time slots (excluding lunch break)
AVAILABLE_TIME_SLOTS = [(day, time) for day in DAYS 
                       for time in (MORNING_SLOTS + AFTERNOON_SLOTS)]

class FastTimeTable:
    def __init__(self, section: str):
        self.section = section
        self.sessions = []
        self.slot_usage = defaultdict(list)
        self.faculty_usage = defaultdict(list)
        self.lab_slots = set()  # Track lab slots for better scheduling
        
    def add_session(self, subject: dict, slots: List[Tuple[str, str]], faculty: List[str], batch: str = None):
        session = {
            'subject': subject,
            'slots': slots,
            'faculty': faculty,
            'batch': batch
        }
        self.sessions.append(session)
        
        for slot in slots:
            self.slot_usage[slot].append(session)
            if subject['is_lab']:
                self.lab_slots.add(slot)
            for f in faculty:
                self.faculty_usage[(f, slot[0], slot[1])].append(session)

    def has_conflicts(self) -> bool:
        return any(len(sessions) > 1 for sessions in self.slot_usage.values()) or \
               any(len(sessions) > 1 for sessions in self.faculty_usage.values())

    def get_available_lab_slots(self) -> List[List[Tuple[str, str]]]:
        """Get available slots specifically for lab sessions"""
        available_labs = []
        
        for day in DAYS:
            # Morning lab slots (3 consecutive periods)
            morning_slots = [(day, slot) for slot in MORNING_SLOTS]
            if len(morning_slots) >= 3:
                for i in range(len(morning_slots) - 2):
                    consecutive = morning_slots[i:i+3]
                    if all(len(self.slot_usage[slot]) == 0 for slot in consecutive):
                        available_labs.append(consecutive)
            
            # Afternoon lab slots (all afternoon periods)
            afternoon_slots = [(day, slot) for slot in AFTERNOON_SLOTS]
            if all(len(self.slot_usage[(day, slot)]) == 0 for slot in AFTERNOON_SLOTS):
                available_labs.append(afternoon_slots)
        
        return available_labs

    @lru_cache(maxsize=1024)
    def get_available_slots(self, consecutive_count: int = 1) -> List[List[Tuple[str, str]]]:
        """Get available slots for regular sessions"""
        if consecutive_count == 1:
            return [[slot] for slot in AVAILABLE_TIME_SLOTS 
                    if len(self.slot_usage[slot]) == 0 and slot not in self.lab_slots]
        
        result = []
        for day in DAYS:
            day_slots = [slot for slot in AVAILABLE_TIME_SLOTS 
                        if slot[0] == day and len(self.slot_usage[slot]) == 0 
                        and slot not in self.lab_slots]
            day_slots.sort(key=lambda x: TIME_SLOTS.index(x[1]))
            
            for i in range(len(day_slots) - consecutive_count + 1):
                consecutive = day_slots[i:i+consecutive_count]
                if all(TIME_SLOTS.index(consecutive[j+1][1]) == 
                      TIME_SLOTS.index(consecutive[j][1]) + 1 
                      for j in range(len(consecutive)-1)):
                    result.append(consecutive)
        return result

def create_timetable_df(timetable: FastTimeTable) -> pd.DataFrame:
    df = pd.DataFrame(columns=TIME_SLOTS, index=DAYS)
    df = df.fillna("-")
    
    # Set lunch break
    df[LUNCH_BREAK] = "LUNCH BREAK"
    
    for session in timetable.sessions:
        entry = f"{SUBJECT_SHORTCUTS[session['subject']['name']]}"
        if session['subject']['is_lab']:
            entry += f" (B{session['batch']}: {', '.join(session['faculty'])})"
        else:
            entry += f" ({session['faculty'][0]})"
            
        for day, time in session['slots']:
            df.at[day, time] = entry
    
    return df

def generate_section_timetable(section: str, progress_bar=None) -> pd.DataFrame:
    if progress_bar:
        progress_bar.progress(0.0, text=f"Generating timetable for Section {section}...")
    
    best_timetable = None
    best_fitness = -1
    max_attempts = 20  # Increased attempts for better results
    
    for attempt in range(max_attempts):
        timetable = FastTimeTable(section)
        subjects = SUBJECTS.copy()
        random.shuffle(subjects)
        
        # First schedule labs with proper time slots
        for subject in [s for s in subjects if s['is_lab']]:
            for batch in LAB_BATCHES:
                available_lab_slots = timetable.get_available_lab_slots()
                if available_lab_slots:
                    slots = random.choice(available_lab_slots)
                    faculty = random.sample(subject['faculty'], 
                                         min(2, len(subject['faculty'])))
                    timetable.add_session(subject, slots, faculty, batch)
        
        # Then schedule regular subjects
        for subject in [s for s in subjects if not s['is_lab']]:
            slots_needed = subject['weekly_hours']
            while slots_needed > 0:
                available = timetable.get_available_slots(1)
                if available:
                    slots = random.choice(available)
                    timetable.add_session(subject, slots, [subject['faculty'][0]])
                    slots_needed -= 1
        
        fitness = calculate_fitness(timetable)
        if fitness > best_fitness:
            best_timetable = timetable
            best_fitness = fitness
        
        if progress_bar:
            progress_bar.progress((attempt + 1) / max_attempts, 
                               text=f"Attempt {attempt + 1}/{max_attempts} for Section {section}")
    
    return create_timetable_df(best_timetable)

def calculate_fitness(timetable: FastTimeTable) -> float:
    if timetable.has_conflicts():
        return 0
    
    score = 1000
    
    # Penalize gaps in schedule (excluding lunch break)
    for day in DAYS:
        morning_slots = [slot[1] for slot in timetable.slot_usage.keys() 
                        if slot[0] == day and slot[1] in MORNING_SLOTS]
        afternoon_slots = [slot[1] for slot in timetable.slot_usage.keys() 
                         if slot[0] == day and slot[1] in AFTERNOON_SLOTS]
        
        # Check gaps separately for morning and afternoon
        for slots in [morning_slots, afternoon_slots]:
            slots.sort(key=lambda x: TIME_SLOTS.index(x))
            for i in range(len(slots) - 1):
                if TIME_SLOTS.index(slots[i+1]) - TIME_SLOTS.index(slots[i]) > 1:
                    score -= 5  # Increased penalty for gaps
    
    # Reward balanced distribution across days
    day_loads = defaultdict(int)
    for session in timetable.sessions:
        for day, _ in session['slots']:
            day_loads[day] += 1
    
    std_dev = np.std(list(day_loads.values()))
    score -= std_dev * 10  # Penalize uneven distribution
    
    return score

def display_timetable(df: pd.DataFrame) -> None:
    # Style the lunch break row differently
    def style_lunch_break(val):
        if val == "LUNCH BREAK":
            return "background-color: #f0f0f0; font-style: italic;"
        return ""
    
    styled_df = df.style.applymap(style_lunch_break)
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=300,
        column_config={
            col: st.column_config.Column(
                width="medium"
            ) for col in df.columns
        }
    )


def main():
    st.set_page_config(page_title="Class Timetable Generator", layout="wide")
    
    st.title("📚 Class Timetable Generator")
    st.write("Generate and visualize class timetables for multiple sections")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    selected_sections = st.sidebar.multiselect(
        "Select Sections",
        SECTIONS,
        default=SECTIONS
    )
    
    if st.sidebar.button("Generate Timetables"):
        progress_container = st.empty()
        progress_bar = st.progress(0)
        
        timetables = {}
        for i, section in enumerate(selected_sections):
            timetables[section] = generate_section_timetable(
                section, 
                progress_bar
            )
            progress_bar.progress((i + 1) / len(selected_sections))
        
        progress_container.empty()
        progress_bar.empty()
        
        # Display timetables
        for section in selected_sections:
            st.header(f"Section {section} Timetable")
            display_timetable(timetables[section])
            
            # Add download buttons
            csv = timetables[section].to_csv().encode('utf-8')
            st.download_button(
                label=f"Download Section {section} Timetable",
                data=csv,
                file_name=f'timetable_section_{section}.csv',
                mime='text/csv'
            )
        
        # Success message
        st.sidebar.success("✅ Timetables generated successfully!")

if __name__ == "__main__":
    main()