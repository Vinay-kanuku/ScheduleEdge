import streamlit as st
from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple
from collections import defaultdict

# Configuration constants (these need to be defined based on your specific needs)
SECTIONS = ["A", "B", "C"]  # Example sections
DAYS = ["Monday", "Tuesday", "Thursday", "Friday", "Saturday"]
MORNING_SLOTS = ["9:00-9:50", "9:50-10:40", "10:40-11:30", "11:30-12:20"]
LUNCH_BREAK = "12:20-1:00"
AFTERNOON_SLOTS = ["1:00-1:50", "1:50-2:40", "2:40-3:30"]
TIME_SLOTS = MORNING_SLOTS + [LUNCH_BREAK] + AFTERNOON_SLOTS
LAB_BATCHES = ["1", "2", "3"]  # Example lab batches

# You'll need to define these based on your requirements
SUBJECTS = [
    {
        "name": "Mathematics",
        "weekly_hours": 4,
        "is_lab": False,
        "faculty": ["Prof. Smith"]
    },
    # Add more subjects here
]

SUBJECT_SHORTCUTS = {
    "Mathematics": "MATH",
    # Add more shortcuts here
}

def create_slots_map():
    """Create a mapping of slots to integers for OR-Tools"""
    all_slots = MORNING_SLOTS + [LUNCH_BREAK] + AFTERNOON_SLOTS
    return {slot: idx for idx, slot in enumerate(all_slots)}

def create_days_map():
    """Create a mapping of days to integers for OR-Tools"""
    return {day: idx for idx, day in enumerate(DAYS)}

class TimeTableSolver:
    def __init__(self, section: str):
        self.section = section
        self.model = cp_model.CpModel()
        self.slots_map = create_slots_map()
        self.days_map = create_days_map()
        self.solver = cp_model.CpSolver()
        self.variables = {}
        self.lab_variables = {}

    # [Previous TimeTableSolver methods remain the same as in the first script]
    def create_variables(self):
        """Create boolean variables for the CP-SAT solver"""
        # For theory subjects
        for subject in [s for s in SUBJECTS if not s['is_lab']]:
            for day in self.days_map:
                for slot in self.slots_map:
                    if slot != LUNCH_BREAK:
                        self.variables[(subject['name'], day, slot)] = \
                            self.model.NewBoolVar(f'{subject["name"]}_{day}_{slot}')
        
        # For lab subjects
        for subject in [s for s in SUBJECTS if s['is_lab']]:
            for batch in LAB_BATCHES:
                for day in self.days_map:
                    valid_start_slots = ["9:00-9:50", "1:00-1:50"]
                    for start_slot in valid_start_slots:
                        self.lab_variables[(subject['name'], batch, day, start_slot)] = \
                            self.model.NewBoolVar(f'{subject["name"]}_B{batch}_{day}_{start_slot}')

    # [Include all other TimeTableSolver methods from the first script]

class FastTimeTable:
    def __init__(self, section: str):
        self.section = section
        self.sessions = []
        self.slot_usage = {}
    
    def has_conflicts(self) -> bool:
        """Check if there are any scheduling conflicts"""
        for day in DAYS:
            for slot in TIME_SLOTS:
                if slot == LUNCH_BREAK:
                    continue
                count = sum(1 for session in self.sessions
                          if (day, slot) in session['slots'])
                if count > 1:
                    return True
        return False
    
    def get_available_lab_slots(self) -> List[List[Tuple[str, str]]]:
        """Get available slots for lab sessions"""
        available_slots = []
        lab_start_slots = ["9:00-9:50", "1:00-1:50"]
        
        for day in DAYS:
            for start_slot in lab_start_slots:
                slots = []
                if start_slot == "9:00-9:50":
                    slots = [(day, slot) for slot in MORNING_SLOTS[:3]]
                else:
                    slots = [(day, slot) for slot in AFTERNOON_SLOTS]
                
                # Check if all required slots are available
                if all((day, slot) not in self.slot_usage for _, slot in slots):
                    available_slots.append(slots)
        
        return available_slots
    
    def get_available_slots(self, count: int) -> List[List[Tuple[str, str]]]:
        """Get available consecutive slots"""
        available_slots = []
        
        for day in DAYS:
            morning_slots = [(day, slot) for slot in MORNING_SLOTS]
            afternoon_slots = [(day, slot) for slot in AFTERNOON_SLOTS]
            
            for slots_group in [morning_slots, afternoon_slots]:
                for i in range(len(slots_group) - count + 1):
                    consecutive_slots = slots_group[i:i + count]
                    if all(slot not in self.slot_usage for slot in consecutive_slots):
                        available_slots.append(consecutive_slots)
        
        return available_slots
    
    def add_session(self, subject: Dict, slots: List[Tuple[str, str]], 
                   faculty: List[str], batch: str = None):
        """Add a teaching session to the timetable"""
        session = {
            'subject': subject['name'],
            'slots': slots,
            'faculty': faculty,
            'batch': batch
        }
        self.sessions.append(session)
        
        for slot in slots:
            self.slot_usage[slot] = session

def generate_section_timetable(section: str, progress_bar=None) -> pd.DataFrame:
    """Generate timetable using either OR-Tools or Fast implementation"""
    if progress_bar:
        progress_bar.progress(0.0, text=f"Generating timetable for Section {section}...")
    
    try:
        # First try using OR-Tools
        solver = TimeTableSolver(section)
        return solver.solve()
    except Exception as e:
        st.warning(f"OR-Tools solver failed, falling back to fast implementation: {str(e)}")
        
        # Fall back to fast implementation
        best_timetable = None
        best_fitness = -1
        max_attempts = 20
        
        for attempt in range(max_attempts):
            timetable = FastTimeTable(section)
            subjects = SUBJECTS.copy()
            random.shuffle(subjects)
            
            # Schedule labs first
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
                                   text=f"Attempt {attempt + 1}/{max_attempts}")
        
        return create_timetable_df(best_timetable)

def calculate_fitness(timetable: FastTimeTable) -> float:
    """Calculate fitness score for a timetable"""
    if timetable.has_conflicts():
        return 0
    
    score = 1000
    
    # Penalize gaps in schedule
    for day in DAYS:
        morning_slots = [slot[1] for slot in timetable.slot_usage.keys() 
                        if slot[0] == day and slot[1] in MORNING_SLOTS]
        afternoon_slots = [slot[1] for slot in timetable.slot_usage.keys() 
                         if slot[0] == day and slot[1] in AFTERNOON_SLOTS]
        
        for slots in [morning_slots, afternoon_slots]:
            slots.sort(key=lambda x: TIME_SLOTS.index(x))
            for i in range(len(slots) - 1):
                if TIME_SLOTS.index(slots[i+1]) - TIME_SLOTS.index(slots[i]) > 1:
                    score -= 5
    
    # Reward balanced distribution
    day_loads = defaultdict(int)
    for session in timetable.sessions:
        for day, _ in session['slots']:
            day_loads[day] += 1
    
    std_dev = np.std(list(day_loads.values()))
    score -= std_dev * 10
    
    return score

def create_timetable_df(timetable: FastTimeTable) -> pd.DataFrame:
    """Create DataFrame from timetable"""
    df = pd.DataFrame(index=DAYS, columns=TIME_SLOTS)
    df.fillna("-")
    df[LUNCH_BREAK] = "LUNCH BREAK"
    
    for session in timetable.sessions:
        subject_name = session['subject']
        faculty = ", ".join(session['faculty'])
        
        for day, slot in session['slots']:
            if session['batch']:
                df.at[day, slot] = f"{SUBJECT_SHORTCUTS[subject_name]} (B{session['batch']}: {faculty})"
            else:
                df.at[day, slot] = f"{SUBJECT_SHORTCUTS[subject_name]} ({faculty})"
    
    return df

def display_timetable(df: pd.DataFrame) -> None:
    """Display timetable with styling"""
    def style_lunch_break(val):
        if val == "LUNCH BREAK":
            return "background-color: #f0f0f0; font-style: italic;"
        return ""
    
    styled_df = df.style.applymap(style_lunch_break)
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=300,
        column_config={col: st.column_config.Column(width="medium") 
                      for col in df.columns}
    )

def main():
    st.set_page_config(page_title="Class Timetable Generator", layout="wide")
    
    st.title("📚 Class Timetable Generator")
    st.write("Generate and visualize class timetables for multiple sections")
    
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
        
        for section in selected_sections:
            st.header(f"Section {section} Timetable")
            display_timetable(timetables[section])
            
            csv = timetables[section].to_csv().encode('utf-8')
            st.download_button(
                label=f"Download Section {section} Timetable",
                data=csv,
                file_name=f'timetable_section_{section}.csv',
                mime='text/csv'
            )
        
        st.sidebar.success("✅ Timetables generated successfully!")

if __name__ == "__main__":
    main()