import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
from typing import List, Dict, Tuple
from collections import defaultdict

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

# Time slots with proper indexing
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

class TimeTableSolver:
    def __init__(self, section: str):
        self.section = section
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
        # Create slot indices for easier constraint handling
        self.slot_indices = {slot: idx for idx, slot in enumerate(TIME_SLOTS)}
        self.day_indices = {day: idx for idx, day in enumerate(DAYS)}
        
        # Initialize variables
        self.theory_vars = {}  # (subject, day, slot) -> var
        self.lab_vars = {}     # (subject, batch, day, start_slot) -> var
        
    def create_variables(self):
        """Create decision variables for the CP-SAT solver"""
        # Theory subject variables
        for subject in [s for s in SUBJECTS if not s['is_lab']]:
            for day in DAYS:
                for slot in TIME_SLOTS:
                    if slot != LUNCH_BREAK:
                        self.theory_vars[(subject['name'], day, slot)] = \
                            self.model.NewBoolVar(f'{subject["name"]}_{day}_{slot}')
        
        # Lab subject variables (only for valid start slots)
        for subject in [s for s in SUBJECTS if s['is_lab']]:
            for batch in LAB_BATCHES:
                for day in DAYS:
                    # Labs can only start at 9:00 or 1:00
                    for start_slot in ["9:00-9:50", "1:00-1:50"]:
                        self.lab_vars[(subject['name'], batch, day, start_slot)] = \
                            self.model.NewBoolVar(f'{subject["name"]}_B{batch}_{day}_{start_slot}')

    def add_weekly_hours_constraint(self):
        """Each subject must have exactly its required weekly hours"""
        for subject in SUBJECTS:
            if not subject['is_lab']:
                # Sum all slots for theory subjects
                slots = []
                for day in DAYS:
                    for slot in TIME_SLOTS:
                        if slot != LUNCH_BREAK:
                            slots.append(self.theory_vars[(subject['name'], day, slot)])
                self.model.Add(sum(slots) == subject['weekly_hours'])
            else:
                # Each lab batch needs one 2.5-hour session
                for batch in LAB_BATCHES:
                    sessions = []
                    for day in DAYS:
                        for start_slot in ["9:00-9:50", "1:00-1:50"]:
                            sessions.append(self.lab_vars[(subject['name'], batch, day, start_slot)])
                    self.model.Add(sum(sessions) == 1)

    def add_single_subject_per_slot_constraint(self):
        """Only one subject can be scheduled in a slot"""
        for day in DAYS:
            for slot in TIME_SLOTS:
                if slot == LUNCH_BREAK:
                    continue
                
                # Collect all subjects that could be in this slot
                slot_vars = []
                
                # Theory subjects
                for subject in [s for s in SUBJECTS if not s['is_lab']]:
                    slot_vars.append(self.theory_vars[(subject['name'], day, slot)])
                
                # Lab subjects that could occupy this slot
                for subject in [s for s in SUBJECTS if s['is_lab']]:
                    for batch in LAB_BATCHES:
                        # Check if this slot could be part of a lab session
                        if slot in ["9:00-9:50", "9:50-10:40", "10:40-11:30"] and \
                           ("9:00-9:50", day) in [(s, d) for s, b, d, _ in self.lab_vars.keys()]:
                            slot_vars.append(self.lab_vars[(subject['name'], batch, day, "9:00-9:50")])
                        elif slot in ["1:00-1:50", "1:50-2:40", "2:40-3:30"] and \
                             ("1:00-1:50", day) in [(s, d) for s, b, d, _ in self.lab_vars.keys()]:
                            slot_vars.append(self.lab_vars[(subject['name'], batch, day, "1:00-1:50")])
                
                # Exactly one subject must be scheduled (no empty slots)
                self.model.Add(sum(slot_vars) == 1)

    def add_consecutive_slots_constraint(self):
        """Theory classes must be consecutive (even across lunch break)"""
        for subject in [s for s in SUBJECTS if not s['is_lab']]:
            for day in DAYS:
                # Get all slots for this day
                day_slots = []
                for slot in TIME_SLOTS:
                    if slot != LUNCH_BREAK:
                        day_slots.append((slot, self.theory_vars[(subject['name'], day, slot)]))
                
                # For each pair of slots used by this subject
                for i in range(len(day_slots)):
                    for j in range(i + 1, len(day_slots)):
                        slot1, var1 = day_slots[i]
                        slot2, var2 = day_slots[j]
                        
                        # If both slots are used, all slots between must be used
                        middle_slots = []
                        for k in range(i + 1, j):
                            if TIME_SLOTS[k] != LUNCH_BREAK:
                                middle_slots.append(self.theory_vars[(subject['name'], day, TIME_SLOTS[k])])
                        
                        if middle_slots:
                            # If slot1 and slot2 are used, all middle slots must be used
                            self.model.Add(sum(middle_slots) >= j - i - 1).OnlyEnforceIf([var1, var2])

    def add_lab_constraints(self):
        """Add constraints specific to lab sessions"""
        for day in DAYS:
            # Morning lab constraints
            morning_lab_vars = []
            for subject in [s for s in SUBJECTS if s['is_lab']]:
                for batch in LAB_BATCHES:
                    if ("9:00-9:50", day) in [(s, d) for s, b, d, _ in self.lab_vars.keys()]:
                        morning_lab_vars.append(
                            self.lab_vars[(subject['name'], batch, day, "9:00-9:50")]
                        )
            
            # Only one morning lab session
            if morning_lab_vars:
                self.model.Add(sum(morning_lab_vars) <= 1)
            
            # Afternoon lab constraints
            afternoon_lab_vars = []
            for subject in [s for s in SUBJECTS if s['is_lab']]:
                for batch in LAB_BATCHES:
                    if ("1:00-1:50", day) in [(s, d) for s, b, d, _ in self.lab_vars.keys()]:
                        afternoon_lab_vars.append(
                            self.lab_vars[(subject['name'], batch, day, "1:00-1:50")]
                        )
            
            # Only one afternoon lab session
            if afternoon_lab_vars:
                self.model.Add(sum(afternoon_lab_vars) <= 1)

    def solve(self) -> pd.DataFrame:
        """Solve the constraint satisfaction problem and return the timetable"""
        # Create and add all constraints
        self.create_variables()
        self.add_weekly_hours_constraint()
        self.add_single_subject_per_slot_constraint()
        self.add_consecutive_slots_constraint()
        self.add_lab_constraints()
        
        # Solve the model
        status = self.solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return self.create_timetable_df()
        else:
            raise Exception("No solution found!")

    def create_timetable_df(self) -> pd.DataFrame:
        """Create a DataFrame from the solved model"""
        df = pd.DataFrame(index=DAYS, columns=TIME_SLOTS)
        df.fillna("-")
        
        # Set lunch break
        df[LUNCH_BREAK] = "LUNCH BREAK"
        
        # Fill theory subjects
        for (subject, day, slot), var in self.theory_vars.items():
            if self.solver.Value(var) == 1:
                faculty = next(s['faculty'][0] for s in SUBJECTS if s['name'] == subject)
                df.at[day, slot] = f"{SUBJECT_SHORTCUTS[subject]} ({faculty})"
        
        # Fill lab subjects
        for (subject, batch, day, start_slot), var in self.lab_vars.items():
            if self.solver.Value(var) == 1:
                faculty = ", ".join(
                    next(s['faculty'] for s in SUBJECTS if s['name'] == subject)[:2]
                )
                if start_slot == "9:00-9:50":
                    slots = ["9:00-9:50", "9:50-10:40", "10:40-11:30"]
                else:
                    slots = ["1:00-1:50", "1:50-2:40", "2:40-3:30"]
                
                for slot in slots:
                    df.at[day, slot] = f"{SUBJECT_SHORTCUTS[subject]} (B{batch}: {faculty})"
        
        return df

def display_timetable(df: pd.DataFrame) -> None:
    """Display the timetable with proper styling"""
    def style_lunch_break(val):
        if val == "LUNCH BREAK":
            return "background-color: #f0f0f0; font-style: italic;"
        return ""
    
    styled_df = df.style.applymap(style_lunch_break)
    
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=300,
        column_config={col: st.column_config.Column(width="medium") for col in df.columns}
    )

def main():
    st.set_page_config(page_title="Class Timetable Generator", layout="wide")
    
    st.title("📚 Class Timetable Generator")
    st.write("Generate optimized class timetables using constraint programming")
    
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
            progress_bar.progress((i) / len(selected_sections), 
                                text=f"Generating timetable for Section {section}...")
            
            try:
                solver = TimeTableSolver(section)
                timetables[section] = solver.solve()
                
                progress_bar.progress((i + 1) / len(selected_sections), 
                                    text=f"Completed Section {section}")
            except Exception as e:
                st.error(f"Error generating timetable for Section {section}: {str(e)}")
                continue
        
        progress_container.empty()
        progress_bar.empty()
        
        # Display timetables
        for section in timetables:
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
        
        if timetables:
            st.sidebar.success("✅ Timetables generated successfully!")

if __name__ == "__main__":
    main()