import streamlit as st
from ortools.sat.python import cp_model
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict

# --- Helper Functions ---
def create_slots_map():
    """Create a mapping of time slots to indices."""
    morning_slots = [
        "9:00-9:50", "9:50-10:40", "10:40-11:30", "11:30-12:20"
    ]
    lunch_break = "12:20-1:00"
    afternoon_slots = [
        "1:00-1:50", "1:50-2:40", "2:40-3:30"
    ]
    all_slots = morning_slots + [lunch_break] + afternoon_slots
    return {slot: idx for idx, slot in enumerate(all_slots)}

def create_days_map():
    """Create a mapping of days to indices."""
    days = ["Monday", "Tuesday", "Thursday", "Friday", "Saturday"]
    return {day: idx for idx, day in enumerate(days)}

# --- Core Logic ---
class TimeTableSolver:
    def __init__(self, section: str):
        self.section = section
        self.model = cp_model.CpModel()
        self.slots_map = create_slots_map()
        self.days_map = create_days_map()
        self.solver = cp_model.CpSolver()
        # Dictionaries to store decision variables
        self.variables = {}      # For theory subjects
        self.lab_variables = {}  # For lab subjects

    def create_variables(self):
        """Create boolean decision variables for all subjects."""
        # For theory subjects
        for subject in [s for s in SUBJECTS if not s['is_lab']]:
            for day in self.days_map:
                for slot in self.slots_map:
                    if slot == "12:20-1:00":  # Skip lunch break
                        continue
                    self.variables[(subject['name'], day, slot)] = \
                        self.model.NewBoolVar(f"{subject['name']}_{day}_{slot}")
        # For lab subjects
        for subject in [s for s in SUBJECTS if s['is_lab']]:
            for batch in LAB_BATCHES:
                for day in self.days_map:
                    # Labs can only start at specific slots
                    valid_start_slots = ["9:00-9:50", "1:00-1:50"]
                    for start_slot in valid_start_slots:
                        self.lab_variables[(subject['name'], batch, day, start_slot)] = \
                            self.model.NewBoolVar(f"{subject['name']}_B{batch}_{day}_{start_slot}")

    def add_subject_hours_constraint(self):
        """Ensure each subject gets exactly its fixed weekly hours."""
        for subject in SUBJECTS:
            if not subject['is_lab']:
                subject_slots = []
                for day in self.days_map:
                    for slot in self.slots_map:
                        if slot == "12:20-1:00":
                            continue
                        subject_slots.append(self.variables[(subject['name'], day, slot)])
                self.model.Add(sum(subject_slots) == subject['weekly_hours'])
            else:
                # For labs, each batch must have exactly one session (assumed 2.5 hours)
                for batch in LAB_BATCHES:
                    lab_sessions = []
                    for day in self.days_map:
                        for start_slot in ["9:00-9:50", "1:00-1:50"]:
                            lab_sessions.append(self.lab_variables[(subject['name'], batch, day, start_slot)])
                    self.model.Add(sum(lab_sessions) == 1)

    def add_single_subject_per_slot_constraint(self):
        """Ensure that at most one subject (theory or lab) occupies a time slot."""
        for day in self.days_map:
            for slot in self.slots_map:
                if slot == "12:20-1:00":
                    continue
                slot_subjects = []
                # Theory subjects for this slot
                for subject in [s for s in SUBJECTS if not s['is_lab']]:
                    slot_subjects.append(self.variables[(subject['name'], day, slot)])
                # Lab subjects can only start at designated slots
                if slot in ["9:00-9:50", "1:00-1:50"]:
                    for subject in [s for s in SUBJECTS if s['is_lab']]:
                        for batch in LAB_BATCHES:
                            slot_subjects.append(self.lab_variables[(subject['name'], batch, day, slot)])
                self.model.Add(sum(slot_subjects) <= 1)

    def add_consecutive_slots_constraint(self):
        """If a theory subject is scheduled for multiple slots in a day, they must be consecutive."""
        for subject in [s for s in SUBJECTS if not s['is_lab']]:
            for day in self.days_map:
                day_slots = []
                # Gather all slots for this subject on the day (excluding lunch)
                for slot in self.slots_map:
                    if slot == "12:20-1:00":
                        continue
                    day_slots.append(self.variables[(subject['name'], day, slot)])
                # Enforce consecutive scheduling if more than one slot is used
                if len(day_slots) > 1:
                    for i in range(len(day_slots) - 1):
                        self.model.Add(day_slots[i] <= day_slots[i + 1])

    def add_lab_constraints(self):
        """Add lab-specific constraints."""
        for subject in [s for s in SUBJECTS if s['is_lab']]:
            for day in self.days_map:
                for start_slot in ["9:00-9:50", "1:00-1:50"]:
                    # Ensure no two lab batches for the same subject occur at the same start slot
                    batch_vars = []
                    for batch in LAB_BATCHES:
                        batch_vars.append(self.lab_variables[(subject['name'], batch, day, start_slot)])
                    self.model.Add(sum(batch_vars) <= 1)
                    # If a lab starts, block other theory sessions during its block period
                    if start_slot == "9:00-9:50":
                        # Assume lab occupies three consecutive morning slots:
                        morning_slots = ["9:00-9:50", "9:50-10:40", "10:40-11:30"]
                        # For each of these slots, no theory subject can be scheduled
                        for slot in morning_slots:
                            for other in [s for s in SUBJECTS if not s['is_lab']]:
                                self.model.Add(self.variables[(other['name'], day, slot)] == 0)
                    elif start_slot == "1:00-1:50":
                        # Assume lab occupies three consecutive afternoon slots:
                        afternoon_slots = ["1:00-1:50", "1:50-2:40", "2:40-3:30"]
                        for slot in afternoon_slots:
                            for other in [s for s in SUBJECTS if not s['is_lab']]:
                                self.model.Add(self.variables[(other['name'], day, slot)] == 0)

    def add_no_empty_slots_constraint(self):
        """Ensure that every slot is occupied by some subject."""
        for day in self.days_map:
            for slot in self.slots_map:
                if slot == "12:20-1:00":
                    continue
                slot_subjects = []
                for subject in [s for s in SUBJECTS if not s['is_lab']]:
                    slot_subjects.append(self.variables[(subject['name'], day, slot)])
                if slot in ["9:00-9:50", "1:00-1:50"]:
                    for subject in [s for s in SUBJECTS if s['is_lab']]:
                        for batch in LAB_BATCHES:
                            slot_subjects.append(self.lab_variables[(subject['name'], batch, day, slot)])
                self.model.Add(sum(slot_subjects) == 1)

    def solve(self) -> pd.DataFrame:
        """Set up constraints, solve the model, and return the timetable as a DataFrame."""
        self.create_variables()
        self.add_subject_hours_constraint()
        self.add_single_subject_per_slot_constraint()
        self.add_consecutive_slots_constraint()
        self.add_lab_constraints()
        self.add_no_empty_slots_constraint()
        
        status = self.solver.Solve(self.model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return self.create_timetable_df()
        else:
            raise Exception("No solution found!")

    def create_timetable_df(self) -> pd.DataFrame:
        """Generate a timetable DataFrame from the solver's solution."""
        df = pd.DataFrame(
            index=list(self.days_map.keys()),
            columns=list(self.slots_map.keys())
        )
        df = df.fillna("-")
        # Set lunch break
        df["12:20-1:00"] = "LUNCH BREAK"
        
        # Fill in theory subjects
        for (subject, day, slot), var in self.variables.items():
            if self.solver.Value(var) == 1:
                faculty = next(s['faculty'][0] for s in SUBJECTS if s['name'] == subject)
                df.at[day, slot] = f"{SUBJECT_SHORTCUTS[subject]} ({faculty})"
        
        # Fill in lab subjects
        for (subject, batch, day, start_slot), var in self.lab_variables.items():
            if self.solver.Value(var) == 1:
                # Get up to two faculty names
                faculty_list = next(s['faculty'] for s in SUBJECTS if s['name'] == subject)
                faculty = ", ".join(faculty_list[:2])
                if start_slot == "9:00-9:50":
                    slots = ["9:00-9:50", "9:50-10:40", "10:40-11:30"]
                else:
                    slots = ["1:00-1:50", "1:50-2:40", "2:40-3:30"]
                for slot in slots:
                    df.at[day, slot] = f"{SUBJECT_SHORTCUTS[subject]} (B{batch}: {faculty})"
        return df

# --- Data Configuration ---
# Define subjects as dictionaries.
SUBJECTS = [
    {"name": "Software Engineering", "weekly_hours": 4, "faculty": ["PKA"], "is_lab": False},
    {"name": "Artificial Intelligence", "weekly_hours": 5, "faculty": ["Dr. GP"], "is_lab": False},
    {"name": "Web Programming", "weekly_hours": 4, "faculty": ["KS"], "is_lab": False},
    {"name": "Data Warehousing and Data Mining", "weekly_hours": 4, "faculty": ["Dr. NR"], "is_lab": False},
    {"name": "Introduction to Data Science", "weekly_hours": 3, "faculty": ["CHS"], "is_lab": False},
    {"name": "Internet of Things and Applications Lab", "weekly_hours": 3, "faculty": ["Dr. KRK", "LIB"], "is_lab": True},
    {"name": "Web Programming Lab", "weekly_hours": 3, "faculty": ["KS", "KB", "Dr. KP"], "is_lab": True},
    {"name": "Artificial Intelligence Lab Using Python", "weekly_hours": 3, "faculty": ["BHKD", "Dr. GP"], "is_lab": True},
    {"name": "Constitution of India", "weekly_hours": 2, "faculty": ["PPS"], "is_lab": False}
]
# Define lab batches (for example, two batches)
LAB_BATCHES = [1, 2]
# Define subject shortcuts for display.
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

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Timetable Generator", layout="wide")
    st.title("📚 Class Timetable Generator")
    st.write("Generate optimized class timetables for multiple sections.")
    
    # Allow the user to select a section
    section = st.sidebar.selectbox("Select Section", ["A", "B", "C"])
    
    if st.button("Generate Timetable"):
        try:
            solver = TimeTableSolver(section)
            timetable_df = solver.solve()
            st.header(f"Section {section} Timetable")
            st.dataframe(timetable_df)
            
            csv = timetable_df.to_csv().encode("utf-8")
            st.download_button(
                label="Download Timetable CSV",
                data=csv,
                file_name=f"timetable_section_{section}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
