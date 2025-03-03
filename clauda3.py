import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
from typing import List, Dict, Tuple
from collections import defaultdict

# Configuration (keeping your original constants)
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
SLOTS_PER_DAY = 7  # Excluding lunch break

class TimetableGenerator:
    def __init__(self, section: str):
        self.section = section
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
        # Create variables for each subject-day-slot combination
        self.slots = {}
        self.lab_slots = {}
        
        # Initialize variables
        self._initialize_variables()
        
        # Add constraints
        self._add_basic_constraints()
        self._add_lab_constraints()
        self._add_faculty_constraints()
        self._add_distribution_constraints()

    def _initialize_variables(self):
        # Regular subjects
        for subject in [s for s in SUBJECTS if not s['is_lab']]:
            for day in range(len(DAYS)):
                for slot in range(SLOTS_PER_DAY):
                    self.slots[(subject['name'], day, slot)] = self.model.NewBoolVar(
                        f'{subject["name"]}_{day}_{slot}'
                    )

        # Lab subjects
        for subject in [s for s in SUBJECTS if s['is_lab']]:
            for batch in LAB_BATCHES:
                for day in range(len(DAYS)):
                    # Labs need 3 consecutive slots
                    for slot in range(SLOTS_PER_DAY - 2):
                        self.lab_slots[(subject['name'], batch, day, slot)] = self.model.NewBoolVar(
                            f'{subject["name"]}_batch{batch}_{day}_{slot}'
                        )

    def _add_basic_constraints(self):
        # Ensure each slot has at most one subject
        for day in range(len(DAYS)):
            for slot in range(SLOTS_PER_DAY):
                slot_vars = []
                # Regular subjects
                for subject in [s for s in SUBJECTS if not s['is_lab']]:
                    slot_vars.append(self.slots[(subject['name'], day, slot)])
                # Lab subjects
                for subject in [s for s in SUBJECTS if s['is_lab']]:
                    for batch in LAB_BATCHES:
                        # Add lab slots that could occupy this time
                        for i in range(max(0, slot - 2), slot + 1):
                            if (subject['name'], batch, day, i) in self.lab_slots:
                                slot_vars.append(self.lab_slots[(subject['name'], batch, day, i)])
                
                self.model.Add(sum(slot_vars) <= 1)

        # Ensure weekly hours are met for regular subjects
        for subject in [s for s in SUBJECTS if not s['is_lab']]:
            subject_slots = []
            for day in range(len(DAYS)):
                for slot in range(SLOTS_PER_DAY):
                    subject_slots.append(self.slots[(subject['name'], day, slot)])
            self.model.Add(sum(subject_slots) == subject['weekly_hours'])

    def _add_lab_constraints(self):
        # Ensure each lab happens exactly once per week per batch
        # and takes exactly 3 consecutive slots
        for subject in [s for s in SUBJECTS if s['is_lab']]:
            for batch in LAB_BATCHES:
                lab_occurrences = []
                for day in range(len(DAYS)):
                    for slot in range(SLOTS_PER_DAY - 2):
                        lab_occurrences.append(
                            self.lab_slots[(subject['name'], batch, day, slot)]
                        )
                self.model.Add(sum(lab_occurrences) == 1)

    def _add_faculty_constraints(self):
        # Ensure faculty members don't have conflicts
        for day in range(len(DAYS)):
            for slot in range(SLOTS_PER_DAY):
                for faculty in set(sum([s['faculty'] for s in SUBJECTS], [])):
                    faculty_slots = []
                    # Regular subjects
                    for subject in [s for s in SUBJECTS if not s['is_lab'] and faculty in s['faculty']]:
                        faculty_slots.append(self.slots[(subject['name'], day, slot)])
                    # Lab subjects
                    for subject in [s for s in SUBJECTS if s['is_lab'] and faculty in s['faculty']]:
                        for batch in LAB_BATCHES:
                            for i in range(max(0, slot - 2), slot + 1):
                                if (subject['name'], batch, day, i) in self.lab_slots:
                                    faculty_slots.append(self.lab_slots[(subject['name'], batch, day, i)])
                    
                    if faculty_slots:
                        self.model.Add(sum(faculty_slots) <= 1)

    def _add_distribution_constraints(self):
        # Try to distribute subjects across days
        for subject in [s for s in SUBJECTS if not s['is_lab']]:
            for day in range(len(DAYS)):
                day_slots = []
                for slot in range(SLOTS_PER_DAY):
                    day_slots.append(self.slots[(subject['name'], day, slot)])
                # Maximum 2 sessions per day for any subject
                self.model.Add(sum(day_slots) <= 2)

    def solve(self) -> pd.DataFrame:
        # Solve the model
        status = self.solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Create timetable DataFrame
            timetable = pd.DataFrame(
                index=DAYS,
                columns=["9:00-9:50", "9:50-10:40", "10:40-11:30", "11:30-12:20", 
                        "12:20-1:00", "1:00-1:50", "1:50-2:40", "2:40-3:30"]
            )
            timetable.fillna("-")
            timetable["12:20-1:00"] = "LUNCH BREAK"

            # Fill regular subjects
            for subject in [s for s in SUBJECTS if not s['is_lab']]:
                for day in range(len(DAYS)):
                    for slot in range(SLOTS_PER_DAY):
                        if self.solver.Value(self.slots[(subject['name'], day, slot)]):
                            col = timetable.columns[slot if slot < 4 else slot + 1]
                            timetable.at[DAYS[day], col] = f"{SUBJECT_SHORTCUTS[subject['name']]} ({subject['faculty'][0]})"

            # Fill lab subjects
            for subject in [s for s in SUBJECTS if s['is_lab']]:
                for batch in LAB_BATCHES:
                    for day in range(len(DAYS)):
                        for slot in range(SLOTS_PER_DAY - 2):
                            if (subject['name'], batch, day, slot) in self.lab_slots and \
                               self.solver.Value(self.lab_slots[(subject['name'], batch, day, slot)]):
                                faculty = ", ".join(subject['faculty'][:2])
                                entry = f"{SUBJECT_SHORTCUTS[subject['name']]} (B{batch}: {faculty})"
                                for i in range(3):
                                    col = timetable.columns[slot + i if slot + i < 4 else slot + i + 1]
                                    timetable.at[DAYS[day], col] = entry

            return timetable
        else:
            raise ValueError("No solution found!")

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
        for section in selected_sections:
            st.header(f"Section {section} Timetable")
            
            try:
                generator = TimetableGenerator(section)
                timetable = generator.solve()
                
                # Display timetable
                st.dataframe(
                    timetable.style.apply(lambda x: ['background-color: #f0f0f0' if v == 'LUNCH BREAK' else '' for v in x]),
                    use_container_width=True,
                    height=300
                )
                
                # Add download button
                csv = timetable.to_csv().encode('utf-8')
                st.download_button(
                    label=f"Download Section {section} Timetable",
                    data=csv,
                    file_name=f'timetable_section_{section}.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"Error generating timetable for Section {section}: {str(e)}")
        
        st.sidebar.success("✅ Timetables generated successfully!")

if __name__ == "__main__":
    main()