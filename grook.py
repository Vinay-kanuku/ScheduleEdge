import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple
from ortools.sat.python import cp_model
from dataclasses import dataclass
from collections import defaultdict

# Define Subject dataclass
@dataclass(frozen=True)
class Subject:
    name: str
    weekly_hours: int
    faculty: tuple
    is_lab: bool

# Subject shortcuts for timetable display
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

# List of subjects with their weekly hours, faculty, and lab status
SUBJECTS = [
    Subject("Software Engineering", 4, ("PKA",), False),
    Subject("Artificial Intelligence", 5, ("Dr. GP",), False),
    Subject("Web Programming", 4, ("KS",), False),
    Subject("Data Warehousing and Data Mining", 4, ("Dr. NR",), False),
    Subject("Introduction to Data Science", 3, ("CHS",), False),
    Subject("Internet of Things and Applications Lab", 3, ("Dr. KRK", "LIB"), True),
    Subject("Web Programming Lab", 3, ("KS", "KB", "Dr. KP"), True),
    Subject("Artificial Intelligence Lab Using Python", 3, ("BHKD", "Dr. GP"), True),
    Subject("Constitution of India", 2, ("PPS",), False)
]

# Constants
SECTIONS = ["A", "B", "C"]
DAYS = ["Monday", "Tuesday", "Thursday", "Friday", "Saturday"]
TIME_SLOTS = [
    "9:00-9:50", "9:50-10:40", "10:40-11:30", "11:30-12:20",
    "LUNCH",
    "1:00-1:55", "1:55-2:50", "2:50-3:30"
]

class TimeTableGenerator:
    def __init__(self, section: str, faculty_schedule: defaultdict):
        """Initialize the timetable generator for a section."""
        self.section = section
        self.faculty_schedule = faculty_schedule
        self.model = cp_model.CpModel()
        self.slots = {}
        # Define possible lab periods: 3 consecutive slots in morning or afternoon
        self.lab_periods = []
        for day in DAYS:
            self.lab_periods.append((day, [0, 1, 2]))  # 9:00-11:30
            self.lab_periods.append((day, [1, 2, 3]))  # 9:50-12:20
            self.lab_periods.append((day, [5, 6, 7]))  # 1:00-3:30

    def generate(self) -> pd.DataFrame:
        """Generate the timetable for the section."""
        # Create decision variables
        for day in DAYS:
            for slot in TIME_SLOTS:
                if slot != "LUNCH":
                    for subject in SUBJECTS:
                        self.slots[(day, slot, subject.name)] = self.model.NewBoolVar(
                            f'slot_{day}_{slot}_{subject.name}'
                        )

        # Add constraints
        self._add_faculty_availability_constraints()
        self._add_single_subject_per_slot_constraint()
        self._add_weekly_hours_constraint()
        self._add_lab_period_constraints()
        self._add_consecutive_theory_constraint()
        self._add_no_single_day_full_assignment()

        # Solve the model
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 120  # Increased timeout to 120 seconds
        status = solver.Solve(self.model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return self._create_timetable_df(solver)
        else:
            st.write(f"Failed to find solution for Section {self.section}. Status: {status}")
            return None

    def _add_faculty_availability_constraints(self):
        """Prevent scheduling a subject if its faculty is already busy."""
        for day in DAYS:
            for slot in TIME_SLOTS:
                if slot != "LUNCH":
                    for subject in SUBJECTS:
                        if any((day, slot) in self.faculty_schedule[fac] for fac in subject.faculty):
                            self.model.Add(self.slots[(day, slot, subject.name)] == 0)

    def _add_single_subject_per_slot_constraint(self):
        """Ensure only one subject per slot."""
        for day in DAYS:
            for slot in TIME_SLOTS:
                if slot != "LUNCH":
                    slot_vars = [self.slots[(day, slot, subject.name)] for subject in SUBJECTS]
                    self.model.Add(sum(slot_vars) <= 1)

    def _add_weekly_hours_constraint(self):
        """Ensure each subject meets its weekly hours."""
        for subject in SUBJECTS:
            subject_slots = [self.slots[(day, slot, subject.name)] 
                             for day in DAYS for slot in TIME_SLOTS if slot != "LUNCH"]
            self.model.Add(sum(subject_slots) == subject.weekly_hours)

    def _add_lab_period_constraints(self):
        """Schedule labs in exactly one block of 3 consecutive slots."""
        for subject in [s for s in SUBJECTS if s.is_lab]:
            period_vars = []
            for day, slot_indices in self.lab_periods:
                period_var = self.model.NewBoolVar(f'lab_period_{subject.name}_{day}_{slot_indices}')
                period_vars.append(period_var)
                for idx in slot_indices:
                    slot_time = TIME_SLOTS[idx]
                    self.model.Add(self.slots[(day, slot_time, subject.name)] == 1).OnlyEnforceIf(period_var)
            self.model.Add(sum(period_vars) == 1)  # Exactly one period per lab

    def _add_consecutive_theory_constraint(self):
        """Ensure theory classes are consecutive within morning/afternoon blocks."""
        morning_slots = ["9:00-9:50", "9:50-10:40", "10:40-11:30", "11:30-12:20"]
        afternoon_slots = ["1:00-1:55", "1:55-2:50", "2:50-3:30"]
        for subject in [s for s in SUBJECTS if not s.is_lab]:
            for day in DAYS:
                morning_vars = [self.slots[(day, slot, subject.name)] for slot in morning_slots]
                for i in range(len(morning_vars) - 1):
                    self.model.Add(morning_vars[i] >= morning_vars[i + 1])
                afternoon_vars = [self.slots[(day, slot, subject.name)] for slot in afternoon_slots]
                for i in range(len(afternoon_vars) - 1):
                    self.model.Add(afternoon_vars[i] >= afternoon_vars[i + 1])

    def _add_no_single_day_full_assignment(self):
        """Limit non-lab subjects to 2 slots per day."""
        for subject in [s for s in SUBJECTS if not s.is_lab]:
            for day in DAYS:
                subject_slots = [self.slots[(day, slot, subject.name)] for slot in TIME_SLOTS if slot != "LUNCH"]
                self.model.Add(sum(subject_slots) <= 2)

    def _create_timetable_df(self, solver: cp_model.CpSolver) -> pd.DataFrame:
        """Create a DataFrame representation of the timetable."""
        df = pd.DataFrame(index=DAYS, columns=TIME_SLOTS)
        df.fillna("-")
        for day in DAYS:
            for slot in TIME_SLOTS:
                if slot == "LUNCH":
                    df.at[day, slot] = "LUNCH BREAK"
                    continue
                for subject in SUBJECTS:
                    if solver.Value(self.slots[(day, slot, subject.name)]):
                        entry = SUBJECT_SHORTCUTS[subject.name]
                        if subject.is_lab:
                            entry += f" ({', '.join(subject.faculty[:2])})"
                        else:
                            entry += f" ({subject.faculty[0]})"
                        df.at[day, slot] = entry
                        break
        return df

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Timetable Generator", layout="wide")
    st.title("📚 Class Timetable Generator")
    st.write("Generate optimized class timetables for multiple sections")

    selected_sections = st.sidebar.multiselect(
        "Select Sections",
        SECTIONS,
        default=SECTIONS
    )

    if st.button("Generate Timetables"):
        with st.spinner("Generating timetables..."):
            faculty_schedule = defaultdict(set)  # Global faculty availability tracker
            for section in selected_sections:
                st.write(f"Generating timetable for Section {section}...")
                generator = TimeTableGenerator(section, faculty_schedule)
                timetable = generator.generate()
                if timetable is not None:
                    # Update faculty_schedule with this section's assignments
                    for day in timetable.index:
                        for slot in timetable.columns:
                            if slot != "LUNCH" and timetable.at[day, slot] != "-":
                                for subj in SUBJECTS:
                                    if SUBJECT_SHORTCUTS[subj.name] in timetable.at[day, slot]:
                                        for fac in subj.faculty:
                                            faculty_schedule[fac].add((day, slot))
                    st.header(f"Section {section} Timetable")
                    st.dataframe(timetable)
                    csv = timetable.to_csv().encode('utf-8')
                    st.download_button(
                        label=f"Download Section {section} Timetable",
                        data=csv,
                        file_name=f'timetable_section_{section}.csv',
                        mime='text/csv'
                    )
                else:
                    st.error(f"Could not generate valid timetable for Section {section}")

if __name__ == "__main__":
    main()