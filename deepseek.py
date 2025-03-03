import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple
from ortools.sat.python import cp_model
import multiprocessing
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from collections import defaultdict

@dataclass(frozen=True)
class Subject:
    name: str
    weekly_hours: int
    faculty: tuple
    is_lab: bool

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

# Adjusted weekly_hours to sum to 35 (original sum was 31)
SUBJECTS = [
    Subject("Software Engineering", 4, tuple(["PKA"]), False),
    Subject("Artificial Intelligence", 5, tuple(["Dr. GP"]), False),
    Subject("Web Programming", 5, tuple(["KS"]), False),  # Increased from 4 to 5
    Subject("Data Warehousing and Data Mining", 4, tuple(["Dr. NR"]), False),
    Subject("Introduction to Data Science", 4, tuple(["CHS"]), False),  # Increased from 3 to 4
    Subject("Internet of Things and Applications Lab", 3, tuple(["Dr. KRK", "LIB"]), True),
    Subject("Web Programming Lab", 3, tuple(["KS", "KB", "Dr. KP"]), True),
    Subject("Artificial Intelligence Lab Using Python", 3, tuple(["BHKD", "Dr. GP"]), True),
    Subject("Constitution of India", 3, tuple(["PPS"]), False)  # Increased from 2 to 3
]

SECTIONS = ["A", "B", "C"]
DAYS = ["Monday", "Tuesday", "Thursday", "Friday", "Saturday"]
TIME_SLOTS = [
    "9:00-9:50", "9:50-10:40", "10:40-11:30", "11:30-12:20",
    "LUNCH",
    "1:00-1:55", "1:55-2:50", "2:50-3:30"
]

class TimeTableGenerator:
    def __init__(self, section: str):
        self.section = section
        self.model = cp_model.CpModel()
        self.slots = {}  # Will hold our decision variables
        
    def generate(self) -> pd.DataFrame:
        # Create variables
        for day in DAYS:
            for slot in TIME_SLOTS:
                if slot != "LUNCH":
                    for subject in SUBJECTS:
                        # Create binary variable for each possible assignment
                        self.slots[(day, slot, subject.name)] = self.model.NewBoolVar(
                            f'slot_{day}_{slot}_{subject.name}_{self.section}'
                        )

        # Add constraints
        self._add_single_subject_per_slot_constraint()
        self._add_weekly_hours_constraint()
        self._add_lab_constraints()
        self._add_consecutive_theory_constraint()
        self._add_no_single_day_full_assignment()
        self._prevent_lab_overlap()

        # Solve the model
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60
        status = solver.Solve(self.model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return self._create_timetable_df(solver)
        return None

    def _add_single_subject_per_slot_constraint(self):
        """Ensure exactly one subject per time slot"""
        for day in DAYS:
            for slot in TIME_SLOTS:
                if slot != "LUNCH":
                    slot_vars = []
                    for subject in SUBJECTS:
                        slot_vars.append(self.slots[(day, slot, subject.name)])
                    self.model.Add(sum(slot_vars) == 1)  # Changed to ==1

    def _add_weekly_hours_constraint(self):
        """Ensure each subject gets its required weekly hours"""
        for subject in SUBJECTS:
            subject_slots = []
            for day in DAYS:
                for slot in TIME_SLOTS:
                    if slot != "LUNCH":
                        subject_slots.append(self.slots[(day, slot, subject.name)])
            self.model.Add(sum(subject_slots) == subject.weekly_hours)

    def _add_lab_constraints(self):
        """Handle lab-specific constraints"""
        afternoon_slots = ["1:00-1:55", "1:55-2:50", "2:50-3:30"]
        
        for subject in [s for s in SUBJECTS if s.is_lab]:
            # Labs only in afternoon
            for day in DAYS:
                for slot in TIME_SLOTS:
                    if slot != "LUNCH" and slot not in afternoon_slots:
                        self.model.Add(self.slots[(day, slot, subject.name)] == 0)
                
                # If lab starts, it must take all afternoon slots
                lab_slots = []
                for slot in afternoon_slots:
                    lab_slots.append(self.slots[(day, slot, subject.name)])
                
                # Either all slots are used or none are
                for i in range(len(lab_slots) - 1):
                    self.model.Add(lab_slots[i] == lab_slots[i + 1])

    def _prevent_lab_overlap(self):
        """Prevent multiple labs from being scheduled on the same day"""
        labs = [s for s in SUBJECTS if s.is_lab]
        for day in DAYS:
            lab_day_vars = []
            for lab in labs:
                # Check if lab is scheduled on this day (any afternoon slot)
                lab_scheduled = self.model.NewBoolVar(f'lab_{lab.name}_{day}_scheduled')
                # Lab is scheduled if all afternoon slots are assigned
                self.model.AddBoolAnd([
                    self.slots[(day, slot, lab.name)] for slot in ["1:00-1:55", "1:55-2:50", "2:50-3:30"]
                ]).OnlyEnforceIf(lab_scheduled)
                self.model.AddBoolOr([
                    self.slots[(day, slot, lab.name)].Not() for slot in ["1:00-1:55", "1:55-2:50", "2:50-3:30"]
                ]).OnlyEnforceIf(lab_scheduled.Not())
                lab_day_vars.append(lab_scheduled)
            # Allow only one lab per day
            self.model.Add(sum(lab_day_vars) <= 1)

    def _add_consecutive_theory_constraint(self):
        """Ensure theory classes are consecutive when multiple in same day"""
        morning_slots = ["9:00-9:50", "9:50-10:40", "10:40-11:30", "11:30-12:20"]
        afternoon_slots = ["1:00-1:55", "1:55-2:50", "2:50-3:30"]
        
        for subject in [s for s in SUBJECTS if not s.is_lab]:
            for day in DAYS:
                # Morning slots
                morning_vars = [self.slots[(day, slot, subject.name)] for slot in morning_slots]
                # Ensure consecutive block
                for i in range(1, len(morning_vars)):
                    # If slot i is assigned and i-1 is not, then all subsequent must be 0
                    self.model.AddImplication(
                        morning_vars[i] & ~morning_vars[i-1],
                        sum(morning_vars[i+1:]) == 0
                    )
                # Afternoon slots
                afternoon_vars = [self.slots[(day, slot, subject.name)] for slot in afternoon_slots]
                for i in range(1, len(afternoon_vars)):
                    self.model.AddImplication(
                        afternoon_vars[i] & ~afternoon_vars[i-1],
                        sum(afternoon_vars[i+1:]) == 0
                    )
                    
    def _add_no_single_day_full_assignment(self):
        """Ensure no subject is assigned all weekly hours in a single day"""
        for subject in SUBJECTS:
            for day in DAYS:
                subject_slots = []
                for slot in TIME_SLOTS:
                    if slot != "LUNCH":
                        subject_slots.append(self.slots[(day, slot, subject.name)])
                self.model.Add(sum(subject_slots) <= subject.weekly_hours // 2 + 1)

    def _create_timetable_df(self, solver: cp_model.CpSolver) -> pd.DataFrame:
        """Create DataFrame from solved model"""
        df = pd.DataFrame(index=DAYS, columns=TIME_SLOTS)
        
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
                else:
                    df.at[day, slot] = "-"  # Fallback if no subject assigned (shouldn't happen)
        
        return df

def generate_section_timetable(section: str) -> pd.DataFrame:
    generator = TimeTableGenerator(section)
    return generator.generate()

def main():
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
            with ProcessPoolExecutor(max_workers=min(len(selected_sections), multiprocessing.cpu_count())) as executor:
                future_to_section = {
                    executor.submit(generate_section_timetable, section): section 
                    for section in selected_sections
                }
                
                for future in concurrent.futures.as_completed(future_to_section):
                    section = future_to_section[future]
                    try:
                        timetable = future.result()
                        if timetable is not None:
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
                    except Exception as e:
                        st.error(f"Error generating timetable for Section {section}: {str(e)}")

if __name__ == "__main__":
    main()