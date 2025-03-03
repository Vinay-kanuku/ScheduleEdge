import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
from typing import List, Dict, Tuple
from collections import defaultdict

# Configuration
SUBJECT_SHORTCUTS = {
    "Internet of Things and Applications Lab": "IOT Lab",
    "Web Programming Lab": "WP Lab",
    "Artificial Intelligence Lab Using Python": "AI Lab"
}

SUBJECTS = [
    # ... (keep your existing subject configuration but update lab hours to 3)
    {"name": "Software Engineering", "weekly_hours": 4, "faculty": ["PKA"], "is_lab": False},
    {"name": "Artificial Intelligence", "weekly_hours": 5, "faculty": ["Dr. GP"], "is_lab": False},
    {"name": "Web Programming", "weekly_hours": 4, "faculty": ["KS"], "is_lab": False},
    {"name": "Data Warehousing and Data Mining", "weekly_hours": 4, "faculty": ["Dr. NR"], "is_lab": False},
    {"name": "Introduction to Data Science", "weekly_hours": 3, "faculty": ["CHS"], "is_lab": False},
    {"name": "Internet of Things and Applications Lab", "weekly_hours": 3, "faculty": ["Dr. KRK", "LIB"], "is_lab": True},
    {"name": "Web Programming Lab", "weekly_hours": 3, "faculty": ["KS", "KB", "Dr. KP"], "is_lab": True},
    {"name": "Artificial Intelligence Lab Using Python", "weekly_hours": 3, "faculty": ["BHKD", "Dr. GP"], "is_lab": True},
    # ... rest of subjects
    {"name": "Constitution of India", "weekly_hours": 2, "faculty": ["PPS"], "is_lab": False}
]

DAYS = ["Monday", "Tuesday", "Thursday", "Friday", "Saturday"]
TIME_SLOTS = [
    "9:00-9:50", "9:50-10:40", "10:40-11:30", "11:30-12:20",
    "1:00-1:50", "1:50-2:40", "2:40-3:30"
]

def create_scheduler_model(section: str):
    model = cp_model.CpModel()
    
    # Create variables
    all_sessions = {}
    for subject in SUBJECTS:
        for day in DAYS:
            for slot in TIME_SLOTS:
                key = (subject['name'], day, slot)
                all_sessions[key] = model.NewBoolVar(str(key))

    # Hard constraints
    for subject in SUBJECTS:
        total_hours = sum(
            all_sessions[(subject['name'], day, slot)]
            for day in DAYS
            for slot in TIME_SLOTS
        )
        model.Add(total_hours == subject['weekly_hours'])

        # Lab specific constraints
        if subject['is_lab']:
            for day in DAYS:
                # Morning lab constraint (3 consecutive slots)
                morning_start = model.NewBoolVar(f"{subject['name']}_{day}_morning_start")
                model.Add(sum(all_sessions[(subject['name'], day, slot)] 
                            for slot in TIME_SLOTS[:3]) == 3).OnlyEnforceIf(morning_start)
                
                # Afternoon lab constraint (3 consecutive slots)
                afternoon_start = model.NewBoolVar(f"{subject['name']}_{day}_afternoon_start")
                model.Add(sum(all_sessions[(subject['name'], day, slot)] 
                            for slot in TIME_SLOTS[4:7]) == 3).OnlyEnforceIf(afternoon_start)
                
                # Only one lab session per day
                model.AddAtMostOne([morning_start, afternoon_start])

    # Faculty constraints
    faculty_assignments = defaultdict(list)
    for (subj, day, slot), var in all_sessions.items():
        faculty = next(s['faculty'] for s in SUBJECTS if s['name'] == subj)
        for f in faculty:
            faculty_assignments[(f, day, slot)].append(var)
    
    for key, vars in faculty_assignments.items():
        model.Add(sum(vars) <= 1)

    # Objective: Maximize slot utilization
    objective = sum(all_sessions.values())
    model.Maximize(objective)

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    return solver, all_sessions

def create_timetable_df(solver, all_sessions) -> pd.DataFrame:
    df = pd.DataFrame(columns=TIME_SLOTS, index=DAYS)
    df = df.fillna("")
    
    for (subj, day, slot), var in all_sessions.items():
        if solver.Value(var):
            df.at[day, slot] = SUBJECT_SHORTCUTS.get(subj, subj.split()[0])
    
    return df

def main():
    st.set_page_config(page_title="Optimized Timetable", layout="wide")
    st.title("Optimized Class Timetable Generator")
    
    if st.button("Generate Optimal Timetable"):
        with st.spinner("Optimizing schedule with OR-Tools..."):
            solver, variables = create_scheduler_model("A")
            if solver.StatusName() == "OPTIMAL":
                df = create_timetable_df(solver, variables)
                st.dataframe(df.style.applymap(lambda x: "background-color: #e6ffe6" if x else ""))
            else:
                st.error("No valid schedule found")

if __name__ == "__main__":
    main()
