import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from io import BytesIO
import zipfile

# Page configuration
st.set_page_config(
    page_title="Educational Timetable Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define constants based on our conversation
DAYS = {
    "II-Year": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    "III-Year": ["Monday", "Tuesday", "Thursday", "Friday", "Saturday"],  # No Wednesday
    "IV-Year": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
}
SLOTS = ["9:00-9:50", "9:50-10:40", "10:40-11:30", "11:30-12:20", "12:20-1:10", "1:10-2:00", "2:00-2:50", "2:50-3:30"]
LUNCH_SLOTS = {
    "II-Year": "11:30-12:20",  # After 4th slot
    "III-Year": "12:20-1:10",  # After 5th slot
    "IV-Year": "12:20-1:10"    # After 5th slot
}
SECTIONS = {
    "II-Year": ["A", "B", "C", "CSIT"],
    "III-Year": ["A", "B", "C"],
    "IV-Year": ["A", "B", "C"]
}

# Constraint weights
CONSTRAINT_WEIGHTS = {
    "faculty_clash": 10.0,
    "max_2_slots_per_day": 8.0,
    "lab_continuity": 10.0,
    "lunch_break": 8.0,
    "daily_load": 5.0,
    "section_balance": 4.0
}

def parse_subject_data(file_content):
    """
    Parse subject and faculty data from your PDF content
    """
    subject_data = []
    lines = file_content.strip().split('\n')
    current_year = None

    for line in lines:
        if "B.Tech II-Year I-Semester" in line:
            current_year = "II-Year"
        elif "B.Tech III-Year I-Semester" in line:
            current_year = "III-Year"
        elif "B.Tech IV-Year I-Semester" in line:
            current_year = "IV-Year"
        elif line.strip() and line[0].isdigit() and current_year:
            parts = line.split('\t')
            if len(parts) > 3:
                subject = {
                    "code": parts[1],
                    "name": parts[2],
                    "coordinator": parts[3],
                    "year": current_year,
                    "type": "Lab" if "Lab" in parts[2] else "Theory",
                    "faculty": {},
                    "hours_per_week": 5 if "Lab" not in parts[2] and "Value Ethics" not in parts[2] and "Constitution" not in parts[2] else 3 if "Lab" in parts[2] else 2
                }
                if current_year == "II-Year" and len(parts) >= 8:
                    subject["faculty"] = {
                        "A": parts[4].split('/') if parts[4] else [parts[3]] if parts[3] else [],
                        "B": parts[5].split('/') if parts[5] else [parts[3]] if parts[3] else [],
                        "C": parts[6].split('/') if parts[6] else [parts[3]] if parts[3] else [],
                        "CSIT": parts[7].split('/') if parts[7] else [parts[3]] if parts[3] else []
                    }
                elif len(parts) >= 7:
                    subject["faculty"] = {
                        "A": parts[4].split('/') if parts[4] else [parts[3]] if parts[3] else [],
                        "B": parts[5].split('/') if parts[5] else [parts[3]] if parts[3] else [],
                        "C": parts[6].split('/') if parts[6] else [parts[3]] if parts[3] else []
                    }
                subject_data.append(subject)

    return subject_data

def initialize_population(subjects, year, sections, pop_size=50):
    """
    Initialize a population of timetables with 2+2+1 slots for theory and 3 consecutive for labs
    """
    population = []
    for _ in range(pop_size):
        timetable = {}
        for section in sections:
            grid = {day: {slot: "-" for slot in SLOTS} for day in DAYS[year]}
            for day in DAYS[year]:
                grid[day][LUNCH_SLOTS[year]] = "LUNCH BREAK"
            
            section_subjects = [s for s in subjects if s["year"] == year and section in s["faculty"]]
            random.shuffle(section_subjects)
            
            for subject in section_subjects:
                if not subject["faculty"].get(section):  # Skip if no faculty
                    continue
                
                if subject["type"] == "Theory":
                    days_needed = 3 if subject["hours_per_week"] == 5 else 1
                    available_days = [d for d in DAYS[year] if sum(1 for s in SLOTS if grid[d][s] == "-") >= 2]
                    if len(available_days) < days_needed:
                        continue
                    
                    days_chosen = random.sample(available_days, days_needed)
                    slot_configs = [(2, days_chosen[0]), (2, days_chosen[1]), (1, days_chosen[2])] if days_needed == 3 else [(2, days_chosen[0])]
                    
                    for slots_needed, day in slot_configs:
                        available_slots = [i for i, s in enumerate(SLOTS) if grid[day][s] == "-" and s != LUNCH_SLOTS[year]]
                        valid_starts = [i for i in available_slots[:-slots_needed+1] if all(i+j in available_slots for j in range(slots_needed))]
                        if not valid_starts:
                            continue
                        
                        start_idx = random.choice(valid_starts)
                        faculty_str = subject["faculty"][section][0]
                        for i in range(slots_needed):
                            grid[day][SLOTS[start_idx + i]] = f"{subject['name']} ({faculty_str})"
                else:  # Lab
                    available_days = [d for d in DAYS[year] if sum(1 for i, s in enumerate(SLOTS) if grid[d][s] == "-" and s != LUNCH_SLOTS[year]) >= 3]
                    if not available_days:
                        continue
                    
                    day = random.choice(available_days)
                    available_slots = [i for i, s in enumerate(SLOTS) if grid[day][s] == "-" and s != LUNCH_SLOTS[year]]
                    valid_starts = [i for i in available_slots[:-2] if all(i+j in available_slots for j in range(3))]
                    if not valid_starts:
                        continue
                    
                    start_idx = random.choice(valid_starts)
                    faculty_str = ", ".join(subject["faculty"][section])
                    for i in range(3):
                        grid[day][SLOTS[start_idx + i]] = f"{subject['name']} ({faculty_str})"
            
            timetable[section] = grid
        population.append(timetable)
    return population

def evaluate_fitness(timetable, subjects, year, constraint_weights=CONSTRAINT_WEIGHTS):
    """
    Evaluate fitness with updated constraints
    """
    penalty = 0

    # Faculty clash across sections
    faculty_schedule = {}
    for section, section_timetable in timetable.items():
        for day in DAYS[year]:
            for slot in SLOTS:
                cell = section_timetable[day][slot]
                if cell != "-" and "LUNCH BREAK" not in cell:
                    faculty_start = cell.find("(")
                    faculty_end = cell.find(")")
                    faculty_list = [f.strip() for f in cell[faculty_start+1:faculty_end].split(",")]
                    for faculty in faculty_list:
                        key = (faculty, day, slot)
                        if key not in faculty_schedule:
                            faculty_schedule[key] = []
                        faculty_schedule[key].append(section)
    penalty += sum([len(sections) - 1 for sections in faculty_schedule.values() if len(sections) > 1]) * constraint_weights["faculty_clash"]

    # Max 2 slots/day for theory
    for section, section_timetable in timetable.items():
        for day in DAYS[year]:
            theory_slots = {}
            for slot in SLOTS:
                cell = section_timetable[day][slot]
                if "Lab" not in cell and cell != "-" and "LUNCH BREAK" not in cell:
                    subject_name = cell.split(" (")[0]
                    theory_slots[subject_name] = theory_slots.get(subject_name, 0) + 1
            penalty += sum([max(0, count - 2) for count in theory_slots.values()]) * constraint_weights["max_2_slots_per_day"]

    # Lab continuity (3 consecutive slots)
    for section, section_timetable in timetable.items():
        for day in DAYS[year]:
            for i in range(len(SLOTS) - 2):
                if "Lab" in section_timetable[day][SLOTS[i]]:
                    if not all("Lab" in section_timetable[day][SLOTS[i+j]] and section_timetable[day][SLOTS[i]] == section_timetable[day][SLOTS[i+j]] for j in range(1, 3)):
                        penalty += constraint_weights["lab_continuity"]

    # Lunch break
    for section, section_timetable in timetable.items():
        for day in DAYS[year]:
            if section_timetable[day][LUNCH_SLOTS[year]] != "LUNCH BREAK":
                penalty += constraint_weights["lunch_break"]

    # Daily load balance
    for section, section_timetable in timetable.items():
        daily_loads = [sum(1 for slot in SLOTS if section_timetable[day][slot] not in ["-", "LUNCH BREAK"]) for day in DAYS[year]]
        penalty += np.std(daily_loads) * constraint_weights["daily_load"]

    # Section balance
    section_patterns = {section: [[section_timetable[day][slot] != "-" for slot in SLOTS] for day in DAYS[year]] for section in timetable}
    section_balance_penalty = 0
    for i, s1 in enumerate(timetable.keys()):
        for s2 in list(timetable.keys())[i+1:]:
            for d in range(len(DAYS[year])):
                section_balance_penalty += sum(a != b for a, b in zip(section_patterns[s1][d], section_patterns[s2][d]))
    penalty += section_balance_penalty * constraint_weights["section_balance"]

    # Penalty for unassigned subjects
    section_subjects = {section: [s for s in subjects if s["year"] == year and section in s["faculty"]] for section in timetable}
    for section, section_timetable in timetable.items():
        assigned_slots = {}
        for day in DAYS[year]:
            for slot in SLOTS:
                cell = section_timetable[day][slot]
                if cell != "-" and "LUNCH BREAK" not in cell:
                    subject_name = cell.split(" (")[0]
                    assigned_slots[subject_name] = assigned_slots.get(subject_name, 0) + 1
        
        for subject in section_subjects[section]:
            expected_slots = subject["hours_per_week"]
            actual_slots = assigned_slots.get(subject["name"], 0)
            penalty += (expected_slots - actual_slots) * 10  # Heavy penalty for missing slots

    return max(0, 1000 - penalty)

def crossover(parent1, parent2):
    child = {}
    for section in set(parent1.keys()).union(parent2.keys()):
        child[section] = parent1[section].copy() if random.random() < 0.5 and section in parent1 else parent2[section].copy()
    return child

def mutate(timetable, year, mutation_rate=0.2):
    mutated = {section: {day: timetable[section][day].copy() for day in DAYS[year]} for section in timetable}
    for section in mutated:
        if random.random() < mutation_rate:
            day1, day2 = random.sample(DAYS[year], 2)
            slot1, slot2 = random.sample([s for s in SLOTS if s != LUNCH_SLOTS[year]], 2)
            mutated[section][day1][slot1], mutated[section][day2][slot2] = mutated[section][day2][slot2], mutated[section][day1][slot1]
    return mutated

def run_genetic_algorithm(subjects, year, sections, generations=50, population_size=50, constraint_weights=CONSTRAINT_WEIGHTS):
    population = initialize_population(subjects, year, sections, population_size)
    best_solution = None
    best_fitness = 0
    fitness_history = []

    for generation in range(generations):
        fitness_scores = [evaluate_fitness(timetable, subjects, year, constraint_weights) for timetable in population]
        max_fitness_idx = fitness_scores.index(max(fitness_scores))
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_solution = population[max_fitness_idx]
        
        fitness_history.append({'generation': generation, 'best_fitness': best_fitness, 'avg_fitness': sum(fitness_scores) / len(fitness_scores)})
        
        new_population = [best_solution]  # Elitism
        while len(new_population) < population_size:
            tournament = random.sample(list(zip(population, fitness_scores)), 3)
            parent1 = max(tournament, key=lambda x: x[1])[0]
            tournament = random.sample(list(zip(population, fitness_scores)), 3)
            parent2 = max(tournament, key=lambda x: x[1])[0]
            child = crossover(parent1, parent2)
            child = mutate(child, year)
            new_population.append(child)
        population = new_population
    
    return best_solution, fitness_history

def convert_to_dataframe(timetable_dict, year):
    return pd.DataFrame(timetable_dict, index=DAYS[year], columns=SLOTS)

def main():
    st.title("📚 Educational Timetable Generator")

    st.sidebar.header("Settings")
    uploaded_file = st.sidebar.file_uploader("Upload Faculty/Subject Data (TXT)", type=["txt"])
    
    if uploaded_file is not None:
        file_content = uploaded_file.getvalue().decode("utf-8")
        st.session_state.subjects_data = parse_subject_data(file_content)
        st.sidebar.success(f"Loaded {len(st.session_state.subjects_data)} subjects")
    elif not st.session_state.get('subjects_data'):
        if st.sidebar.button("Use Sample Data"):
            st.session_state.subjects_data = parse_subject_data(open("sample_data.txt", "r").read())
            st.sidebar.success(f"Loaded {len(st.session_state.subjects_data)} sample subjects")

    st.sidebar.subheader("Select Years and Sections")
    selected_years = st.sidebar.multiselect("Year", ["II-Year", "III-Year", "IV-Year"], default=["II-Year"])
    selected_sections = {}
    for year in selected_years:
        selected_sections[year] = st.sidebar.multiselect(f"Sections ({year})", SECTIONS[year], default=SECTIONS[year][:2])

    st.sidebar.subheader("Algorithm Parameters")
    population_size = st.sidebar.slider("Population Size", 20, 200, 50, 10)
    generations = st.sidebar.slider("Generations", 10, 200, 30, 10)
    generate_button = st.sidebar.button("Generate Timetables", type="primary")

    # Warn about skipped subjects
    if st.session_state.get('subjects_data'):
        skipped_subjects = [s["name"] for s in st.session_state.subjects_data if not any(s["faculty"].values())]
        if skipped_subjects:
            st.sidebar.warning(f"Skipped subjects with no faculty: {', '.join(skipped_subjects)}")

    tabs = st.tabs(["Timetables", "Subject Data", "Analysis"])

    with tabs[1]:
        st.header("Subject and Faculty Data")
        if st.session_state.get('subjects_data'):
            subject_df = pd.DataFrame([
                {
                    "Code": s["code"],
                    "Name": s["name"],
                    "Type": s["type"],
                    "Year": s["year"],
                    "Hours/Week": s["hours_per_week"],
                    **{f"Faculty (Sec {sec})": ", ".join(s["faculty"].get(sec, [])) for sec in SECTIONS[s["year"]]}
                }
                for s in st.session_state.subjects_data if s["year"] in selected_years
            ])
            st.dataframe(subject_df, use_container_width=True)
        else:
            st.info("Please upload subject data or use sample data")

    with tabs[2]:
        st.header("Timetable Analysis")
        if st.session_state.get('fitness_history'):
            fitness_df = pd.DataFrame(st.session_state.fitness_history)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(fitness_df['generation'], fitness_df['best_fitness'], label='Best Fitness')
            ax.plot(fitness_df['generation'], fitness_df['avg_fitness'], label='Average Fitness')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness Score')
            ax.set_title('Fitness Evolution Over Generations')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            if st.session_state.get('section_timetables'):
                st.subheader("Faculty Workload Analysis")
                faculty_hours = {}
                for year, timetables in st.session_state.section_timetables.items():
                    for section, df in timetables.items():
                        for day in DAYS[year]:
                            for slot in SLOTS:
                                cell = df.at[day, slot]
                                if "LUNCH BREAK" not in cell and cell != "-":
                                    faculty_list = cell.split(" (")[1][:-1].split(", ")
                                    for faculty in faculty_list:
                                        faculty_hours[faculty] = faculty_hours.get(faculty, 0) + 1
                faculty_df = pd.DataFrame(list(faculty_hours.items()), columns=['Faculty', 'Hours']).sort_values('Hours', ascending=False)
                st.dataframe(faculty_df)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(faculty_df['Faculty'], faculty_df['Hours'])
                ax.set_xlabel('Faculty')
                ax.set_ylabel('Hours')
                plt.xticks(rotation=90)
                st.pyplot(fig)
        else:
            st.info("Generate timetables to view analysis")

    with tabs[0]:
        st.header("Generated Timetables")
        with st.expander("Advanced Options"):
            st.subheader("Constraint Weights")
            for key, value in CONSTRAINT_WEIGHTS.items():
                st.session_state.custom_constraint_weights[key] = st.slider(key.replace("_", " ").title(), 1.0, 20.0, value, 0.5)

        if generate_button and st.session_state.get('subjects_data') and any(selected_sections.values()):
            with st.spinner("Generating optimized timetables..."):
                st.session_state.section_timetables = {}
                st.session_state.fitness_history = []  # Reset fitness history
                for year in selected_years:
                    best_timetable, fitness_history = run_genetic_algorithm(
                        st.session_state.subjects_data, year, selected_sections[year],
                        generations=generations, population_size=population_size,
                        constraint_weights=st.session_state.custom_constraint_weights
                    )
                    st.session_state.section_timetables[year] = {section: convert_to_dataframe(best_timetable[section], year) for section in best_timetable}
                    st.session_state.fitness_history.extend(fitness_history)
            st.success("Timetables generated successfully!")

        if st.session_state.get('section_timetables'):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Download Excel Timetables"):
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                        for year, timetables in st.session_state.section_timetables.items():
                            for section, df in timetables.items():
                                df.to_excel(writer, sheet_name=f"{year} Sec {section}")
                                worksheet = writer.sheets[f"{year} Sec {section}"]
                                for row in range(1, len(df.index) + 1):
                                    for col in range(1, len(df.columns) + 1):
                                        cell_value = df.iloc[row-1, col-1]
                                        worksheet.write(row, col, cell_value, writer.book.add_format({
                                            'bg_color': '#FFCCCC' if "LUNCH BREAK" in cell_value else '#E6FFE6' if "Lab" in cell_value else '#E6F2FF' if cell_value != "-" else '#FFFFFF',
                                            'border': 1
                                        }))
                    st.download_button("Download Excel Timetables", excel_buffer.getvalue(), "timetables.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            with col2:
                if st.button("Download ZIP of HTML Timetables"):
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        for year, timetables in st.session_state.section_timetables.items():
                            for section, df in timetables.items():
                                html_content = f"""
                                <!DOCTYPE html><html><head><title>Timetable - {year} Section {section}</title>
                                <style>table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid #ddd; padding: 8px; }} 
                                .theory {{ background-color: #E6F2FF; }} .lab {{ background-color: #E6FFE6; }} .lunch {{ background-color: #FFCCCC; }}</style>
                                </head><body><h1>Timetable - {year} Section {section}</h1><table><tr><th>Day / Time</th>{''.join(f'<th>{slot}</th>' for slot in SLOTS)}</tr>"""
                                for day, row in df.iterrows():
                                    html_content += f"<tr><th>{day}</th>"
                                    for cell in row:
                                        cell_class = "lunch" if "LUNCH BREAK" in cell else "lab" if "Lab" in cell else "theory" if cell != "-" else ""
                                        html_content += f'<td class="{cell_class}">{cell}</td>'
                                    html_content += "</tr>"
                                html_content += "</table></body></html>"
                                zip_file.writestr(f"timetable_{year}_section_{section}.html", html_content)
                    st.download_button("Download ZIP of HTML Timetables", zip_buffer.getvalue(), "timetables_html.zip", "application/zip")

            for year in selected_years:
                if year in st.session_state.section_timetables:
                    st.subheader(f"{year} Timetables")
                    section_tabs = st.tabs(selected_sections[year])
                    for i, section in enumerate(selected_sections[year]):
                        with section_tabs[i]:
                            st.dataframe(st.session_state.section_timetables[year][section], use_container_width=True, height=400)
        else:
            st.info("Please upload data, select sections, and click 'Generate Timetables'")

if __name__ == "__main__":
    if 'subjects_data' not in st.session_state:
        st.session_state.subjects_data = []
    if 'section_timetables' not in st.session_state:
        st.session_state.section_timetables = {}
    if 'custom_constraint_weights' not in st.session_state:
        st.session_state.custom_constraint_weights = CONSTRAINT_WEIGHTS.copy()
    if 'fitness_history' not in st.session_state:
        st.session_state.fitness_history = []
    main()