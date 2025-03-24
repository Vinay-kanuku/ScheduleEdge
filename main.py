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

# Define constants
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SLOTS = ["9:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-1:00", "1:00-2:00", "2:00-3:00", "3:00-4:00", "4:00-5:00"]
LUNCH_SLOT = "1:00-2:00"  # Lunch break
SECTIONS = ["A", "B", "C", "CSIT"]  # All available sections
BATCH_TYPES = ["Theory", "Lab"]

# Default constraint weights
CONSTRAINT_WEIGHTS = {
    "faculty_clash": 10.0,
    "continuous_periods": 5.0,
    "lunch_break": 8.0,
    "lab_continuity": 10.0,
    "daily_load": 5.0,
    "section_balance": 4.0
}

def parse_subject_data(pdf_content):
    """
    Parse subject data from the PDF content
    """
    subject_data = []
    
    # Process the text data to extract subject information
    lines = pdf_content.strip().split('\n')
    
    current_year = None
    current_semester = None
    current_section = None
    
    for line in lines:
        if "B.Tech II-Year I-Semester" in line:
            current_year = 2
            current_semester = 1
        elif "B.Tech III-Year I-Semester" in line:
            current_year = 3
            current_semester = 1
        elif "B.Tech IV-Year I-Semester" in line:
            current_year = 4
            current_semester = 1
            
        # Look for subject lines
        if line.strip() and line[0].isdigit() and ":" not in line:
            parts = line.split()
            if len(parts) > 3:  # Basic validation
                subject_data.append({
                    "code": parts[1] if len(parts) > 1 else "",
                    "name": " ".join(parts[2:-1]) if len(parts) > 3 else "",
                    "coordinator": parts[-1] if len(parts) > 3 else "",
                    "year": current_year,
                    "semester": current_semester,
                    "type": "Theory" if "Lab" not in parts[2:] else "Lab",
                    "faculty": {}
                })
    
    # Now process the faculty assignments (simplified version)
    # For a real implementation, you'd need to parse the actual faculty assignments
    # from the complex formatted data in your PDF
    
    # Extract faculty information (simplified for demonstration)
    faculty_table = []
    reading_faculty = False
    for line in lines:
        if line.strip() and "Section-A" in line and "Section-B" in line:
            reading_faculty = True
            continue
        
        if reading_faculty and line.strip() and line[0].isdigit():
            parts = line.split()
            if len(parts) >= 5:
                faculty_entry = {
                    "sno": parts[0],
                    "code": parts[1],
                    "secA": "",
                    "secB": "",
                    "secC": "",
                    "secCSIT": ""
                }
                
                # Find faculty names in the line
                faculty_parts = line.split()
                for i, part in enumerate(faculty_parts):
                    if part.endswith("Section-A"):
                        faculty_entry["secA"] = faculty_parts[i+1] if i+1 < len(faculty_parts) else ""
                    elif part.endswith("Section-B"):
                        faculty_entry["secB"] = faculty_parts[i+1] if i+1 < len(faculty_parts) else ""
                    elif part.endswith("Section-C"):
                        faculty_entry["secC"] = faculty_parts[i+1] if i+1 < len(faculty_parts) else ""
                    elif part.endswith("Section-CSIT"):
                        faculty_entry["secCSIT"] = faculty_parts[i+1] if i+1 < len(faculty_parts) else ""
                
                faculty_table.append(faculty_entry)
    
    # For demonstration, create sample faculty data
    # In a real implementation, you would extract this from your PDF
    for subject in subject_data:
        if subject["type"] == "Theory":
            # Assign faculty based on year
            if subject["year"] == 2:
                subject["faculty"] = {
                    "A": ["Dr. " + subject["coordinator"]],
                    "B": ["Dr. " + subject["coordinator"]],
                    "C": ["Prof. " + subject["coordinator"]],
                    "CSIT": ["Prof. " + subject["coordinator"]]
                }
                subject["hours_per_week"] = 3
            else:
                subject["faculty"] = {
                    "A": ["Dr. " + subject["coordinator"]],
                    "B": ["Dr. " + subject["coordinator"]],
                    "C": ["Dr. " + subject["coordinator"]]
                }
                subject["hours_per_week"] = 3
        else:  # Lab
            if subject["year"] == 2:
                subject["faculty"] = {
                    "A": ["Dr. " + subject["coordinator"], "Asst. Prof. Lab"],
                    "B": ["Dr. " + subject["coordinator"], "Asst. Prof. Lab"],
                    "C": ["Dr. " + subject["coordinator"], "Asst. Prof. Lab"],
                    "CSIT": ["Dr. " + subject["coordinator"], "Asst. Prof. Lab"]
                }
                subject["hours_per_week"] = 3  # Labs typically 3 hours
                subject["batch_size"] = 2  # A1, A2, etc.
            else:
                subject["faculty"] = {
                    "A": ["Dr. " + subject["coordinator"], "Asst. Prof. Lab"],
                    "B": ["Dr. " + subject["coordinator"], "Asst. Prof. Lab"],
                    "C": ["Dr. " + subject["coordinator"], "Asst. Prof. Lab"]
                }
                subject["hours_per_week"] = 3
                subject["batch_size"] = 2
    
    return subject_data

# Replace this with your actual parsing function for your PDF data
def extract_subjects_from_pdf(pdf_content):
    """
    Extract subject information from the PDF content
    This is a simplified version - you'll need to improve parsing based on actual format
    """
    # For demonstration, parse from the provided text
    return parse_subject_data(pdf_content)

# Initialize session state
if 'subjects_data' not in st.session_state:
    st.session_state.subjects_data = []

if 'section_timetables' not in st.session_state:
    st.session_state.section_timetables = {}

if 'custom_constraint_weights' not in st.session_state:
    st.session_state.custom_constraint_weights = CONSTRAINT_WEIGHTS.copy()

if 'fitness_history' not in st.session_state:
    st.session_state.fitness_history = []

# Genetic Algorithm functions
def initialize_population(subjects, sections, pop_size=50):
    """
    Initialize a population of timetables
    """
    population = []
    
    for _ in range(pop_size):
        # Create timetable for each section
        timetable = {}
        for section in sections:
            # Initialize empty timetable grid
            grid = {day: {slot: "-" for slot in SLOTS} for day in DAYS}
            
            # Set lunch break
            for day in DAYS:
                grid[day][LUNCH_SLOT] = "LUNCH BREAK"
            
            # Assign subjects
            section_subjects = [s for s in subjects if section in s["faculty"]]
            
            for subject in section_subjects:
                hours_remaining = subject["hours_per_week"]
                
                if subject["type"] == "Theory":
                    # Distribute theory classes
                    while hours_remaining > 0:
                        day = random.choice(DAYS)
                        slot = random.choice([s for s in SLOTS if s != LUNCH_SLOT])
                        
                        if grid[day][slot] == "-":
                            grid[day][slot] = f"{subject['name']} ({subject['faculty'][section][0]})"
                            hours_remaining -= 1
                else:  # Lab sessions
                    # Find continuous slots for lab
                    lab_length = min(hours_remaining, 3)  # Labs are typically 3 hours
                    
                    day = random.choice(DAYS)
                    # Find starting slots that have enough consecutive slots
                    possible_start_slots = []
                    for i in range(len(SLOTS) - lab_length + 1):
                        if all(grid[day][SLOTS[i+j]] == "-" for j in range(lab_length) if SLOTS[i+j] != LUNCH_SLOT):
                            possible_start_slots.append(i)
                    
                    if possible_start_slots:
                        start_slot_idx = random.choice(possible_start_slots)
                        # For batches (e.g., A1, A2)
                        batch_info = ""
                        if "batch_size" in subject and subject["batch_size"] > 1:
                            batch_info = f" Batch 1/{subject['batch_size']}"
                        
                        for j in range(lab_length):
                            slot_idx = start_slot_idx + j
                            if slot_idx < len(SLOTS) and SLOTS[slot_idx] != LUNCH_SLOT:
                                faculty_str = ", ".join(subject["faculty"][section])
                                grid[day][SLOTS[slot_idx]] = f"{subject['name']}{batch_info} ({faculty_str})"
                        
                        hours_remaining -= lab_length
            
            # Convert grid dict to a more usable format
            section_timetable = {}
            for day in DAYS:
                section_timetable[day] = {slot: grid[day][slot] for slot in SLOTS}
            
            timetable[section] = section_timetable
        
        population.append(timetable)
    
    return population

def evaluate_fitness(timetable, subjects, constraint_weights=CONSTRAINT_WEIGHTS):
    """
    Evaluate the fitness of a timetable based on various constraints
    Higher fitness is better
    """
    total_fitness = 0
    penalty = 0
    
    # Check for faculty clashes across sections
    faculty_schedule = {}
    for section, section_timetable in timetable.items():
        for day in DAYS:
            for slot in SLOTS:
                if slot in section_timetable[day]:
                    cell = section_timetable[day][slot]
                    if "LUNCH BREAK" not in cell and cell != "-":
                        # Extract faculty from the cell
                        faculty_start = cell.find("(")
                        faculty_end = cell.find(")")
                        if faculty_start != -1 and faculty_end != -1:
                            faculty_str = cell[faculty_start+1:faculty_end]
                            faculty_list = [f.strip() for f in faculty_str.split(",")]
                            
                            for faculty in faculty_list:
                                key = (faculty, day, slot)
                                if key not in faculty_schedule:
                                    faculty_schedule[key] = []
                                faculty_schedule[key].append(section)
    
    # Calculate faculty clash penalty
    faculty_clash_count = 0
    for key, sections in faculty_schedule.items():
        if len(sections) > 1:
            faculty_clash_count += 1
    
    # Penalize faculty clashes
    penalty += faculty_clash_count * constraint_weights["faculty_clash"]
    
    # Check continuous teaching periods constraint
    for section, section_timetable in timetable.items():
        for day in DAYS:
            continuous_count = 0
            for slot in SLOTS:
                if slot in section_timetable[day]:
                    cell = section_timetable[day][slot]
                    if "LUNCH BREAK" not in cell and cell != "-":
                        continuous_count += 1
                    else:
                        continuous_count = 0
                        
                    # Penalize more than 3 continuous periods
                    if continuous_count > 3:
                        penalty += constraint_weights["continuous_periods"]
    
    # Check lunch break constraint
    lunch_violations = 0
    for section, section_timetable in timetable.items():
        for day in DAYS:
            if LUNCH_SLOT in section_timetable[day]:
                if section_timetable[day][LUNCH_SLOT] != "LUNCH BREAK":
                    lunch_violations += 1
    
    penalty += lunch_violations * constraint_weights["lunch_break"]
    
    # Check lab continuity constraint
    lab_continuity_violations = 0
    for section, section_timetable in timetable.items():
        for day in DAYS:
            for i in range(len(SLOTS) - 1):
                if i < len(SLOTS) - 1 and SLOTS[i] in section_timetable[day] and SLOTS[i+1] in section_timetable[day]:
                    current_cell = section_timetable[day][SLOTS[i]]
                    next_cell = section_timetable[day][SLOTS[i+1]]
                    
                    # If current is a lab but next isn't the same lab
                    if "Lab" in current_cell and current_cell != next_cell and "LUNCH BREAK" not in next_cell and next_cell != "-":
                        lab_continuity_violations += 1
    
    penalty += lab_continuity_violations * constraint_weights["lab_continuity"]
    
    # Check daily teaching load balance
    load_balance_penalty = 0
    for section, section_timetable in timetable.items():
        daily_loads = []
        for day in DAYS:
            day_load = sum(1 for slot in SLOTS if slot in section_timetable[day] and 
                           section_timetable[day][slot] != "-" and 
                           "LUNCH BREAK" not in section_timetable[day][slot])
            daily_loads.append(day_load)
        
        # Calculate standard deviation of daily loads
        if daily_loads:
            load_std = np.std(daily_loads)
            load_balance_penalty += load_std
    
    penalty += load_balance_penalty * constraint_weights["daily_load"]
    
    # Section balance constraint - ensure similar patterns across sections
    section_balance_penalty = 0
    section_patterns = {}
    
    for section, section_timetable in timetable.items():
        section_patterns[section] = []
        for day in DAYS:
            day_pattern = []
            for slot in SLOTS:
                if slot in section_timetable[day]:
                    cell_type = "Empty" if section_timetable[day][slot] == "-" else (
                        "Lunch" if section_timetable[day][slot] == "LUNCH BREAK" else (
                            "Theory" if "Lab" not in section_timetable[day][slot] else "Lab"
                        )
                    )
                    day_pattern.append(cell_type)
            section_patterns[section].append(day_pattern)
    
    # Compare patterns between sections
    sections_list = list(section_patterns.keys())
    for i in range(len(sections_list)):
        for j in range(i+1, len(sections_list)):
            sect1 = sections_list[i]
            sect2 = sections_list[j]
            
            pattern_diff = 0
            for day_idx in range(len(DAYS)):
                pattern1 = section_patterns[sect1][day_idx]
                pattern2 = section_patterns[sect2][day_idx]
                
                for slot_idx in range(min(len(pattern1), len(pattern2))):
                    if pattern1[slot_idx] != pattern2[slot_idx]:
                        pattern_diff += 1
            
            section_balance_penalty += pattern_diff
    
    penalty += section_balance_penalty * constraint_weights["section_balance"]
    
    # Calculate final fitness (higher is better)
    total_fitness = 1000 - penalty
    
    return max(0, total_fitness)  # Ensure non-negative fitness

def crossover(parent1, parent2):
    """
    Perform crossover between two parent timetables
    Uses section-level crossover
    """
    child = {}
    
    # Get all sections
    all_sections = set(list(parent1.keys()) + list(parent2.keys()))
    
    for section in all_sections:
        # 50% chance to inherit from each parent
        if section in parent1 and section in parent2:
            if random.random() < 0.5:
                child[section] = parent1[section].copy()
            else:
                child[section] = parent2[section].copy()
        elif section in parent1:
            child[section] = parent1[section].copy()
        else:
            child[section] = parent2[section].copy()
    
    return child

def mutate(timetable, mutation_rate=0.2):
    """
    Mutate a timetable with a given probability
    """
    mutated = {section: {day: section_timetable[day].copy() for day in section_timetable} 
               for section, section_timetable in timetable.items()}
    
    for section, section_timetable in mutated.items():
        # Swap two random slots with a certain probability
        if random.random() < mutation_rate:
            # Select two random days and slots
            day1, day2 = random.sample(DAYS, 2)
            valid_slots = [s for s in SLOTS if s != LUNCH_SLOT]
            if valid_slots:
                slot1, slot2 = random.sample(valid_slots, 2)
                
                # Swap the classes
                temp = section_timetable[day1][slot1]
                section_timetable[day1][slot1] = section_timetable[day2][slot2]
                section_timetable[day2][slot2] = temp
    
    return mutated

def run_genetic_algorithm(subjects, sections, generations=50, population_size=50, constraint_weights=CONSTRAINT_WEIGHTS):
    """
    Run the genetic algorithm to generate optimized timetables
    """
    # Initialize population
    population = initialize_population(subjects, sections, population_size)
    
    # Track best solution and fitness history
    best_solution = None
    best_fitness = 0
    fitness_history = []
    
    for generation in range(generations):
        # Evaluate fitness for each timetable
        fitness_scores = [evaluate_fitness(timetable, subjects, constraint_weights) for timetable in population]
        
        # Track best solution
        max_fitness_idx = fitness_scores.index(max(fitness_scores))
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_solution = population[max_fitness_idx]
        
        # Record fitness history
        fitness_history.append({
            'generation': generation,
            'best_fitness': best_fitness,
            'avg_fitness': sum(fitness_scores) / len(fitness_scores)
        })
        
        # Create new population
        new_population = []
        
        # Elitism - keep best solution
        new_population.append(population[max_fitness_idx])
        
        # Generate rest of population
        while len(new_population) < population_size:
            # Tournament selection
            tournament_size = 3
            tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
            parent1 = max(tournament, key=lambda x: x[1])[0]
            
            tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
            parent2 = max(tournament, key=lambda x: x[1])[0]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child)
            
            new_population.append(child)
        
        population = new_population
    
    return best_solution, fitness_history

def convert_to_dataframe(timetable_dict):
    """
    Convert the timetable dictionary to pandas DataFrame format
    """
    # Create DataFrame from the dictionary
    df = pd.DataFrame(index=DAYS, columns=SLOTS)
    
    # Fill in the data
    for day in DAYS:
        for slot in SLOTS:
            if slot in timetable_dict[day]:
                df.at[day, slot] = timetable_dict[day][slot]
            else:
                df.at[day, slot] = "-"
    
    return df

def main():
    st.title("📚 Educational Timetable Generator")
    
    st.sidebar.header("Settings")
    
    # Upload subject data
    uploaded_file = st.sidebar.file_uploader("Upload Faculty/Subject Data (PDF or CSV)", type=["pdf", "csv", "txt"])
    
    if uploaded_file is not None:
        file_content = uploaded_file.getvalue().decode("utf-8")
        
        # Extract subjects from the file
        subjects_data = extract_subjects_from_pdf(file_content)
        st.session_state.subjects_data = subjects_data
        
        # Show success message
        st.sidebar.success(f"Loaded {len(subjects_data)} subjects")
    elif len(st.session_state.subjects_data) == 0:
        # If no data is uploaded yet, show sample data option
        if st.sidebar.button("Use Sample Data"):
            # Sample data based on your PDF content
            sample_data = [
                {
                    "code": "GR22A2067", 
                    "name": "Digital Logic Design",
                    "coordinator": "TNP",
                    "year": 2,
                    "semester": 1,
                    "type": "Theory",
                    "hours_per_week": 3,
                    "faculty": {
                        "A": ["T.N.P. Madhuri"],
                        "B": ["T.N.P. Madhuri"],
                        "C": ["P. Gopala Krishna"],
                        "CSIT": ["P.Bharathi"]
                    }
                },
                {
                    "code": "GR22A2068",
                    "name": "Java Programming",
                    "coordinator": "CHV",
                    "year": 2,
                    "semester": 1,
                    "type": "Theory",
                    "hours_per_week": 3,
                    "faculty": {
                        "A": ["Dr. Y. J. Nagendra Kumar"],
                        "B": ["Dr Ch.Vidyadhari"],
                        "C": ["Dr Ch.Vidyadhari"],
                        "CSIT": ["A.Srilakshmi"]
                    }
                },
                {
                    "code": "GR22A2071",
                    "name": "Java Programming Lab",
                    "coordinator": "VP",
                    "year": 2,
                    "semester": 1,
                    "type": "Lab",
                    "hours_per_week": 3,
                    "batch_size": 2,
                    "faculty": {
                        "A": ["A.Vani Pushpavathi", "M. UshaRani"],
                        "B": ["A.Vani Pushpavathi", "T. Nishitha"],
                        "C": ["Dr. Ch.Vidyadhari", "A Srilakshmi"],
                        "CSIT": ["A Srilakshmi", "A.Vani Pushpavathi"]
                    }
                },
                {
                    "code": "GR22A3052",
                    "name": "Software Engineering",
                    "coordinator": "Dr.RVSS",
                    "year": 3,
                    "semester": 1,
                    "type": "Theory",
                    "hours_per_week": 3,
                    "faculty": {
                        "A": ["Dr. R V S S S Nagini"],
                        "B": ["Dr. R V S S S Nagini"],
                        "C": ["P. K. Abhilash"]
                    }
                },
                {
                    "code": "GR22A3058",
                    "name": "Web Programming Lab",
                    "coordinator": "KS",
                    "year": 3,
                    "semester": 1,
                    "type": "Lab",
                    "hours_per_week": 3,
                    "batch_size": 2,
                    "faculty": {
                        "A": ["K. Sandeep", "R. Madhuri"],
                        "B": ["K. Sandeep", "R. Madhuri"],
                        "C": ["P. Bharathi", "R. Madhuri", "Dr. K. Prasanna Lakshmi"]
                    }
                },
                {
                    "code": "GR20A4056",
                    "name": "Unified Modelling Language",
                    "coordinator": "PKA",
                    "year": 4,
                    "semester": 1,
                    "type": "Theory",
                    "hours_per_week": 3,
                    "faculty": {
                        "A": ["P. K. Abhilash"],
                        "B": ["P. K. Abhilash"],
                        "C": ["P. K. Abhilash"]
                    }
                },
                {
                    "code": "GR20A4064",
                    "name": "Unified Modelling Language Lab",
                    "coordinator": "UR",
                    "year": 4,
                    "semester": 1,
                    "type": "Lab",
                    "hours_per_week": 3,
                    "batch_size": 2,
                    "faculty": {
                        "A": ["M. Usharani", "J. Alekya"],
                        "B": ["M. Usharani", "J. Alekya"],
                        "C": ["M. Usharani", "J. Alekya"]
                    }
                }
            ]
            
            st.session_state.subjects_data = sample_data
            st.sidebar.success(f"Loaded {len(sample_data)} sample subjects")
    
    # Section selection
    st.sidebar.subheader("Select Sections")
    selected_years = st.sidebar.multiselect(
        "Year", 
        [2, 3, 4],
        default=[2]
    )
    
    available_sections = []
    if 2 in selected_years:
        available_sections.extend(["A", "B", "C", "CSIT"])
    if 3 in selected_years:
        available_sections.extend(["A", "B", "C"])
    if 4 in selected_years:
        available_sections.extend(["A", "B", "C"])
    
    # Remove duplicates
    available_sections = list(set(available_sections))
    
    selected_sections = st.sidebar.multiselect(
        "Sections",
        available_sections,
        default=available_sections[:2] if available_sections else []
    )
    
    # Genetic Algorithm parameters
    st.sidebar.subheader("Algorithm Parameters")
    
    population_size = st.sidebar.slider(
        "Population Size", 
        min_value=20, 
        max_value=200, 
        value=50,
        step=10,
        help="Number of timetables in each generation"
    )
    
    generations = st.sidebar.slider(
        "Generations", 
        min_value=10, 
        max_value=200, 
        value=30,
        step=10,
        help="Number of evolution cycles"
    )
    
    # Generate timetables button
    generate_button = st.sidebar.button("Generate Timetables", type="primary")
    
    # Main content area
    tabs = st.tabs(["Timetables", "Subject Data", "Analysis"])
    
    # Subject Data tab
    with tabs[1]:
        st.header("Subject and Faculty Data")
        
        if st.session_state.subjects_data:
            # Filter subjects for selected years
            filtered_subjects = [s for s in st.session_state.subjects_data if s["year"] in selected_years]
            
            # Create a DataFrame for display
            subject_df = pd.DataFrame([
                {
                    "Code": s["code"],
                    "Name": s["name"],
                    "Type": s["type"],
                    "Year": s["year"],
                    "Hours/Week": s["hours_per_week"],
                    "Faculty (Sec A)": ", ".join(s["faculty"].get("A", [])),
                    "Faculty (Sec B)": ", ".join(s["faculty"].get("B", [])),
                    "Faculty (Sec C)": ", ".join(s["faculty"].get("C", [])),
                    "Faculty (Sec CSIT)": ", ".join(s["faculty"].get("CSIT", []))
                }
                for s in filtered_subjects
            ])
            
            st.dataframe(subject_df, use_container_width=True)
        else:
            st.info("Please upload subject data or use sample data")
    
    # Analysis tab
    with tabs[2]:
        st.header("Timetable Analysis")
        
        if 'fitness_history' in st.session_state and st.session_state.fitness_history:
            # Create evolution chart
            st.subheader("Fitness Evolution")
            
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
            
            # Faculty workload analysis
            if 'section_timetables' in st.session_state:
                st.subheader("Faculty Workload Analysis")
                
                faculty_hours = {}
                
                for section, timetable in st.session_state.section_timetables.items():
                    for day in DAYS:
                        for slot in SLOTS:
                            if day in timetable.index and slot in timetable.columns:
                                cell = timetable.at[day, slot]
                                if "LUNCH BREAK" not in cell and cell != "-":
                                    # Extract faculty from the cell
                                    faculty_start = cell.find("(")
                                    faculty_end = cell.find(")")
                                    if faculty_start != -1 and faculty_end != -1:
                                        faculty_str = cell[faculty_start+1:faculty_end]
                                        faculty_list = [f.strip() for f in faculty_str.split(",")]
                                        
                                        for faculty in faculty_list:
                                            if faculty not in faculty_hours:
                                                faculty_hours[faculty] = 0
                                            faculty_hours[faculty] += 1
                
                faculty_df = pd.DataFrame({
                    'Faculty': list(faculty_hours.keys()),
                    'Hours': list(faculty_hours.values())
                }).sort_values('Hours', ascending=False)
                
                # Display the faculty workload
                st.dataframe(faculty_df, use_container_width=True)
                
                # Plot faculty workload chart
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(faculty_df['Faculty'], faculty_df['Hours'])
                ax.set_xlabel('Faculty')
                ax.set_ylabel('Hours')
                ax.set_title('Faculty Teaching Hours per Week')
                plt.xticks(rotation=90)
                plt.tight_layout()
                
                st.pyplot(fig)
        else:
            st.info("Generate timetables to view analysis")
    
    # Timetables tab
    with tabs[0]:
        st.header("Generated Timetables")
        
        # Advanced options
        with st.expander("Advanced Options"):
            st.subheader("Constraint Weights")
            st.info("Adjust the importance of different constraints (higher values = stricter enforcement)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                faculty_clash_weight = st.slider(
                    "Faculty Clash Penalty", 
                    min_value=1.0, 
                    max_value=20.0, 
                    value=st.session_state.custom_constraint_weights["faculty_clash"],
                    step=0.5,
                    help="Penalty for faculty teaching in multiple sections at the same time"
                )
                
                continuous_periods_weight = st.slider(
                    "Continuous Periods Penalty", 
                    min_value=1.0, 
                    max_value=20.0, 
                    value=st.session_state.custom_constraint_weights["continuous_periods"],
                    step=0.5,
                    help="Penalty for more than 3 continuous teaching periods"
                )
                
                lunch_break_weight = st.slider(
                    "Lunch Break Violation", 
                    min_value=1.0, 
                    max_value=20.0, 
                    value=st.session_state.custom_constraint_weights["lunch_break"],
                    step=0.5,
                    help="Penalty for scheduling classes during lunch break"
                )
            
            with col2:
                lab_continuity_weight = st.slider(
                    "Lab Continuity Penalty", 
                    min_value=1.0, 
                    max_value=20.0, 
                    value=st.session_state.custom_constraint_weights["lab_continuity"],
                    step=0.5,
                    help="Penalty for breaking up lab sessions"
                )
                
                daily_load_weight = st.slider(
                    "Daily Load Balance", 
                    min_value=1.0, 
                    max_value=20.0, 
                    value=st.session_state.custom_constraint_weights["daily_load"],
                    step=0.5,
                    help="Penalty for uneven distribution of classes across days"
                )
                
                section_balance_weight = st.slider(
                    "Section Balance Penalty", 
                    min_value=1.0, 
                    max_value=20.0, 
                    value=st.session_state.custom_constraint_weights["section_balance"],
                    step=0.5,
                    help="Penalty for different patterns across sections"
                )
            
            # Update constraint weights
            st.session_state.custom_constraint_weights = {
                "faculty_clash": faculty_clash_weight,
                "continuous_periods": continuous_periods_weight,
                "lunch_break": lunch_break_weight,
                "lab_continuity": lab_continuity_weight,
                "daily_load": daily_load_weight,
                "section_balance": section_balance_weight
            }
        
        if generate_button and st.session_state.subjects_data and selected_sections:
            # Filter subjects for selected years
            filtered_subjects = [s for s in st.session_state.subjects_data if s["year"] in selected_years]
            
            # Show progress indicator
            with st.spinner("Generating optimized timetables..."):
                # Run the genetic algorithm
                best_timetable, fitness_history = run_genetic_algorithm(
                    filtered_subjects,
                    selected_sections,
                    generations=generations,
                    population_size=population_size,
                    constraint_weights=st.session_state.custom_constraint_weights
                )
                
                # Store results in session state
                st.session_state.section_timetables = {
                    section: convert_to_dataframe(best_timetable[section])
                    for section in best_timetable
                }
                st.session_state.fitness_history = fitness_history
            
            st.success("Timetables generated successfully!")
        
        # Display timetables
        if 'section_timetables' in st.session_state and st.session_state.section_timetables:
            # Show download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download Excel Timetables"):
                    # Create an Excel file with multiple sheets
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                        for section, df in st.session_state.section_timetables.items():
                            df.to_excel(writer, sheet_name=f"Section {section}")
                            
                            # Get the xlsxwriter workbook and worksheet objects
                            workbook = writer.book
                            worksheet = writer.sheets[f"Section {section}"]
                            
                            # Define formats
                            header_format = workbook.add_format({
                                'bold': True,
                                'bg_color': '#D3D3D3',
                                'border': 1
                            })
                            
                            theory_format = workbook.add_format({
                                'bg_color': '#E6F2FF',
                                'border': 1
                            })
                            
                            lab_format = workbook.add_format({
                                'bg_color': '#E6FFE6',
                                'border': 1
                            })
                            
                            lunch_format = workbook.add_format({
                                'bg_color': '#FFCCCC',
                                'border': 1,
                                'bold': True
                            })
                            
                            # Apply formats based on cell content
                            for row_num in range(1, len(df.index) + 1):
                                for col_num in range(1, len(df.columns) + 1):
                                    cell_value = df.iloc[row_num-1, col_num-1]
                                    
                                    if "LUNCH BREAK" in str(cell_value):
                                        worksheet.write(row_num, col_num, cell_value, lunch_format)
                                    elif "Lab" in str(cell_value):
                                        worksheet.write(row_num, col_num, cell_value, lab_format)
                                    elif cell_value != "-":
                                        worksheet.write(row_num, col_num, cell_value, theory_format)
                    
                    # Provide the download button
                    excel_data = excel_buffer.getvalue()
                    st.download_button(
                        label="Download Excel Timetables",
                        data=excel_data,
                        file_name="timetables.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col2:
                if st.button("Download ZIP of HTML Timetables"):
                    # Create a ZIP file with HTML timetables
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        for section, df in st.session_state.section_timetables.items():
                            # Convert DataFrame to HTML with styling
                            html_content = f"""
                            <!DOCTYPE html>
                            <html>
                            <head>
                                <title>Timetable - Section {section}</title>
                                <style>
                                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                    h1 {{ text-align: center; color: #333; }}
                                    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                                    th {{ background-color: #f2f2f2; font-weight: bold; }}
                                    .theory {{ background-color: #E6F2FF; }}
                                    .lab {{ background-color: #E6FFE6; }}
                                    .lunch {{ background-color: #FFCCCC; font-weight: bold; }}
                                    .empty {{ background-color: #ffffff; }}
                                </style>
                            </head>
                            <body>
                                <h1>Timetable - Section {section}</h1>
                                <table>
                                    <tr>
                                        <th>Day / Time</th>
                            """
                            
                            # Add column headers (time slots)
                            for slot in df.columns:
                                html_content += f"<th>{slot}</th>"
                            
                            html_content += "</tr>"
                            
                            # Add rows
                            for day, row in zip(df.index, df.values):
                                html_content += f"<tr><th>{day}</th>"
                                
                                for cell in row:
                                    cell_class = "empty"
                                    if "LUNCH BREAK" in str(cell):
                                        cell_class = "lunch"
                                    elif "Lab" in str(cell) and cell != "-":
                                        cell_class = "lab"
                                    elif cell != "-":
                                        cell_class = "theory"
                                    
                                    html_content += f'<td class="{cell_class}">{cell}</td>'
                                
                                html_content += "</tr>"
                            
                            html_content += """
                                </table>
                            </body>
                            </html>
                            """
                            
                            # Add HTML file to ZIP
                            zip_file.writestr(f"timetable_section_{section}.html", html_content)
                    
                    # Provide the download button
                    zip_data = zip_buffer.getvalue()
                    st.download_button(
                        label="Download ZIP of HTML Timetables",
                        data=zip_data,
                        file_name="timetables_html.zip",
                        mime="application/zip"
                    )
            
            # Display timetables in tabs
            section_tabs = st.tabs(selected_sections)
            
            for i, section in enumerate(selected_sections):
                with section_tabs[i]:
                    if section in st.session_state.section_timetables:
                        timetable_df = st.session_state.section_timetables[section]
                        
                        # Display the timetable with styling
                        st.dataframe(
                            timetable_df,
                            use_container_width=True,
                            height=400
                        )
        else:
            if not st.session_state.subjects_data:
                st.info("Please upload subject data or use sample data")
            elif not selected_sections:
                st.info("Please select at least one section")
            else:
                st.info("Click 'Generate Timetables' to create optimized timetables")

if __name__ == "__main__":
    main()