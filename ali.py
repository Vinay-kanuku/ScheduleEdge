import streamlit as st
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
import copy
import matplotlib.pyplot as plt
from io import BytesIO
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Constants for the timetable
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
SLOTS = ['9:00-9:40', '9:40-10:20', '10:20-11:00', '11:20-12:00', '12:00-12:40',
         '12:40-13:20', '13:20-14:00', '14:00-14:40', '14:40-15:20', '15:20-16:00']

# Constraint weights for sophisticated fitness calculation
CONSTRAINT_WEIGHTS = {
    'faculty_conflict': 10.0,
    'lab_continuity': 8.0,
    'subject_distribution': 6.0,
    'lunch_break': 9.0,
    'consecutive_lectures': 4.0,
    'daily_load_balance': 5.0,
    'preferred_slots': 3.0,
    'multi_section_faculty_conflict': 12.0  # New constraint for multi-section
}

@dataclass
class TimeSlot:
    day: str
    start_time: str
    duration: float

@dataclass
class Session:
    subject: str
    faculty: List[str]
    time_slot: TimeSlot
    section: str  # Added section identifier
    batch: str = None  # Only for labs

class MultiSectionTimetableGenerator:
    def __init__(self, subjects: List[Dict], sections: List[str], population_size: int = 50):
        self.population_size = population_size
        self.subjects = subjects
        self.sections = sections
        self.subject_shortcuts = {s["name"]: s["name"][:4].upper() for s in subjects}
        self.fitness_history = []
        self.progress_value = 0
        
        # Pre-compute valid slots to avoid repeated calculations
        self.valid_slots = {
            1.0: [slot for slot in SLOTS if (slot < "12:20-13:00" and SLOTS.index(slot) + 2 <= len(SLOTS)) or slot > "13:00-13:40"],
            2.5: [slot for slot in SLOTS if (slot < "12:20-13:00" and SLOTS.index(slot) + 5 <= len(SLOTS)) or slot > "13:00-13:40"]
        }
        
        # Cache for fitness scores to avoid recalculation
        self.fitness_cache = {}
    
    def update_progress(self, value: int):
        """Update progress value"""
        self.progress_value = value
        
    def get_progress(self) -> int:
        """Get current progress value"""
        return self.progress_value

    def generate_timetables(self) -> Dict[str, pd.DataFrame]:
        """Main generation loop for multiple sections"""
        best_solution = None
        best_fitness = float('-inf')
        
        # Create initial population
        population = self._initialize_population()
        
        # Main GA loop - reduced to 10 generations for faster results
        for generation in range(10):
            if generation > 0:
                population = self._evolve_population(population)
            
            # Parallel fitness evaluation
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all solutions for evaluation
                future_to_solution = {
                    executor.submit(self._get_cached_fitness, solution): solution 
                    for solution in population
                }
                
                # Process results as they complete
                for future in as_completed(future_to_solution):
                    solution = future_to_solution[future]
                    try:
                        fitness = future.result()
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_solution = solution
                    except Exception as e:
                        print(f"Error evaluating solution: {e}")
            
            self.fitness_history.append(best_fitness)
            self.update_progress(int((generation + 1) / 10 * 100))
            
            # Early termination if we have a good solution
            if best_fitness > 85:  # Slightly lower threshold for multi-section
                break
        
        # Convert best solution to separate dataframes for each section
        section_timetables = {}
        for section in self.sections:
            section_sessions = [s for s in best_solution if s.section == section]
            section_timetables[section] = self._convert_to_dataframe(section_sessions)
        
        return section_timetables

    def _initialize_population(self) -> List[List[Session]]:
        """Initialize random population of multi-section timetables"""
        population = []
        for _ in range(self.population_size):
            population.append(self._create_multi_section_timetable())
        return population

    def _create_multi_section_timetable(self) -> List[Session]:
        """Create a single random timetable for all sections"""
        timetable = []
        
        # For each section, add all required sessions
        for section in self.sections:
            # Add regular subjects
            for subject in [s for s in self.subjects if not s["is_lab"]]:
                slots_needed = int(subject["weekly_hours"])
                for _ in range(slots_needed):
                    time_slot = self._get_random_time_slot(1)
                    session = Session(
                        subject=subject["name"],
                        faculty=subject["faculty"],
                        time_slot=time_slot,
                        section=section
                    )
                    timetable.append(session)
            
            # Add lab sessions
            for subject in [s for s in self.subjects if s["is_lab"]]:
                for batch in ["1", "2"]:
                    time_slot = self._get_random_time_slot(2.5)
                    session = Session(
                        subject=subject["name"],
                        faculty=subject["faculty"],
                        time_slot=time_slot,
                        section=section,
                        batch=batch
                    )
                    timetable.append(session)
        
        return timetable

    def _get_cached_fitness(self, solution: List[Session]) -> float:
        """Get fitness from cache or calculate if not present"""
        # Create a hashable representation of the solution
        solution_key = self._get_solution_hash(solution)
        
        if solution_key in self.fitness_cache:
            return self.fitness_cache[solution_key]
        
        fitness = self._evaluate_fitness(solution)
        self.fitness_cache[solution_key] = fitness
        return fitness
    
    def _get_solution_hash(self, solution: List[Session]) -> tuple:
        """Create a hashable representation of a solution"""
        return tuple(sorted([
            (s.subject, s.time_slot.day, s.time_slot.start_time, 
             s.time_slot.duration, s.section, s.batch if s.batch else '')
            for s in solution
        ]))

    def _evaluate_fitness(self, solution: List[Session]) -> float:
        """Calculate weighted fitness score based on multiple constraints"""
        score = 100.0
        penalties = {
            'faculty_conflict': 0,
            'lab_continuity': 0,
            'subject_distribution': 0,
            'lunch_break': 0,
            'consecutive_lectures': 0,
            'daily_load_balance': 0,
            'preferred_slots': 0,
            'multi_section_faculty_conflict': 0
        }
        
        # Faculty conflicts within a section
        section_faculty_slots = {}
        for section in self.sections:
            section_faculty_slots[section] = {}
        
        # Cross-section faculty conflicts
        all_faculty_slots = {}
        
        for session in solution:
            section = session.section
            for faculty in session.faculty:
                key = (session.time_slot.day, session.time_slot.start_time)
                
                # Check section-specific conflicts
                if section in section_faculty_slots:
                    if key in section_faculty_slots[section] and faculty in section_faculty_slots[section][key]:
                        penalties['faculty_conflict'] += 1
                    if key not in section_faculty_slots[section]:
                        section_faculty_slots[section][key] = []
                    section_faculty_slots[section][key].append(faculty)
                
                # Check cross-section conflicts for the same faculty
                if key in all_faculty_slots and faculty in all_faculty_slots[key]:
                    penalties['multi_section_faculty_conflict'] += 1
                if key not in all_faculty_slots:
                    all_faculty_slots[key] = []
                all_faculty_slots[key].append(faculty)
        
        # Evaluate per-section constraints
        for section in self.sections:
            section_sessions = [s for s in solution if s.section == section]
            
            # Lab continuity
            for session in [s for s in section_sessions if s.batch]:
                if not self._is_lab_continuous(session):
                    penalties['lab_continuity'] += 1
            
            # Subject distribution and consecutive lectures
            subject_day_slots = {}
            for session in section_sessions:
                key = (session.subject, session.time_slot.day)
                if key not in subject_day_slots:
                    subject_day_slots[key] = []
                subject_day_slots[key].append(session.time_slot.start_time)
                
            # Only check once we have all slots
            for key, slots in subject_day_slots.items():
                if len(slots) > 2:
                    penalties['subject_distribution'] += 1
                
                # Check for more than 2 consecutive lectures
                sorted_slots = sorted(slots)
                for i in range(len(sorted_slots)-2):
                    if self._are_slots_consecutive(sorted_slots[i:i+3]):
                        penalties['consecutive_lectures'] += 1
            
            # Daily load balance per section
            day_loads = {}
            for session in section_sessions:
                day = session.time_slot.day
                day_loads[day] = day_loads.get(day, 0) + session.time_slot.duration
            
            std_dev = np.std(list(day_loads.values())) if day_loads else 0
            penalties['daily_load_balance'] += std_dev
        
        # Normalize daily load balance by number of sections
        penalties['daily_load_balance'] /= max(1, len(self.sections))
        
        # Apply weighted penalties
        for constraint, penalty in penalties.items():
            score -= penalty * CONSTRAINT_WEIGHTS[constraint]
        
        return max(0, score)

    def _get_random_time_slot(self, duration: float) -> TimeSlot:
        """Generate a random valid time slot using pre-computed valid slots"""
        day = random.choice(DAYS)
        start_time = random.choice(self.valid_slots[duration])
        return TimeSlot(day=day, start_time=start_time, duration=duration)

    def _evolve_population(self, population: List[List[Session]]) -> List[List[Session]]:
        """Evolve population using genetic operators - optimized version"""
        new_population = []
        
        # Get fitness for each solution
        fitness_values = [self._get_cached_fitness(solution) for solution in population]
        
        # Pair solutions with fitness
        solutions_with_fitness = list(zip(population, fitness_values))
        
        # Sort by fitness (descending)
        solutions_with_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Elitism: Keep top 20% solutions
        elite_size = max(1, self.population_size // 5)
        new_population.extend([copy.deepcopy(sol) for sol, _ in solutions_with_fitness[:elite_size]])
        
        # Create remaining solutions through selection, crossover, mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select(solutions_with_fitness)
            parent2 = self._tournament_select(solutions_with_fitness)
            
            # Crossover
            if random.random() < 0.8:
                child = self._crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)
            
            # Mutation
            if random.random() < 0.3:  # Increased mutation rate
                child = self._mutate(child)
            
            new_population.append(child)
            
        return new_population

    def _tournament_select(self, solutions_with_fitness, tournament_size: int = 3):
        """Select best solution from random tournament"""
        tournament = random.sample(solutions_with_fitness, min(tournament_size, len(solutions_with_fitness)))
        return max(tournament, key=lambda x: x[1])[0]

    def _crossover(self, parent1: List[Session], parent2: List[Session]) -> List[Session]:
        """Perform crossover between two parent solutions"""
        # Group by section and subject
        p1_sections = {}
        p2_sections = {}
        
        for section in self.sections:
            p1_section_sessions = [s for s in parent1 if s.section == section]
            p2_section_sessions = [s for s in parent2 if s.section == section]
            
            p1_sections[section] = self._group_by_subject(p1_section_sessions)
            p2_sections[section] = self._group_by_subject(p2_section_sessions)
        
        child_sessions = []
        
        # For each section, select subjects from either parent
        for section in self.sections:
            p1_subjects = list(p1_sections[section].keys())
            crossover_point = random.randint(1, len(p1_subjects) - 1) if len(p1_subjects) > 1 else 1
            
            for subject in p1_subjects[:crossover_point]:
                child_sessions.extend(p1_sections[section][subject])
            
            for subject in p1_subjects[crossover_point:]:
                if subject in p2_sections[section]:
                    child_sessions.extend(p2_sections[section][subject])
                else:
                    child_sessions.extend(p1_sections[section][subject])
        
        return child_sessions

    def _mutate(self, solution: List[Session]) -> List[Session]:
        """Apply mutation operators to solution"""
        mutated = copy.deepcopy(solution)
        
        # Apply multiple mutations
        num_mutations = random.randint(1, 3)
        mutations = [self._swap_time_slots, self._shift_session, self._swap_days]
        
        for _ in range(num_mutations):
            mutation = random.choice(mutations)
            mutated = mutation(mutated)
        
        return mutated

    def _swap_time_slots(self, solution: List[Session]) -> List[Session]:
        """Swap time slots of two random sessions within the same section"""
        # Group sessions by section
        section_sessions = {section: [] for section in self.sections}
        for session in solution:
            section_sessions[session.section].append(session)
        
        # Select a random section
        section = random.choice(self.sections)
        sessions = section_sessions[section]
        
        if len(sessions) < 2:
            return solution
        
        # Select two random sessions with the same duration
        compatible_pairs = [(i, j) for i in range(len(sessions)) 
                            for j in range(i+1, len(sessions)) 
                            if sessions[i].time_slot.duration == sessions[j].time_slot.duration]
        
        if compatible_pairs:
            idx1, idx2 = random.choice(compatible_pairs)
            sessions[idx1].time_slot, sessions[idx2].time_slot = (
                sessions[idx2].time_slot, sessions[idx1].time_slot
            )
            
        return solution

    def _shift_session(self, solution: List[Session]) -> List[Session]:
        """Shift a random session to a new time slot"""
        if not solution:
            return solution
            
        idx = random.randint(0, len(solution) - 1)
        solution[idx].time_slot = self._get_random_time_slot(solution[idx].time_slot.duration)
        return solution

    def _swap_days(self, solution: List[Session]) -> List[Session]:
        """Swap all sessions between two random days for a single section"""
        section = random.choice(self.sections)
        day1, day2 = random.sample(DAYS, 2)
        
        for session in solution:
            if session.section == section:
                if session.time_slot.day == day1:
                    session.time_slot.day = day2
                elif session.time_slot.day == day2:
                    session.time_slot.day = day1
                    
        return solution

    def _group_by_subject(self, solution: List[Session]) -> Dict[str, List[Session]]:
        """Group sessions by subject"""
        groups = {}
        for session in solution:
            if session.subject not in groups:
                groups[session.subject] = []
            groups[session.subject].append(session)
        return groups

    def _is_lab_continuous(self, session: Session) -> bool:
        """Check if lab session spans continuous slots"""
        if not session.batch:
            return True
            
        start_idx = SLOTS.index(session.time_slot.start_time)
        duration_slots = int(session.time_slot.duration * 2)
        
        for i in range(duration_slots):
            if start_idx + i >= len(SLOTS):
                return False
            next_slot = SLOTS[start_idx + i]
            if next_slot == "11:00-11:20":
                return False
                
        return True

    def _are_slots_consecutive(self, slots: List[str]) -> bool:
        """Check if time slots are consecutive"""
        indices = [SLOTS.index(slot) for slot in slots]
        indices.sort()
        return all(indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1))

    def _convert_to_dataframe(self, solution: List[Session]) -> pd.DataFrame:
        """Convert solution to pandas DataFrame format"""
        df = pd.DataFrame(index=DAYS, columns=SLOTS)
        df.fillna("-", inplace=True)
        
        # Create a break row between morning and afternoon slots
        lunch_slot = "11:00-11:20"
        if lunch_slot not in df.columns:
            # Get all columns in order
            all_cols = list(df.columns)
            # Find where to insert lunch
            lunch_idx = 0
            for i, col in enumerate(all_cols):
                if col > lunch_slot:
                    lunch_idx = i
                    break
            # Create new column list with lunch inserted
            new_cols = all_cols[:lunch_idx] + [lunch_slot] + all_cols[lunch_idx:]
            # Reindex the DataFrame
            df = df.reindex(columns=new_cols)
        
        # Add lunch break
        for day in DAYS:
            df.at[day, lunch_slot] = "LUNCH BREAK"
        
        for session in solution:
            start_idx = SLOTS.index(session.time_slot.start_time)
            duration_slots = int(session.time_slot.duration * 2)
            
            entry = self._format_session_entry(session)
            
            for i in range(duration_slots):
                if start_idx + i < len(SLOTS):
                    slot = SLOTS[start_idx + i]
                    if slot != lunch_slot:  # Skip lunch slot
                        df.at[session.time_slot.day, slot] = entry
        
        return df

    def _format_session_entry(self, session: Session) -> str:
        """Format session entry for display"""
        if session.batch:
            return f"{self.subject_shortcuts[session.subject]} (B{session.batch}: {', '.join(session.faculty)})"
        return f"{self.subject_shortcuts[session.subject]} ({session.faculty[0]})"

    def get_fitness_history(self) -> List[float]:
        """Return fitness history for plotting"""
        return self.fitness_history


def main():
    st.set_page_config(page_title="Multi-Section Academic Timetable Generator", layout="wide")
    
    st.title("🎓 Multi-Section Academic Timetable Generator")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Basic Settings
        population_size = st.slider("Population Size", 20, 200, 50)
        
        # Section Configuration
        st.subheader("Sections")
        num_sections = st.number_input("Number of Sections", 1, 5, 3)
        sections = [f"{chr(65 + i)}" for i in range(num_sections)]  # A, B, C, etc.
        
        # Display current sections
        st.write("Sections:", ", ".join(sections))
        
        # Subject Configuration
        st.subheader("Subjects")
        
        if 'subjects_data' not in st.session_state:
            st.session_state.subjects_data = []
        
        num_subjects = st.number_input("Number of Subjects", 1, 10, 3)
        
        # Clear button for subjects
        if st.button("Clear Subjects"):
            st.session_state.subjects_data = []
        
        # Subject input form
        with st.form("subject_form"):
            subjects_data = []
            for i in range(num_subjects):
                st.markdown(f"**Subject {i+1}**")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Check if we have existing data and it's a list with enough items
                    if isinstance(st.session_state.subjects_data, list) and i < len(st.session_state.subjects_data):
                        name = st.text_input(
                            f"Name #{i+1}", 
                            value=st.session_state.subjects_data[i]["name"]
                        )
                        weekly_hours = st.number_input(
                            f"Weekly Hours #{i+1}", 
                            1.0, 10.0, 
                            value=st.session_state.subjects_data[i]["weekly_hours"]
                        )
                    else:
                        name = st.text_input(f"Name #{i+1}", value=f"Subject {i+1}")
                        weekly_hours = st.number_input(f"Weekly Hours #{i+1}", 1.0, 10.0, value=4.0)
                
                with col2:
                    if isinstance(st.session_state.subjects_data, list) and i < len(st.session_state.subjects_data):
                        faculty = st.text_input(
                            f"Faculty #{i+1}", 
                            value=st.session_state.subjects_data[i]["faculty"][0]
                        )
                        is_lab = st.checkbox(
                            f"Is Lab #{i+1}", 
                            value=st.session_state.subjects_data[i]["is_lab"]
                        )
                    else:
                        faculty = st.text_input(f"Faculty #{i+1}", value=f"Prof. {i+1}")
                        is_lab = st.checkbox(f"Is Lab #{i+1}", value=False)
                
                subjects_data.append({
                    "name": name,
                    "faculty": [faculty],
                    "weekly_hours": weekly_hours,
                    "is_lab": is_lab
                })
            
            if st.form_submit_button("Save Subjects"):
                st.session_state.subjects_data = subjects_data
                st.success("Subjects saved!")

    # Main content area
    if not st.session_state.subjects_data:
        st.warning("Please configure and save subjects in the sidebar first.")
        return
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Generate", "Analysis", "Export"])
    
    with tab1:
        st.header("Generate Timetables")
        
        generate_button = st.button("Generate Timetables for All Sections", type="primary")
        early_stop = st.checkbox("Early termination (faster)", value=True)
        
        if generate_button:
            try:
                # Progress container
                progress_container = st.empty()
                progress_bar = progress_container.progress(0)
                
                # Create placeholder for fitness plot
                plot_container = st.empty()
                results_container = st.empty()
                
                # Create generator
                generator = MultiSectionTimetableGenerator(
                    subjects=st.session_state.subjects_data,
                    sections=sections,
                    population_size=population_size
                )
                
                # Run generation in the main thread with progress updates
                with st.spinner("Generating timetables for all sections..."):
                    # Start time tracking
                    start_time = time.time()
                    
                    # Generate timetables
                    section_timetables = generator.generate_timetables()
                    
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    
                    # Store results
                    st.session_state.section_timetables = section_timetables
                    st.session_state.generator = generator
                    
                    # Update progress bar
                    progress_bar.progress(100)
                    
                    # Display results
                    results_container.success(f"Timetables generated successfully in {elapsed_time:.2f} seconds!")
                    
                    # Plot fitness history
                    if generator.fitness_history:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(generator.fitness_history, '-o')
                        ax.set_title('Timetable Evolution Progress')
                        ax.set_xlabel('Generation')
                        ax.set_ylabel('Best Fitness Score')
                        ax.grid(True)
                        plot_container.pyplot(fig)
                        plt.close()
                
            except Exception as e:
                st.error(f"Error generating timetables: {str(e)}")
        
        # Display current timetables if available
        if 'section_timetables' in st.session_state:
            # Create tabs for each section
            section_tabs = st.tabs(sections)
            
            for i, section in enumerate(sections):
                with section_tabs[i]:
                    st.subheader(f"Section {section} Timetable")
                    
                    if section in st.session_state.section_timetables:
                        # Style the dataframe
                        def highlight_cells(val):
                            if val == "LUNCH BREAK":
                                return 'background-color: #f4d03f'
                            elif val != "-":
                                return 'background-color: #aed6f1'
                            return ''
                        
                        timetable = st.session_state.section_timetables[section]
                        styled_timetable = timetable.style.applymap(highlight_cells)
                        st.dataframe(styled_timetable, use_container_width=True)
                    else:
                        st.warning(f"No timetable available for Section {section}")
    
    with tab2:
        if 'generator' in st.session_state and 'section_timetables' in st.session_state:
            st.header("Timetable Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Evolution Progress")
                fitness_history = st.session_state.generator.get_fitness_history()
                if fitness_history:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(fitness_history, '-o')
                    ax.set_title('Fitness Evolution')
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Fitness Score')
                    ax.grid(True)
                    st.pyplot(fig)
                else:
                    st.info("No evolution data available yet.")
            
            with col2:
                st.subheader("Section Comparison")
                section_free_slots = {}
                
                for section, timetable in st.session_state.section_timetables.items():
                    section_free_slots[section] = timetable.apply(
                        lambda x: x.value_counts().get("-", 0), axis=1
                    ).mean()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                pd.Series(section_free_slots).plot(kind='bar', ax=ax)
                ax.set_title('Average Free Slots per Section')
                ax.set_xlabel('Section')
                ax.set_ylabel('Average Number of Free Slots')
                st.pyplot(fig)
            
            # Faculty load analysis
            st.subheader("Faculty Load Analysis")
            
            faculty_loads = {}
            for section, timetable in st.session_state.section_timetables.items():
                for subject in st.session_state.subjects_data:
                    faculty_name = subject["faculty"][0]
                    shortcut = subject["name"][:4].upper()
                    
                    if faculty_name not in faculty_loads:
                        faculty_loads[faculty_name] = 0
                    
                    # Count occurrences in timetable
                    for day in DAYS:
                        for slot in SLOTS:
                            if slot in timetable.columns and shortcut in str(timetable.at[day, slot]):
                                faculty_loads[faculty_name] += 1
            
            faculty_df = pd.DataFrame({'Hours': faculty_loads}).sort_values('Hours', ascending=False)
            st.dataframe(faculty_df, use_container_width=True)
    
    with tab3:
        if 'section_timetables' in st.session_state:
            st.header("Export Options")
            
            # Select section to export
            section_to_export = st.selectbox("Select Section", ["All"] + sections)
            
            col1, col2 = st.columns(2)
            
            if section_to_export == "All":
                with col1:
                    # Export all to Excel
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        for section, timetable in st.session_state.section_timetables.items():
                            timetable.to_excel(writer, sheet_name=f'Section {section}')
                    
                    st.download_button(
                        label="Download All Sections (Excel)",
                        data=buffer.getvalue(),
                        file_name="all_timetables.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    # Export all to CSV (zip file with multiple CSVs)
                    zip_buffer = BytesIO()
                    
                    import zipfile
                    with zipfile.ZipFile(zip_buffer, 'w') as zf:
                        for section, timetable in st.session_state.section_timetables.items():
                            csv_buffer = BytesIO()
                            timetable.to_csv(csv_buffer)
                            zf.writestr(f"section_{section}_timetable.csv", csv_buffer.getvalue())
                    
 

                    st.download_button(
                        label="Download All Sections (ZIP of CSVs)",
                        data=zip_buffer.getvalue(),
                        file_name="all_timetables.zip",
                        mime="application/zip"
                    )
            else:
                # Export single section
                section_timetable = st.session_state.section_timetables[section_to_export]
                
                with col1:
                    # Export to Excel
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        section_timetable.to_excel(writer, sheet_name=f'Section {section_to_export}')
                    
                    st.download_button(
                        label=f"Download Section {section_to_export} (Excel)",
                        data=buffer.getvalue(),
                        file_name=f"section_{section_to_export}_timetable.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                with col2:
                    # Export to CSV
                    csv_buffer = BytesIO()
                    section_timetable.to_csv(csv_buffer)
                    
                    st.download_button(
                        label=f"Download Section {section_to_export} (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=f"section_{section_to_export}_timetable.csv",
                        mime="text/csv"
                    )
            
            # Add print-friendly view
            st.subheader("Print-Friendly View")
            
            if section_to_export == "All":
                for section, timetable in st.session_state.section_timetables.items():
                    st.markdown(f"### Section {section}")
                    
                    # Style the dataframe for print
                    def highlight_cells_html(val):
                        if val == "LUNCH BREAK":
                            return f'<div style="background-color: #f4d03f; padding: 5px; text-align: center;">{val}</div>'
                        elif val != "-":
                            return f'<div style="background-color: #aed6f1; padding: 5px; text-align: center;">{val}</div>'
                        return f'<div style="padding: 5px; text-align: center;">{val}</div>'
                    
                    # Convert to HTML with styling
                    html_table = timetable.applymap(highlight_cells_html).to_html(escape=False)
                    st.markdown(html_table, unsafe_allow_html=True)
                    st.markdown("---")
            else:
                section_timetable = st.session_state.section_timetables[section_to_export]
                
                # Style the dataframe for print
                def highlight_cells_html(val):
                    if val == "LUNCH BREAK":
                        return f'<div style="background-color: #f4d03f; padding: 5px; text-align: center;">{val}</div>'
                    elif val != "-":
                        return f'<div style="background-color: #aed6f1; padding: 5px; text-align: center;">{val}</div>'
                    return f'<div style="padding: 5px; text-align: center;">{val}</div>'
                
                # Convert to HTML with styling
                html_table = section_timetable.applymap(highlight_cells_html).to_html(escape=False)
                st.markdown(html_table, unsafe_allow_html=True)
        else:
            st.info("Generate timetables first to enable export options.")
    
    # Add additional features
    st.sidebar.markdown("---")
    with st.sidebar.expander("Advanced Options"):
        st.markdown("### Constraint Weights")
        st.info("Adjust weights to prioritize different constraints")
        
        # Allow adjusting constraint weights
        constraint_weights = {}
        for constraint, weight in CONSTRAINT_WEIGHTS.items():
            display_name = constraint.replace('_', ' ').title()
            constraint_weights[constraint] = st.slider(
                display_name, 
                0.0, 20.0, 
                float(weight),
                help=f"Set importance of {display_name} constraint"
            )
        
        # Apply button for constraint weights
        if st.button("Apply Custom Weights"):
            # Create a copy of the original weights
            custom_weights = CONSTRAINT_WEIGHTS.copy()
            # Update with user values
            for constraint, weight in constraint_weights.items():
                custom_weights[constraint] = weight
            
            # Store for next generation
            st.session_state.custom_constraint_weights = custom_weights
            st.success("Custom weights will be applied to next generation")
    
    # Add conflict visualization tab
    if 'section_timetables' in st.session_state:
        with st.expander("Conflict Visualization"):
            st.subheader("Cross-section Conflict Analysis")
            
            # Find all faculty
            all_faculty = set()
            for subject in st.session_state.subjects_data:
                all_faculty.update(subject["faculty"])
            
            # Create conflict matrix
            conflict_matrix = {}
            for faculty in all_faculty:
                faculty_schedule = {}
                
                # Track all slots where this faculty teaches
                for section, timetable in st.session_state.section_timetables.items():
                    for day in DAYS:
                        for slot in [s for s in SLOTS if s in timetable.columns]:
                            cell_value = timetable.at[day, slot]
                            if faculty in str(cell_value):
                                key = (day, slot)
                                if key not in faculty_schedule:
                                    faculty_schedule[key] = []
                                faculty_schedule[key].append(section)
                
                # Count conflicts (slots where faculty teaches in multiple sections)
                conflicts = [sections for slot, sections in faculty_schedule.items() if len(sections) > 1]
                conflict_matrix[faculty] = len(conflicts)
            
            # Display conflicts
            conflict_df = pd.DataFrame({
                'Faculty': list(conflict_matrix.keys()),
                'Conflicts': list(conflict_matrix.values())
            }).sort_values('Conflicts', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(conflict_df, use_container_width=True)
            
            with col2:
                # Create a bar chart of faculty conflicts
                if not conflict_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(conflict_df['Faculty'], conflict_df['Conflicts'])
                    ax.set_title('Faculty Cross-section Conflicts')
                    ax.set_xlabel('Faculty')
                    ax.set_ylabel('Number of Conflicts')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No conflicts data available")
    
    # Add a features section at the bottom
    st.sidebar.markdown("---")
    with st.sidebar.expander("About This App"):
        st.markdown("""
        ### Features
        - Multi-section timetable generation
        - Optimized genetic algorithm
        - Customizable subjects and faculty
        - Lab session support
        - Export to Excel, CSV, or print view
        - Fitness evolution tracking
        - Faculty load analysis
        - Cross-section conflict detection
        
        ### How It Works
        This app uses a genetic algorithm to generate optimal timetables for multiple sections simultaneously, 
        ensuring minimal conflicts while maintaining educational constraints.
        """)

if __name__ == "__main__":
    main()