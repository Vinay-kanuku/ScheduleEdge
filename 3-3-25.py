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
    'preferred_slots': 3.0
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
    batch: str = None  # Only for labs

class TimetableGenerator:
    def __init__(self, subjects: List[Dict], population_size: int = 50):
        self.population_size = population_size
        self.subjects = subjects
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

    def generate_timetable(self, section: str) -> pd.DataFrame:
        """Main generation loop with optimized GA approach"""
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
            if best_fitness > 90:
                break
        
        return self._convert_to_dataframe(best_solution)

    def _initialize_population(self) -> List[List[Session]]:
        """Initialize random population of timetables"""
        # Generate diverse population
        population = []
        for _ in range(self.population_size):
            population.append(self._create_single_timetable())
        return population

    def _create_single_timetable(self) -> List[Session]:
        """Create a single random timetable"""
        timetable = []
        
        # Add regular subjects
        for subject in [s for s in self.subjects if not s["is_lab"]]:
            slots_needed = int(subject["weekly_hours"])
            for _ in range(slots_needed):
                time_slot = self._get_random_time_slot(1)
                session = Session(
                    subject=subject["name"],
                    faculty=subject["faculty"],
                    time_slot=time_slot
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
             s.time_slot.duration, s.batch if s.batch else '')
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
            'preferred_slots': 0
        }
        
        # Faculty conflicts - optimized check
        faculty_slots = {}
        for session in solution:
            for faculty in session.faculty:
                key = (session.time_slot.day, session.time_slot.start_time)
                if key in faculty_slots and faculty in faculty_slots[key]:
                    penalties['faculty_conflict'] += 1
                if key not in faculty_slots:
                    faculty_slots[key] = []
                faculty_slots[key].append(faculty)
        
        # Lab continuity
        for session in [s for s in solution if s.batch]:
            if not self._is_lab_continuous(session):
                penalties['lab_continuity'] += 1
        
        # Subject distribution and consecutive lectures
        subject_day_slots = {}
        for session in solution:
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
        
        # Daily load balance
        day_loads = {}
        for session in solution:
            day = session.time_slot.day
            day_loads[day] = day_loads.get(day, 0) + session.time_slot.duration
        
        std_dev = np.std(list(day_loads.values())) if day_loads else 0
        penalties['daily_load_balance'] = std_dev
        
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
        p1_sessions = self._group_by_subject(parent1)
        p2_sessions = self._group_by_subject(parent2)
        
        child_sessions = []
        subjects = list(p1_sessions.keys())
        crossover_point = random.randint(1, len(subjects) - 1) if len(subjects) > 1 else 1
        
        for subject in subjects[:crossover_point]:
            child_sessions.extend(p1_sessions[subject])
        
        for subject in subjects[crossover_point:]:
            child_sessions.extend(p2_sessions[subject])
        
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
        """Swap time slots of two random sessions"""
        if len(solution) < 2:
            return solution
        
        idx1, idx2 = random.sample(range(len(solution)), 2)
        if solution[idx1].time_slot.duration == solution[idx2].time_slot.duration:
            solution[idx1].time_slot, solution[idx2].time_slot = (
                solution[idx2].time_slot, solution[idx1].time_slot
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
        """Swap all sessions between two random days"""
        day1, day2 = random.sample(DAYS, 2)
        for session in solution:
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
    st.set_page_config(page_title="Academic Timetable Generator", layout="wide")
    
    st.title("🎓 Academic Timetable Generator")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Basic Settings
        population_size = st.slider("Population Size", 20, 200, 50)
        
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
        st.header("Generate Timetable")
        col1, col2 = st.columns(2)
        
        with col1:
            section = st.text_input("Section", "A")
        
        with col2:
            generate_button = st.button("Generate Timetable", type="primary")
            early_stop = st.checkbox("Early termination (faster)", value=True)
        
        if generate_button:
            try:
                # Progress container
                progress_container = st.empty()
                progress_bar = progress_container.progress(0)
                
                # Create placeholder for fitness plot
                plot_container = st.empty()
                results_container = st.empty()
                
                # Create generator without caching to avoid tuple conversion issues
                generator = TimetableGenerator(
                    subjects=st.session_state.subjects_data,
                    population_size=population_size
                )
                
                # Run generation in the main thread with progress updates
                with st.spinner("Generating timetable..."):
                    # Start time tracking
                    start_time = time.time()
                    
                    # Generate timetable
                    timetable = generator.generate_timetable(section)
                    
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    
                    # Store results
                    st.session_state.timetable = timetable
                    st.session_state.generator = generator
                    
                    # Update progress bar
                    progress_bar.progress(100)
                    
                    # Display results
                    results_container.success(f"Timetable generated successfully in {elapsed_time:.2f} seconds!")
                    
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
                st.error(f"Error generating timetable: {str(e)}")
        
        # Display current timetable if available
        if 'timetable' in st.session_state:
            st.subheader("Current Timetable")
            
            # Style the dataframe
            def highlight_cells(val):
                if val == "LUNCH BREAK":
                    return 'background-color: #ffcdd2'
                elif val != "-":
                    return 'background-color: #e3f2fd'
                return ''
            
            styled_timetable = st.session_state.timetable.style.applymap(highlight_cells)
            st.dataframe(styled_timetable, use_container_width=True)
    
    with tab2:
        if 'generator' in st.session_state and 'timetable' in st.session_state:
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
                st.subheader("Daily Distribution")
                daily_counts = st.session_state.timetable.apply(
                    lambda x: x.value_counts().get("-", 0), axis=1
                )
                fig, ax = plt.subplots(figsize=(10, 6))
                daily_counts.plot(kind='bar', ax=ax)
                ax.set_title('Free Slots per Day')
                ax.set_xlabel('Day')
                ax.set_ylabel('Number of Free Slots')
                st.pyplot(fig)
            
            # Subject distribution
            st.subheader("Subject Distribution")
            subject_counts = pd.DataFrame()
            for subject in st.session_state.subjects_data:
                shortcut = subject["name"][:4].upper()
                for day in DAYS:
                    count = sum(1 for slot in SLOTS 
                              if slot in st.session_state.timetable.columns and 
                              shortcut in str(st.session_state.timetable.at[day, slot]))
                    subject_counts.at[subject["name"], day] = count
            
            st.dataframe(subject_counts, use_container_width=True)
    
    with tab3:
        if 'timetable' in st.session_state:
            st.header("Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export to CSV
                csv = st.session_state.timetable.to_csv(index=True)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="timetable.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export to Excel
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    st.session_state.timetable.to_excel(writer, sheet_name='Timetable')
                
                st.download_button(
                    label="Download Excel",
                    data=buffer.getvalue(),
                    file_name="timetable.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()