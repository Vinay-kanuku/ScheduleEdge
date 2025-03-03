import streamlit as st
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import google.generativeai as genai
from dataclasses import dataclass
import copy
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from tqdm import tqdm
import base64
from io import BytesIO

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


def create_timetable(generator: 'TimetableGenerator') -> List[Session]:
    return generator._create_single_timetable()

class TimetableGenerator:
    def __init__(self, api_key: str, subjects: List[Dict], population_size: int = 50, num_processes: int = 4):
        self.population_size = population_size
        self.num_processes = num_processes
        self.subjects = subjects
        self.subject_shortcuts = {s["name"]: s["name"][:4].upper() for s in subjects}
        self.fitness_history = []
        
        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # For progress tracking
        self.progress_callback = None
        
    def set_progress_callback(self, callback):
        self.progress_callback = callback
        
    def generate_timetable(self, section: str) -> pd.DataFrame:
        """Main generation loop combining GA and AI approaches"""
        best_solution = None
        best_fitness = float('-inf')
        
        # First attempt: Pure GA approach with parallel processing
        for generation in range(20):
            if best_solution is None:
                population = self._initialize_population()
            else:
                population = self._evolve_population(population)
            
            # Parallel fitness evaluation
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                fitness_scores = list(executor.map(self._evaluate_fitness, population))
            
            for solution, fitness in zip(population, fitness_scores):
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution
            
            self.fitness_history.append(best_fitness)
            
            # Update progress if callback is set
            if self.progress_callback:
                self.progress_callback(generation, 20)
            
            if self._is_solution_valid(best_solution):
                return self._convert_to_dataframe(best_solution)
        
        # If GA fails, try AI-assisted repair
        if best_solution:
            repaired_solution = self._ai_repair_solution(best_solution)
            if self._is_solution_valid(repaired_solution):
                return self._convert_to_dataframe(repaired_solution)
        
        # If both fail, try constraint relaxation
        return self._generate_with_relaxed_constraints(section)

    def _initialize_population(self) -> List[List[Session]]:
        """Initialize random population of timetables"""
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            population = list(executor.map(
                lambda _: self._create_single_timetable(),
                range(self.population_size)
            ))
        return population

    def _create_single_timetable(self) -> List[Session]:
        """
        Create a single random timetable for the given population 
        """
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
        
        # Faculty conflicts
        faculty_slots = {}
        for session in solution:
            for faculty in session.faculty:
                key = (session.time_slot.day, session.time_slot.start_time)
                if key in faculty_slots:
                    penalties['faculty_conflict'] += 1
                faculty_slots[key] = faculty
        
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
            
            if len(subject_day_slots[key]) > 2:
                penalties['subject_distribution'] += 1
            
            # Check for more than 2 consecutive lectures
            sorted_slots = sorted(subject_day_slots[key])
            for i in range(len(sorted_slots)-2):
                if self._are_slots_consecutive(sorted_slots[i:i+3]):
                    penalties['consecutive_lectures'] += 1
        
        # Daily load balance
        day_loads = {}
        for session in solution:
            day = session.time_slot.day
            day_loads[day] = day_loads.get(day, 0) + session.time_slot.duration
        
        std_dev = np.std(list(day_loads.values()))
        penalties['daily_load_balance'] = std_dev
        
        # Apply weighted penalties
        for constraint, penalty in penalties.items():
            score -= penalty * CONSTRAINT_WEIGHTS[constraint]
        
        return max(0, score)

    def _get_random_time_slot(self, duration: float) -> TimeSlot:
        """Generate a random valid time slot"""
        day = random.choice(DAYS)
        valid_slots = [
            slot for slot in SLOTS 
            if slot < "12:20-13:00" and SLOTS.index(slot) + int(duration * 2) <= SLOTS.index("12:20-13:00") or
            slot > "13:00-13:40"
        ]
        start_time = random.choice(valid_slots)
        return TimeSlot(day=day, start_time=start_time, duration=duration)

    def _evolve_population(self, population: List[List[Session]]) -> List[List[Session]]:
        """Evolve population using genetic operators"""
        new_population = []
        
        # Elitism: Keep best solutions
        sorted_pop = sorted(population, key=lambda x: self._evaluate_fitness(x), reverse=True)
        elite_size = self.population_size // 10
        new_population.extend(copy.deepcopy(sorted_pop[:elite_size]))
        
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)
            
            # Crossover
            if random.random() < 0.8:
                child = self._crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)
            
            # Mutation
            if random.random() < 0.2:
                child = self._mutate(child)
            
            new_population.append(child)
            
        return new_population

    def _tournament_select(self, population: List[List[Session]], tournament_size: int = 3) -> List[Session]:
        """Select best solution from random tournament"""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: self._evaluate_fitness(x))

    def _crossover(self, parent1: List[Session], parent2: List[Session]) -> List[Session]:
        """Perform crossover between two parent solutions"""
        p1_sessions = self._group_by_subject(parent1)
        p2_sessions = self._group_by_subject(parent2)
        
        child_sessions = []
        subjects = list(p1_sessions.keys())
        crossover_point = random.randint(1, len(subjects) - 1)
        
        for subject in subjects[:crossover_point]:
            child_sessions.extend(p1_sessions[subject])
        
        for subject in subjects[crossover_point:]:
            child_sessions.extend(p2_sessions[subject])
        
        return child_sessions

    def _mutate(self, solution: List[Session]) -> List[Session]:
        """Apply mutation operators to solution"""
        mutated = copy.deepcopy(solution)
        mutations = [self._swap_time_slots, self._shift_session, self._swap_days]
        mutation = random.choice(mutations)
        return mutation(mutated)

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
            if next_slot == "12:20-13:00":
                return False
                
        return True

    def _are_slots_consecutive(self, slots: List[str]) -> bool:
        """Check if time slots are consecutive"""
        indices = [SLOTS.index(slot) for slot in slots]
        return all(indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1))

    def _is_solution_valid(self, solution: List[Session]) -> bool:
        """Check if solution meets all hard constraints"""
        if not solution:
            return False
            
        # Check faculty conflicts
        faculty_slots = {}
        for session in solution:
            for faculty in session.faculty:
                key = (session.time_slot.day, session.time_slot.start_time)
                if key in faculty_slots and faculty_slots[key] != faculty:
                    return False
                faculty_slots[key] = faculty
        
        # Check lab continuity
        for session in [s for s in solution if s.batch]:
            if not self._is_lab_continuous(session):
                return False
        
        return True

    def _convert_to_dataframe(self, solution: List[Session]) -> pd.DataFrame:
        """Convert solution to pandas DataFrame format"""
        df = pd.DataFrame(index=DAYS, columns=SLOTS)
        df.fillna("-", inplace=True)
        df["12:20-13:00"] = "LUNCH BREAK"
        
        for session in solution:
            start_idx = SLOTS.index(session.time_slot.start_time)
            duration_slots = int(session.time_slot.duration * 2)
            
            entry = self._format_session_entry(session)
            
            for i in range(duration_slots):
                if start_idx + i < len(SLOTS):
                    slot = SLOTS[start_idx + i]
                    if slot != "12:20-13:00":
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
    def _initialize_population(self) -> List[List[Session]]:
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            population = list(executor.map(create_timetable, [self] * self.population_size))
        return population

def main():
    st.set_page_config(page_title="Academic Timetable Generator", layout="wide")
    
    st.title("🎓 Academic Timetable Generator")
    
    # Sidebar configuration
    # Continue from previous main() function
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input("Gemini API Key", type="password")
        
        # Basic Settings
        population_size = st.slider("Population Size", 20, 200, 50)
        num_processes = st.slider("Number of Processes", 1, 8, 4)
        
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
                    name = st.text_input(
                        f"Name #{i+1}", 
                        value=st.session_state.subjects_data[i]["name"] if i < len(st.session_state.subjects_data) else f"Subject {i+1}"
                    )
                    weekly_hours = st.number_input(
                        f"Weekly Hours #{i+1}", 
                        1.0, 10.0, 
                        st.session_state.subjects_data[i]["weekly_hours"] if i < len(st.session_state.subjects_data) else 4.0
                    )
                
                with col2:
                    faculty = st.text_input(
                        f"Faculty #{i+1}", 
                        value=st.session_state.subjects_data[i]["faculty"][0] if i < len(st.session_state.subjects_data) else f"Prof. {i+1}"
                    )
                    is_lab = st.checkbox(
                        f"Is Lab #{i+1}", 
                        value=st.session_state.subjects_data[i]["is_lab"] if i < len(st.session_state.subjects_data) else False
                    )
                
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
    if not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar to continue.")
        return
    
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
            if st.button("Generate Timetable", type="primary"):
                try:
                    # Progress container
                    progress_container = st.empty()
                    progress_bar = progress_container.progress(0)
                    
                    # Create placeholder for fitness plot
                    plot_container = st.empty()
                    results_container = st.empty()
                    
                    def update_progress(current, total):
                        progress = int((current / total) * 100)
                        progress_bar.progress(progress)
                        
                        # Update fitness plot if there's history
                        if 'generator' in st.session_state and st.session_state.generator.fitness_history:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(st.session_state.generator.fitness_history, '-o')
                            ax.set_title('Timetable Evolution Progress')
                            ax.set_xlabel('Generation')
                            ax.set_ylabel('Best Fitness Score')
                            ax.grid(True)
                            plot_container.pyplot(fig)
                            plt.close()
                    
                    # Initialize generator
                    generator = TimetableGenerator(
                        api_key=api_key,
                        subjects=st.session_state.subjects_data,
                        population_size=population_size,
                        num_processes=num_processes
                    )
                    generator.set_progress_callback(update_progress)
                    st.session_state.generator = generator
                    
                    # Generate timetable
                    timetable = generator.generate_timetable(section)
                    st.session_state.timetable = timetable
                    
                    # Clear progress indicators
                    progress_container.empty()
                    
                    # Display results
                    results_container.success("Timetable generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating timetable: {e}")
        
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
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(st.session_state.generator.get_fitness_history(), '-o')
                ax.set_title('Fitness Evolution')
                ax.set_xlabel('Generation')
                ax.set_ylabel('Fitness Score')
                ax.grid(True)
                st.pyplot(fig)
            
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
                              if shortcut in str(st.session_state.timetable.at[day, slot]))
                    subject_counts.at[subject["name"], day] = count
            
            st.dataframe(subject_counts, use_container_width=True)
    
    with tab3:
        if 'timetable' in st.session_state:
            st.header("Export Options")
            
            col1, col2, col3 = st.columns(3)
            
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
            
            with col3:
                # Export analysis as PDF
                if st.button("Generate Report"):
                    st.info("Report generation feature coming soon!")

if __name__ == "__main__":
    main()