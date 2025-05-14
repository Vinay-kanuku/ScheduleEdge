import pandas as pd
import random
import itertools
from collections import defaultdict

class TimetableGenerator:
    """Generates timetables based on input data blocks with hard constraint enforcement"""
    
    def __init__(self, blocks):
        """Initialize with data blocks"""
        self.blocks = blocks
        self.days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        self.periods = ['9:00-9:50', '9:50-10:40', '10:40-11:30', '11:30-12:20', '1:00-1:55', '1:55-2:50', '2:50-3:45']
        self.slots = [(d, p) for d in self.days for p in self.periods]
        
        # Map periods to indices for easy lookup
        self.period_indices = {p: i for i, p in enumerate(self.periods)}
        
        # Define blackout days (days when certain years shouldn't have classes)
        self.blackout_days = {'II': 'Wednesday', 'III': 'Friday', 'IV': 'Monday'}
        
        # Track faculty workload for balance
        self.faculty_workload = defaultdict(int)
        
        # Set max hours per subject per day
        self.max_hours_per_subject_per_day = 2
        
        # Track subject hours per day
        self.subject_hours_per_day = defaultdict(lambda: defaultdict(int))
        
        # Validation counters and scoring
        self.constraint_violations = defaultdict(int)
        self.timetable_score = 0
    
    def is_blackout(self, sec, day):
        """Check if a given day is a blackout day for the section"""
        year = sec.split('-')[0]
        return self.blackout_days.get(year) == day
    
    def is_consecutive_slot(self, idx1, idx2):
        """Check if two slot indices represent consecutive periods on the same day"""
        day1, period1 = self.slots[idx1]
        day2, period2 = self.slots[idx2]
        
        if day1 != day2:
            return False
        
        p1_idx = self.period_indices[period1]
        p2_idx = self.period_indices[period2]
        
        # Check if periods are consecutive (account for lunch break)
        if p1_idx == 3 and p2_idx == 4:  # Before and after lunch
            return False
        
        return abs(p1_idx - p2_idx) == 1
    
    def find_consecutive_slots(self, sec, faculty, needed, timetables):
        """Find available consecutive slots for multi-hour classes"""
        available_slots = []
        
        for idx in range(len(self.slots) - 1):  # -1 because we need at least 2 consecutive slots
            day, period = self.slots[idx]
            
            # Skip if blackout day or slot already occupied
            if self.is_blackout(sec, day) or not pd.isna(timetables[sec].at[day, period]):
                continue
            
            # For 2-hour blocks:
            if needed == 2:
                # Check if next slot is available and consecutive
                if idx + 1 < len(self.slots):
                    next_day, next_period = self.slots[idx + 1]
                    
                    if day == next_day and self.is_consecutive_slot(idx, idx + 1):
                        # Check if next slot is free
                        if pd.isna(timetables[sec].at[next_day, next_period]):
                            # Check if faculty is available for both slots
                            if (faculty not in self.faculty_timetables or 
                                (pd.isna(self.faculty_timetables[faculty].at[day, period]) and 
                                 pd.isna(self.faculty_timetables[faculty].at[next_day, next_period]))):
                                available_slots.append([idx, idx + 1])
            
            # For 3-hour blocks (labs):
            elif needed == 3:
                if idx + 2 < len(self.slots):
                    next_day1, next_period1 = self.slots[idx + 1]
                    next_day2, next_period2 = self.slots[idx + 2]
                    
                    # All in same day and consecutive
                    if (day == next_day1 == next_day2 and 
                        self.is_consecutive_slot(idx, idx + 1) and 
                        self.is_consecutive_slot(idx + 1, idx + 2)):
                        
                        # Check if all slots are free
                        if (pd.isna(timetables[sec].at[next_day1, next_period1]) and 
                            pd.isna(timetables[sec].at[next_day2, next_period2])):
                            
                            # Check faculty availability for all three slots
                            if (faculty not in self.faculty_timetables or 
                                (pd.isna(self.faculty_timetables[faculty].at[day, period]) and 
                                 pd.isna(self.faculty_timetables[faculty].at[next_day1, next_period1]) and 
                                 pd.isna(self.faculty_timetables[faculty].at[next_day2, next_period2]))):
                                available_slots.append([idx, idx + 1, idx + 2])
        
        return available_slots
    
    def book_slot(self, timetables, sec, label, slot_idxs, faculty=None):
        """Book a set of slots in the timetable"""
        for idx in slot_idxs:
            day, per = self.slots[idx]
            timetables[sec].at[day, per] = label
            
            # Track subject hours per day
            subject_key = label.split("(")[0].strip()  # Extract base subject name
            self.subject_hours_per_day[sec][(day, subject_key)] += 1
            
            # Update faculty timetable if provided
            if faculty:
                if faculty not in self.faculty_timetables:
                    self.faculty_timetables[faculty] = pd.DataFrame(index=self.days, columns=self.periods)
                self.faculty_timetables[faculty].at[day, per] = f"{label} ({sec})"
                self.faculty_workload[faculty] += 1
    
    def validate_timetable(self, timetables):
        """Validate the generated timetables against hard constraints"""
        self.constraint_violations.clear()
        
        for sec, timetable in timetables.items():
            # Check for blackout days
            year = sec.split('-')[0]
            blackout_day = self.blackout_days.get(year)
            if blackout_day:
                if not timetable.loc[blackout_day].isna().all():
                    self.constraint_violations["blackout_days"] += 1
            
            # Check for non-consecutive multi-hour subjects
            for day in self.days:
                day_schedule = timetable.loc[day]
                subjects_seen = {}
                
                for period_idx, period in enumerate(self.periods):
                    cell = day_schedule[period]
                    if isinstance(cell, str):
                        base_subject = cell.split("(")[0].strip()
                        
                        if base_subject in subjects_seen:
                            prev_period_idx = subjects_seen[base_subject]
                            
                            # If subject hours > 1 per day and not consecutive, flag it
                            if self.subject_hours_per_day[sec][(day, base_subject)] > 1:
                                if abs(period_idx - prev_period_idx) != 1:
                                    self.constraint_violations["non_consecutive"] += 1
                        
                        subjects_seen[base_subject] = period_idx
        
        # Check faculty conflicts
        faculty_conflicts = 0
        for faculty, timetable in self.faculty_timetables.items():
            for day in self.days:
                for period in self.periods:
                    class_count = 0
                    cell = timetable.at[day, period]
                    if isinstance(cell, str) and len(cell.strip()) > 0:
                        class_count += 1
                    
                    if class_count > 1:
                        faculty_conflicts += 1
                        self.constraint_violations["faculty_conflicts"] += 1
        
        # Calculate score (lower is better)
        self.timetable_score = sum(self.constraint_violations.values())
        
        return self.constraint_violations, self.timetable_score
    
    def generate(self, max_attempts=10):
        """Generate timetables for all sections and faculty with constraint enforcement"""
        best_score = float('inf')
        best_timetables = None
        best_faculty_timetables = None
        
        for attempt in range(max_attempts):
            # Reset data structures for this attempt
            self.faculty_timetables = {}
            self.faculty_workload.clear()
            self.subject_hours_per_day.clear()
            
            # Initialize timetables
            timetables = {sec: pd.DataFrame(index=self.days, columns=self.periods) for sec in self.blocks}
            lab_occupied = set()
            section_lab_days = {sec: set() for sec in self.blocks}
            
            # Step 1: Prepare course list
            courses_by_section = {}
            for sec, df_block in self.blocks.items():
                courses = []
                for _, row in df_block.iterrows():
                    if pd.isna(row['Faculty Name']):
                        continue
                    is_lab = 'Lab' in str(row['Subject Name'])
                    courses.append({
                        'Subject Name': row['Subject Name'],
                        'Faculty Name': row['Faculty Name'],
                        'is_lab': is_lab
                    })
                courses_by_section[sec] = courses
            
            # Step 2: First allocate labs (they need consecutive 3-hour blocks)
            for sec, course_list in courses_by_section.items():
                lab_courses = [c for c in course_list if c['is_lab']]
                
                if len(lab_courses) >= 3:
                    # Create lab pairings (A1/A2 batches)
                    a1_labs = [lab_courses[0], lab_courses[1], lab_courses[2]]
                    a2_labs = [lab_courses[1], lab_courses[2], lab_courses[0]]
                    pair_blocks = list(zip(a1_labs, a2_labs))
                    
                    block_found = 0
                    for day in self.days:
                        if block_found >= len(pair_blocks) or day in section_lab_days[sec]:
                            continue
                            
                        if self.is_blackout(sec, day):
                            continue
                        
                        # Try to find morning (first 3 periods) or afternoon (last 3 periods) blocks
                        for period_start_idx in [0, 4]:  # Morning or afternoon
                            if period_start_idx + 2 >= len(self.periods):
                                continue
                                
                            slots = []
                            for i in range(3):  # 3-hour block
                                day_period = (day, self.periods[period_start_idx + i])
                                slot_idx = self.slots.index(day_period)
                                slots.append(slot_idx)
                            
                            # Check if all slots are available
                            if all(pd.isna(timetables[sec].at[day, self.periods[period_start_idx + i]]) for i in range(3)):
                                # Check if slots not already used for labs
                                if not any(slot in lab_occupied for slot in slots):
                                    # Assign Batch 1 and Batch 2 labs in parallel
                                    lab1 = pair_blocks[block_found][0]
                                    lab2 = pair_blocks[block_found][1]
                                    
                                    # Book slots for both batches
                                    self.book_slot(timetables, sec, 
                                                  f"{lab1['Subject Name']} (Batch 1 - A1)", 
                                                  slots, lab1['Faculty Name'])
                                    
                                    self.book_slot(timetables, sec, 
                                                  f"{lab2['Subject Name']} (Batch 2 - A2)", 
                                                  slots, lab2['Faculty Name'])
                                    
                                    # Mark slots as occupied for labs
                                    lab_occupied.update(slots)
                                    section_lab_days[sec].add(day)
                                    block_found += 1
                                    break
            
            # Step 3: Assign theory classes (give consecutive slots for multiple hours)
            for sec, course_list in courses_by_section.items():
                theory_courses = [c for c in course_list if not c['is_lab']]
                
                for course in theory_courses:
                    faculty = course['Faculty Name']
                    subject_name = course['Subject Name']
                    
                    # Determine how many hours this subject needs
                    weekly_hours = random.choice([5, 6])
                    remaining_hours = weekly_hours
                    
                    # First, allocate 2-hour blocks (try to have 2-3 of these)
                    two_hour_blocks = min(3, remaining_hours // 2)
                    
                    for _ in range(two_hour_blocks):
                        # Find consecutive slots
                        available_consecutive = self.find_consecutive_slots(sec, faculty, 2, timetables)
                        
                        if available_consecutive:
                            # Pick a random consecutive slot pair
                            slots = random.choice(available_consecutive)
                            self.book_slot(timetables, sec, subject_name, slots, faculty)
                            remaining_hours -= 2
                    
                    # Allocate remaining single hours
                    while remaining_hours > 0:
                        # Find available single slots
                        available = []
                        
                        for idx, (day, period) in enumerate(self.slots):
                            if self.is_blackout(sec, day):
                                continue
                                
                            # Skip if slot already occupied
                            if not pd.isna(timetables[sec].at[day, period]):
                                continue
                                
                            # Skip if faculty not available
                            if faculty in self.faculty_timetables and not pd.isna(self.faculty_timetables[faculty].at[day, period]):
                                continue
                                
                            # Check subject hours limit per day
                            if self.subject_hours_per_day[sec][(day, subject_name)] >= self.max_hours_per_subject_per_day:
                                continue
                                
                            available.append(idx)
                        
                        if available:
                            # Book a single slot
                            slot = [random.choice(available)]
                            self.book_slot(timetables, sec, subject_name, slot, faculty)
                            remaining_hours -= 1
                        else:
                            # No more slots available
                            break
            
            # Validate and score the timetable
            violations, score = self.validate_timetable(timetables)
            
            # Keep the best timetable
            if score < best_score:
                best_score = score
                best_timetables = {sec: df.copy() for sec, df in timetables.items()}
                best_faculty_timetables = {faculty: df.copy() for faculty, df in self.faculty_timetables.items()}
            
            # If no violations, we're done
            if score == 0:
                break
        
        # Return the best timetable found
        validation_summary = {
            "score": best_score,
            "violations": self.constraint_violations,
            "attempts": attempt + 1
        }
        
        return best_timetables, best_faculty_timetables, validation_summary