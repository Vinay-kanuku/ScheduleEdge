import pandas as pd
import random        

class TimetableGenerator:
    """Generates timetables based on input data blocks"""
    
    def __init__(self, blocks):
        """Initialize with data blocks"""
        self.blocks = blocks
        self.days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        self.periods = ['9:00-9:50', '9:50-10:40', '10:40-11:30', '11:30-12:20', '1:00-1:55', '1:55-2:50', '2:50-3:45']
        self.slots = [(d, p) for d in self.days for p in self.periods]
        self.blackout_days = {'II': 'Wednesday', 'III': 'Friday', 'IV': 'Monday'}
    
    def is_blackout(self, sec, idx):
        """Check if a given slot is in a blackout day for the section"""
        year = sec.split('-')[0]
        day = self.slots[idx][0]
        return self.blackout_days.get(year) == day
    
    def book_slot(self, timetables, sec, label, slot_idxs):
        """Book a set of slots in the timetable"""
        for idx in slot_idxs:
            day, per = self.slots[idx]
            timetables[sec].at[day, per] = label  
    
    def generate(self):
        """Generate timetables for all sections and faculty"""
        timetables = {sec: pd.DataFrame(index=self.days, columns=self.periods) for sec in self.blocks}
        faculty_timetables = {}
        occupied = {}
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
        
        # Step 2: Allocate labs with A1/A2 batches in rotational pairings
        for sec, course_list in courses_by_section.items():
            lab_courses = [c for c in course_list if c['is_lab']]
            theory_courses = [c for c in course_list if not c['is_lab']]
            
            if len(lab_courses) >= 3:
                a1_labs = [lab_courses[0], lab_courses[1], lab_courses[2]]
                a2_labs = [lab_courses[1], lab_courses[2], lab_courses[0]]
                pair_blocks = list(zip(a1_labs, a2_labs))
                
                block_found = 0
                for idx in range(len(self.slots) - 2):
                    if block_found >= len(pair_blocks):
                        break
                    block = [idx, idx+1, idx+2]
                    block_periods = [self.slots[i][1] for i in block]
                    day = self.slots[block[0]][0]
                    
                    if day in section_lab_days[sec]:
                        continue
                    
                    if (block[0] % 7) <= 4:  # Only allow full morning or full afternoon
                        morning = all(p in self.periods[:4] for p in block_periods)
                        afternoon = all(p in self.periods[4:] for p in block_periods)
                        if not (morning or afternoon):
                            continue
                        
                        if any(self.is_blackout(sec, i) for i in block):
                            continue
                        
                        if any(i in lab_occupied for i in block):
                            continue
                        
                        if all(pd.isna(timetables[sec].at[day, per]) for per in block_periods):
                            # Assign Batch 1 and Batch 2 labs in one single slot (same period)
                            lab1 = pair_blocks[block_found][0]
                            lab2 = pair_blocks[block_found][1]
                            self.book_slot(timetables, sec, 
                                          f"{lab1['Subject Name']} (Batch 1 - A1)", block)  # Batch 1 Lab
                            self.book_slot(timetables, sec, 
                                          f"{lab2['Subject Name']} (Batch 2 - A2)", block)  # Batch 2 Lab
                            lab_occupied.update(block)
                            section_lab_days[sec].add(day)
                            block_found += 1
            
            # Step 3: Assign theory classes
            for course in theory_courses:
                count = random.choice([5, 6])
                faculty = course['Faculty Name']
                available = [i for i in range(len(self.slots))
                             if i not in occupied.get(faculty, set()) and
                             not self.is_blackout(sec, i) and
                             pd.isna(timetables[sec].at[self.slots[i][0], self.slots[i][1]])]
                random.shuffle(available)
                for idx in available[:count]:
                    day, per = self.slots[idx]
                    timetables[sec].at[day, per] = course['Subject Name']
                    occupied.setdefault(faculty, set()).add(idx)
        
        # Create Faculty Timetable with Lab info
        for sec, df_block in self.blocks.items():
            for _, row in df_block.iterrows():
                if pd.isna(row['Faculty Name']):
                    continue
                faculty = row['Faculty Name']
                if faculty not in faculty_timetables:
                    faculty_timetables[faculty] = pd.DataFrame(index=self.days, columns=self.periods)
                
                is_lab = 'Lab' in str(row['Subject Name'])
                subject_name = row['Subject Name']
                label = f"{subject_name} (Lab)" if is_lab else subject_name
                for idx in range(len(self.slots)):
                    day, per = self.slots[idx]
                    cell_value = timetables[sec].at[day, per]
                    if isinstance(cell_value, str) and subject_name in cell_value:
                        faculty_timetables[faculty].at[day, per] = label
        
        return timetables, faculty_timetables