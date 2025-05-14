import pandas as pd
from collections import defaultdict

class TimetableValidator:
    """Validates timetables against hard and soft constraints"""

    def __init__(self, timetables, faculty_timetables, blocks):
        """Initialize with timetables and configuration"""
        self.timetables = timetables
        self.faculty_timetables = faculty_timetables
        self.blocks = blocks
        self.days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        self.periods = ['9:00-9:50', '9:50-10:40', '10:40-11:30', '11:30-12:20', 
                        '1:00-1:55', '1:55-2:50', '2:50-3:45']
        self.blackout_days = {'II': 'Wednesday', 'III': 'Friday', 'IV': 'Monday'}
        self.violations = defaultdict(int)
        
    def calculate_faculty_workload(self):
        """Calculate workload per faculty"""
        workload = defaultdict(int)
        
        for faculty, timetable in self.faculty_timetables.items():
            for day in self.days:
                for period in self.periods:
                    if not pd.isna(timetable.at[day, period]):
                        workload[faculty] += 1
        
        return workload
    
    def check_blackout_days(self):
        """Check if classes are scheduled on blackout days"""
        for sec, timetable in self.timetables.items():
            year = sec.split('-')[0]
            blackout_day = self.blackout_days.get(year)
            
            if blackout_day:
                day_schedule = timetable.loc[blackout_day]
                for period in self.periods:
                    if not pd.isna(day_schedule[period]):
                        self.violations["blackout_day"] += 1
    
    def check_consecutive_classes(self):
        """Check if multi-hour subjects have consecutive periods"""
        for sec, timetable in self.timetables.items():
            for day in self.days:
                # Track subjects for this day
                subjects_count = defaultdict(list)
                
                # Collect periods for each subject
                for i, period in enumerate(self.periods):
                    cell = timetable.at[day, period]
                    if isinstance(cell, str):
                        subject = cell.split("(")[0].strip()
                        subjects_count[subject].append(i)
                
                # Check if multi-hour subjects have consecutive periods
                for subject, periods in subjects_count.items():
                    if len(periods) > 1:
                        # Sort periods
                        periods.sort()
                        
                        # Check for non-consecutive periods
                        for i in range(len(periods) - 1):
                            # Allow jump at lunch break (3 to 4)
                            if periods[i] == 3 and periods[i+1] == 4:
                                continue
                                
                            if periods[i+1] - periods[i] > 1:
                                self.violations["non_consecutive"] += 1
    
    def check_faculty_conflicts(self):
        """Check for faculty being scheduled in multiple places at once"""
        for faculty, timetable in self.faculty_timetables.items():
            for day in self.days:
                # Get unique classes for each period (considering batch teaching)
                for period in self.periods:
                    cell = timetable.at[day, period]
                    
                    if isinstance(cell, str) and "|" in cell:  # Faculty teaching multiple classes
                        classes = cell.split("|")
                        if len(set(classes)) > 1:  # More than one unique class
                            self.violations["faculty_conflict"] += 1
    
    def check_lab_schedules(self):
        """Check if labs are properly scheduled in 3-hour blocks"""
        for sec, timetable in self.timetables.items():
            for day in self.days:
                lab_periods = []
                
                for i, period in enumerate(self.periods):
                    cell = timetable.at[day, period]
                    if isinstance(cell, str) and "Lab" in cell:
                        lab_periods.append(i)
                
                # Labs should be in blocks of 3
                if lab_periods and len(lab_periods) % 3 != 0:
                    self.violations["incomplete_lab"] += 1
                    
                # Check if lab periods are consecutive
                if lab_periods:
                    lab_periods.sort()
                    for i in range(len(lab_periods) - 1):
                        # Allow jump at lunch break
                        if lab_periods[i] == 3 and lab_periods[i+1] == 4:
                            continue
                            
                        if lab_periods[i+1] - lab_periods[i] > 1:
                            self.violations["non_consecutive_lab"] += 1
    
    def validate(self):
        """Run all validation checks and return summary"""
        self.violations.clear()
        
        self.check_blackout_days()
        self.check_consecutive_classes()
        self.check_faculty_conflicts()
        self.check_lab_schedules()
        
        # Calculate workload balance
        workload = self.calculate_faculty_workload()
        workload_std = 0
        if workload:
            values = list(workload.values())
            mean = sum(values) / len(values) if values else 0
            workload_std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5 if values else 0
        
        # Calculate overall quality score (lower is better)
        score = sum(self.violations.values()) + workload_std
        
        return {
            "violations": dict(self.violations),
            "workload_balance": {
                "std_dev": workload_std,
                "min": min(workload.values()) if workload else 0,
                "max": max(workload.values()) if workload else 0,
                "avg": sum(workload.values()) / len(workload) if workload else 0
            },
            "score": score
        }