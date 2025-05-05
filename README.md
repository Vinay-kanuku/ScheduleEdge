# 🗓️ Automatic Timetable Generator

An intelligent timetable generation system designed to automatically create optimized weekly class schedules for different year groups, with a focus on minimizing gaps and respecting fixed constraints like lab days.

## 🚀 Features

- Auto-generates timetables for multiple student years.
 
- Optimized scheduling to **minimize gaps** and compress idle time.
- Modular design for easy updates and future constraints (e.g., faculty availability, placement training days).

## 🧠 Core Logic

1. **Lab Sessions Pre-Allocation**  
   Labs are scheduled in advance on fixed days (Tue & Wed) for 2nd and 3rd years.

2. **Theory Class Allocation**  
   Remaining slots are filled with theory classes using a constraint-solving approach.

3. **Gap Minimization**  
   Timetables are optimized to reduce gaps between sessions, especially on non-lab days.

## ⚙️ How to Use

1. Clone the repo.
``` markdown 
git clone https://github.com/Vinay-kanuku/ScheduleEdge.git

```
2. Set up the environment (requirements.txt or conda env).
3. Add your input data (subjects, slots, years, faculty, etc.).
4. Run the timetable generation script.
5. Review the output timetables (console or exported file).

 