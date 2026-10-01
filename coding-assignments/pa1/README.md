Due: 10/12/2026 before the lesson (based on pull request timestamp).

# Goals: 
- Students get familiar with Python programming, including read/write files, functions, arithmetic manipulations and JSON format.
- Students can practice basic model inference and evaluation flow, including GT labeling, defining metrics and implementation.

# Instructions
1. Copy the whole coding-assignments/pa1 directory.
2. Run 'pip install easyocr' in your Python environment to install the OCR package.
3. Try to run sample.py to see if your environment is ready!
4. Please review coding-assignments/pa1/input/cases.json, where you will find a JSON file listing cases and its groundtruth.
5. Please review coding-assignments/pa1/output.json, it shows a sample output for your program. (your output should be different from it. this is just a sample)
6. The goal for this assignment is to implement a complete program:
- can be run as python hw2.py input/cases.json output.json
- iterate through all cases mentioned in input/cases.json, run OCR and collect text
- compare results with groundtruth, define your own metrics.
- write out OCR results and metric numbers in output.json 

# Requirements:
1. Please turn in your pa1.py and your expected output.json.
2. Please submit a PR to your own repository with Title [PA1], this PR should include the final version of your own output.json and pa2.py and invite your instructor as the reviewer. The timestamp for this PR to be published is the submission time.
3. Tip: you can always create a homework branch, and keep all the commit history on that homework branch until your PR.

# Rubrics:
1. - (100%) Finish all requirements on time.
2. - (90%) Finish all requirements , but submitted late.
3. The final score is computed by (0.9 or 1.0) x (Requirement points).
4. Requirements and partial points
- Python file can parse cases.json properly (20%)
- Python file can iterate on all examples in cases.json (30%)
- Python file can create or overwrite output.json (10%)
- Python file can generate "results" section correctly (20%)
- Python file can generate "metrics" section with at least **TWO** metric values correctly (20%)

# References
1. https://github.com/JaidedAI/EasyOCR

