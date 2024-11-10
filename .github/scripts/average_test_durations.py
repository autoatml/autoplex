# .github/scripts/average_test_durations.py
import glob
import json
from collections import defaultdict

# Define the path to the consolidated durations file
consolidated_file = "tests/test_data/.pytest-split-durations"

# Dictionary to store total duration and count for each test
durations = defaultdict(lambda: {'total_duration': 0, 'count': 0})

# Iterate over all downloaded duration artifacts
for file in glob.glob("test-durations-*"):
    with open(file, "r") as f:
        data = json.load(f)
        for test, duration in data.items():
            durations[test]['total_duration'] += duration
            durations[test]['count'] += 1

# Calculate the average duration for each test
averaged_durations = {test: info['total_duration'] / info['count'] for test, info in durations.items()}

# Write the averaged durations to the consolidated file
with open(consolidated_file, "w") as f:
    json.dump(averaged_durations, f, indent=4)