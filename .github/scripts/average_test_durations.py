# Description: This script averages the test durations from all the artifacts
import glob
import json
import subprocess
from collections import defaultdict
import os

# Function to collect test names using pytest
def collect_tests_with_pytest():
    # Run pytest with --collect-only to get the list of test cases
    result = subprocess.run(
        ["pytest", "--collect-only"],
        capture_output=True,
        text=True
    )

    collected_tests = defaultdict(list)

    # Parse the output to organize tests under their respective modules
    for line in result.stdout.splitlines():
        if line.startswith("tests/"):
            collected_tests[line] = 0

    return collected_tests


# Consolidate durations from existing artifacts
def consolidate_durations():
    durations = defaultdict(lambda: {'total_duration': 0, 'count': 0})

    # Iterate over all downloaded duration artifacts
    for folder in glob.glob("test-durations-*"):
        # The path to the duration file in each directory
        duration_file_path = os.path.join(folder, ".pytest-split-durations")

        if os.path.isfile(duration_file_path):
            with open(duration_file_path, "r") as f:
                data = json.load(f)
                for test, duration in data.items():
                    durations[test]['total_duration'] += duration
                    durations[test]['count'] += 1

    # Calculate the average duration for each test
    return {test: info['total_duration'] / info['count'] for test, info in durations.items()}

# Define the path to the consolidated durations file
CONSOLIDATED_FILE = "tests/test_data/.pytest-split-durations"


# Main script logic
def main():
    # Collect tests grouped by modules using pytest
    collected_tests = collect_tests_with_pytest()

    # Consolidate durations from artifacts
    consolidated_durations = consolidate_durations()

    
    # Merge and update with consolidated durations
    updated_durations = {}
    for test, duration in collected_tests.items():
        # Update average test durations and exclude test which not exists (can happen on renaming or removing tests) 
        if test in consolidated_durations:
            updated_durations[test] = consolidated_durations[test]
        
    # Load the existing durations file if it exists
    existing_durations = {}
    if os.path.isfile(CONSOLIDATED_FILE):
        with open(CONSOLIDATED_FILE, "r") as f:
            existing_durations = json.load(f)

    
    # Sort the keys to compare the tests in both dictionaries
    updated_durations_key = sorted(updated_durations.keys())
    existing_durations_key = sorted(existing_durations.keys())


    # Check if all keys in updated_durations are in existing_durations
    if updated_durations_key == existing_durations_key:
        print("No new tests detected; durations file remains unchanged.")
    else:
        # Write the updated durations to the consolidated file
        with open(CONSOLIDATED_FILE, "w") as f:
            json.dump(updated_durations, f, indent=4)
        print("New tests detected; updated the durations file.")

if __name__ == "__main__":
    main()
