import numpy as np
from unittest.mock import MagicMock

# Minimal data structures to reproduce the bug
JOBS_DELIVERED_DEBUG = []

def run_debug_bugged():
    """
    This version of the function contains the bug.
    It will raise a ZeroDivisionError if JOBS_DELIVERED_DEBUG is empty.
    """
    units = -1
    if len(JOBS_DELIVERED_DEBUG) > 0:
        units = len(JOBS_DELIVERED_DEBUG)

    # The following line will raise a ZeroDivisionError
    gtt = sum(job.get_GTT() for job in JOBS_DELIVERED_DEBUG) / units
    print(f"Bugged function calculated GTT: {gtt}")

def run_debug_fixed():
    """
    This version of the function contains the fix.
    It checks if units > 0 before performing the division.
    """
    units = 0
    if len(JOBS_DELIVERED_DEBUG) > 0:
        units = len(JOBS_DELIVERED_DEBUG)

    gtt = 0.0
    if units > 0:
        gtt = sum(job.get_GTT() for job in JOBS_DELIVERED_DEBUG) / units

    print(f"Fixed function calculated GTT: {gtt}")

if __name__ == "__main__":
    print("--- Running bugged function ---")
    try:
        run_debug_bugged()
    except ZeroDivisionError as e:
        print(f"Successfully caught expected error: {e}")

    print("\n--- Running fixed function ---")
    run_debug_fixed()

    # Now, let's test with some data
    job1 = MagicMock()
    job1.get_GTT.return_value = 10
    job2 = MagicMock()
    job2.get_GTT.return_value = 20
    JOBS_DELIVERED_DEBUG = [job1, job2]

    print("\n--- Rerunning with data ---")
    run_debug_fixed()