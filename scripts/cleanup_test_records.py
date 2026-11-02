"""
LPU Placement Intelligence Platform - Assessment History Database Sanitizer
Usage:
    python scripts/cleanup_test_records.py --dry-run
    python scripts/cleanup_test_records.py --apply
"""

import os
import sys
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.history_store import AssessmentHistoryStore

def main():
    parser = argparse.ArgumentParser(description="Purge automated test records from assessment_history.db safely.")
    parser.add_argument("--apply", action="store_true", help="Execute deletion of test artifacts. Default is dry-run.")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Scan and report matching test records without deleting.")
    args = parser.parse_args()

    is_dry_run = not args.apply

    store = AssessmentHistoryStore()
    print("=" * 68)
    print(" LPU Placement Intelligence: Database Test Record Sanitizer")
    print(f" Target Database : {store.db_path}")
    print(f" Mode            : {'DRY RUN (Scan Only)' if is_dry_run else 'APPLY (Purge Records)'}")
    print("=" * 68)

    result = store.delete_test_records(dry_run=is_dry_run)
    count = result["matched_count"]
    ids = result["matched_ids"]

    print(f"\nFound {count} test run artifact record(s) matching unit test patterns:")
    if ids:
        print(f"  Record IDs: {ids[:25]}{'...' if len(ids) > 25 else ''}")
        
    if is_dry_run:
        print("\n[DRY RUN COMPLETE] No records were deleted. To permanently delete test records, run:")
        print("  python scripts/cleanup_test_records.py --apply")
    else:
        # Flag demo records
        conn = store._get_connection()
        try:
            with conn:
                conn.execute("UPDATE assessment_history SET is_demo = 1 WHERE student_id LIKE 'LPU_DEMO_%' OR student_id LIKE 'DEMO_%'")
                conn.execute("UPDATE assessment_history SET readiness_category = 'Strong preparation' WHERE readiness_band = 'Strong preparation' OR readiness_tier IN ('Placement Ready', 'Advanced')")
                conn.execute("UPDATE assessment_history SET readiness_category = 'Developing' WHERE readiness_band = 'Developing' OR readiness_tier = 'Developing'")
                conn.execute("UPDATE assessment_history SET readiness_category = 'Needs attention' WHERE readiness_band = 'Needs attention' OR readiness_tier = 'Foundation'")
        finally:
            conn.close()
        print(f"\n[PURGE COMPLETE] Successfully removed {count} test record(s) and flagged demo profiles.")

if __name__ == "__main__":
    main()
