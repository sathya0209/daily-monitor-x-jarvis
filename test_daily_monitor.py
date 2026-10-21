
import sqlite3
from datetime import date

DB_PATH = "data/routine.db"
TODAY = date.today().isoformat()

with sqlite3.connect(DB_PATH) as conn:
    conn.row_factory = sqlite3.Row

    print(f"\nDAILY MONITOR TEST — {TODAY}")
    print("=" * 45)

    # Screen activity totals
    rows = conn.execute(
        """
        SELECT state, COUNT(*) AS records,
               COALESCE(SUM(duration_seconds), 0) AS seconds
        FROM screen_activity
        WHERE substr(start_time, 1, 10) = ?
        GROUP BY state
        ORDER BY state
        """,
        (TODAY,),
    ).fetchall()

    print("\n1. Activity by state")
    if rows:
        for row in rows:
            print(
                f"{row['state']}: {row['records']} records, "
                f"{row['seconds']} seconds "
                f"({row['seconds'] / 60:.1f} minutes)"
            )
    else:
        print("No activity records found for today.")

    # Application usage totals
    apps = conn.execute(
        """
        SELECT app_name, state,
               COUNT(*) AS records,
               SUM(duration_seconds) AS seconds
        FROM screen_activity
        WHERE substr(start_time, 1, 10) = ?
        GROUP BY app_name, state
        ORDER BY seconds DESC
        """,
        (TODAY,),
    ).fetchall()

    print("\n2. Usage by application")
    if apps:
        for row in apps:
            print(
                f"{row['app_name']} [{row['state']}]: "
                f"{row['seconds'] or 0} seconds "
                f"across {row['records']} records"
            )
    else:
        print("No application usage found.")

    # Tasks recorded today
    tasks = conn.execute(
        """
        SELECT task, category, minutes, completed, entry_date
        FROM tasks
        WHERE entry_date = ?
        """,
        (TODAY,),
    ).fetchall()

    print("\n3. Tasks recorded today")
    if tasks:
        for task in tasks:
            status = "Completed" if task["completed"] else "Incomplete"
            print(
                f"{task['task']} | {task['category']} | "
                f"{task['minutes']} minutes | {status}"
            )
    else:
        print("No tasks recorded today.")

    print("\nEnd of read-only test.")
