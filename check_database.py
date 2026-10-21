
import sqlite3

with sqlite3.connect("data/routine.db") as connection:
    rows = connection.execute(
        """
        SELECT start_time, end_time, app_name, state, duration_seconds
        FROM screen_activity
        WHERE app_name LIKE ?
        ORDER BY id DESC
        LIMIT 10
        """,
        ("%Notepad%",),
    ).fetchall()

    print("Latest Notepad activity records:")
    if rows:
        for row in rows:
            print(row)
    else:
        print("No Notepad records found.")
