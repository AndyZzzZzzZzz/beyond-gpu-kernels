import sqlite3
import os
import random
import string

DB_PATH = "workloads/single_step/mock_db_payload.db"
TOTAL_ROWS = 5_000_000  # Generates roughly ~400-500MB depending on string length
BATCH_SIZE = 100_000

def generate_random_log(is_critical=False):
    """Generates a padded log string, occasionally inserting our target keyword."""
    base = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=80))
    if is_critical:
        return f"[TIMESTAMP] {base[:20]} CRITICAL ERROR {base[20:]}"
    return f"[TIMESTAMP] {base}"

if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    print(f"Building massive SQLite database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Extreme optimizations for mass insertion
    cursor.execute("PRAGMA synchronous = OFF;")
    cursor.execute("PRAGMA journal_mode = MEMORY;")
    cursor.execute("PRAGMA temp_store = MEMORY;")

    cursor.execute("""
        CREATE TABLE server_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_level TEXT,
            log_message TEXT
        )
    """)

    levels = ["INFO", "DEBUG", "WARN"]
    
    print(f"Inserting {TOTAL_ROWS:,} rows in batches of {BATCH_SIZE:,}...")
    for batch_idx in range(TOTAL_ROWS // BATCH_SIZE):
        batch_data = []
        for _ in range(BATCH_SIZE):
            # Give a ~1% chance to insert a CRITICAL ERROR for the C++ tool to find
            is_critical = random.random() < 0.01
            level = "FATAL" if is_critical else random.choice(levels)
            msg = generate_random_log(is_critical)
            batch_data.append((level, msg))
            
        cursor.executemany("INSERT INTO server_logs (log_level, log_message) VALUES (?, ?)", batch_data)
        conn.commit()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  -> {(batch_idx + 1) * BATCH_SIZE:,} rows inserted...")

    conn.close()
    
    db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"Done! Database size is {db_size_mb:.2f} MB.")
    print("This size should reliably defeat the CPU hardware prefetcher and L3 cache.")