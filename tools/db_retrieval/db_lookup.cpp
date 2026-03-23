#include <iostream>
#include <string>
#include <sqlite3.h>
#include <papi.h>

int main(int argc, char* argv[]) {
    // Agent provides the database path via command line
    if (argc != 2) {
        std::cerr << "{\"error\": \"Usage: ./db_lookup_baseline <db_file_path>\"}" << std::endl;
        return 1;
    }

    std::string db_path = argv[1];
    sqlite3* db;
    
    // Open the connection outside the profiling window
    if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK) {
        std::cerr << "{\"error\": \"Can't open database: " << sqlite3_errmsg(db) << "\"}" << std::endl;
        return 1;
    }

    // Force a full table scan bypassing indexes to thrash the LLC
    const char* sql_query = "SELECT COUNT(*), SUM(LENGTH(log_message)) FROM server_logs WHERE log_message LIKE '%CRITICAL ERROR%';";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(db, sql_query, -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "{\"error\": \"Failed to prepare statement: " << sqlite3_errmsg(db) << "\"}" << std::endl;
        sqlite3_close(db);
        return 1;
    }

    long long match_count = 0;
    long long total_bytes_scanned = 0;

    // --- PAPI START ---
    int Events[4] = {PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_BR_MSP, PAPI_L3_TCM};
    long long values[4] = {0, 0, 0, 0};
    
    if (PAPI_start_counters(Events, 4) != PAPI_OK) {
        std::cerr << "{\"error\": \"Failed to start PAPI counters.\"}" << std::endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return 1;
    }

    // --- MEMORY BOUND WORKLOAD ---
    // Execute the full table scan
    int step_result = sqlite3_step(stmt);
    if (step_result == SQLITE_ROW) {
        match_count = sqlite3_column_int64(stmt, 0);
        total_bytes_scanned = sqlite3_column_int64(stmt, 1);
    } else if (step_result != SQLITE_DONE) {
        // Handle execution error silently within JSON format
        match_count = -1; 
    }

    // --- PAPI STOP ---
    if (PAPI_stop_counters(values, 4) != PAPI_OK) {
        std::cerr << "{\"error\": \"Failed to stop PAPI counters.\"}" << std::endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return 1;
    }
    
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    if (match_count == -1) {
        std::cerr << "{\"error\": \"Database execution failed during step.\"}" << std::endl;
        return 1;
    }

    // Output a clean JSON string to stdout
    std::cout << "{"
              << "\"status\": \"success\", "
              << "\"result\": \"Found " << match_count << " critical errors. Scanned " << total_bytes_scanned << " bytes of match data.\", "
              << "\"metrics\": {"
              << "\"cycles\": " << values[0] << ", "
              << "\"instructions\": " << values[1] << ", "
              << "\"branch_misses\": " << values[2] << ", "
              << "\"l3_misses\": " << values[3]
              << "}"
              << "}" << std::endl;

    return 0;
}