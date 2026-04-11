#include <iostream>
#include <string>
#include <filesystem>
#include <papi.h>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // Agent provides the target directory path via command line
    if (argc != 2) {
        std::cerr << "{\"error\": \"Usage: ./io_walker_baseline <directory_path>\"}" << std::endl;
        return 1;
    }

    std::string dir_path = argv[1];
    
    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        std::cerr << "{\"error\": \"Invalid directory path: " << dir_path << "\"}" << std::endl;
        return 1;
    }

    long long total_files = 0;
    long long total_size_bytes = 0;

    // --- PAPI START ---
    int Events[4] = {PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_BR_MSP, PAPI_L3_TCM};
    long long values[4] = {0, 0, 0, 0};
    
    if (PAPI_start_counters(Events, 4) != PAPI_OK) {
        std::cerr << "{\"error\": \"Failed to start PAPI counters.\"}" << std::endl;
        return 1;
    }

    // --- I/O BOUND WORKLOAD ---
    // Recursively walk the directory tree, querying the OS for metadata on every single file.
    try {
        for (const auto& entry : fs::recursive_directory_iterator(dir_path, fs::directory_options::skip_permission_denied)) {
            if (fs::is_regular_file(entry)) {
                total_files++;
                total_size_bytes += fs::file_size(entry);
            }
        }
    } catch (const fs::filesystem_error& e) {
        PAPI_stop_counters(values, 4);
        std::cerr << "{\"error\": \"Filesystem error: " << e.what() << "\"}" << std::endl;
        return 1;
    }

    // --- PAPI STOP ---
    if (PAPI_stop_counters(values, 4) != PAPI_OK) {
        std::cerr << "{\"error\": \"Failed to stop PAPI counters.\"}" << std::endl;
        return 1;
    }
    
    // Output a clean JSON string to stdout
    std::cout << "{"
              << "\"status\": \"success\", "
              << "\"result\": \"Found " << total_files << " files (" << total_size_bytes << " bytes)\", "
              << "\"metrics\": {"
              << "\"cycles\": " << values[0] << ", "
              << "\"instructions\": " << values[1] << ", "
              << "\"branch_misses\": " << values[2] << ", "
              << "\"l3_misses\": " << values[3]
              << "}"
              << "}" << std::endl;

    return 0;
}