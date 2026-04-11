#include <iostream>
#include <string>
#include <filesystem>
#include <papi.h>
#include <omp.h>
#include <atomic>

namespace fs = std::filesystem;

// The parallel recursive task
void walk_directory_omp(const fs::path& path, std::atomic<long long>& file_count, std::atomic<long long>& byte_count) {
    try {
        for (const auto& entry : fs::directory_iterator(path, fs::directory_options::skip_permission_denied)) {
            if (fs::is_directory(entry)) {
                // Spawn a new OpenMP task for every sub-directory
                #pragma omp task shared(file_count, byte_count)
                walk_directory_omp(entry.path(), file_count, byte_count);
            } else if (fs::is_regular_file(entry)) {
                file_count++;
                byte_count += fs::file_size(entry);
            }
        }
    } catch (...) {
        // Silently skip permission denied folders
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "{\"error\": \"Usage: ./io_walker_omp <directory_path>\"}" << std::endl;
        return 1;
    }

    std::string dir_path = argv[1];
    
    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        std::cerr << "{\"error\": \"Invalid directory path: " << dir_path << "\"}" << std::endl;
        return 1;
    }

    std::atomic<long long> total_files{0};
    std::atomic<long long> total_size_bytes{0};

    // --- PAPI START ---
    int Events[4] = {PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_BR_MSP, PAPI_L3_TCM};
    long long values[4] = {0, 0, 0, 0};
    
    if (PAPI_start_counters(Events, 4) != PAPI_OK) {
        std::cerr << "{\"error\": \"Failed to start PAPI counters.\"}" << std::endl;
        return 1;
    }

    // --- I/O BOUND WORKLOAD (OPENMP PARALLEL) ---
    // Start the parallel region
    #pragma omp parallel
    {
        // Only one thread starts the recursion tree; tasks spread out from there
        #pragma omp single
        {
            walk_directory_omp(dir_path, total_files, total_size_bytes);
        }
    }

    // --- PAPI STOP ---
    if (PAPI_stop_counters(values, 4) != PAPI_OK) {
        std::cerr << "{\"error\": \"Failed to stop PAPI counters.\"}" << std::endl;
        return 1;
    }
    
    std::cout << "{"
              << "\"status\": \"success\", "
              << "\"result\": \"Found " << total_files.load() << " files (" << total_size_bytes.load() << " bytes)\", "
              << "\"metrics\": {"
              << "\"cycles\": " << values[0] << ", "
              << "\"instructions\": " << values[1] << ", "
              << "\"branch_misses\": " << values[2] << ", "
              << "\"l3_misses\": " << values[3]
              << "}"
              << "}" << std::endl;

    return 0;
}