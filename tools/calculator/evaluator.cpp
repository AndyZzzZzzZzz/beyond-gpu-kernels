#include <iostream>
#include <string>
#include <cctype>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <sstream>
#include <papi.h>

class Evaluator {
private:
    std::string expr;
    size_t pos;

    // Helper to skip whitespace
    void skipWhitespace() {
        while (pos < expr.length() && std::isspace(expr[pos])) {
            pos++;
        }
    }

    // Parse numbers (doubles)
    double parseNumber() {
        skipWhitespace();
        size_t start = pos;
        while (pos < expr.length() && (std::isdigit(expr[pos]) || expr[pos] == '.')) {
            pos++;
        }
        if (start == pos) throw std::runtime_error("Expected a number");
        return std::stod(expr.substr(start, pos - start));
    }

    // Parse factors: Numbers or (Expression)
    double parseFactor() {
        skipWhitespace();
        if (pos < expr.length() && expr[pos] == '(') {
            pos++; // Consume '('
            double val = parseExpression();
            skipWhitespace();
            if (pos >= expr.length() || expr[pos] != ')') {
                throw std::runtime_error("Mismatched parentheses");
            }
            pos++; // Consume ')'
            return val;
        }
        return parseNumber();
    }

    // Parse terms: * and /
    double parseTerm() {
        double val = parseFactor();
        while (true) {
            skipWhitespace();
            if (pos < expr.length() && expr[pos] == '*') {
                pos++;
                val *= parseFactor();
            } else if (pos < expr.length() && expr[pos] == '/') {
                pos++;
                double divisor = parseFactor();
                if (divisor == 0) throw std::runtime_error("Division by zero");
                val /= divisor;
            } else {
                break;
            }
        }
        return val;
    }

    // Parse expressions: + and -
    double parseExpression() {
        double val = parseTerm();
        while (true) {
            skipWhitespace();
            if (pos < expr.length() && expr[pos] == '+') {
                pos++;
                val += parseTerm();
            } else if (pos < expr.length() && expr[pos] == '-') {
                pos++;
                val -= parseTerm();
            } else {
                break;
            }
        }
        return val;
    }

public:
    double evaluate(const std::string& expression) {
        expr = expression;
        pos = 0;
        double result = parseExpression();
        skipWhitespace();
        if (pos < expr.length()) {
            throw std::runtime_error("Unexpected characters at end of expression");
        }
        return result;
    }
};

int main(int argc, char* argv[]) {
    // 1. Agent provides the file path via command line
    if (argc != 2) {
        std::cerr << "{\"error\": \"Usage: ./eval_baseline <file_path>\"}" << std::endl;
        return 1;
    }

    std::string file_path = argv[1];
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        std::cerr << "{\"error\": \"Could not open file: " << file_path << "\"}" << std::endl;
        return 1;
    }

    // 2. Load the entire stress-test payload into memory
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string heavy_expression = buffer.str();
    file.close();

    Evaluator eval;
    double result = 0;

    // --- PAPI START ---
    int Events[4] = {PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_BR_MSP, PAPI_L3_TCM};
    long long values[4] = {0, 0, 0, 0};
    
    if (PAPI_start_counters(Events, 4) != PAPI_OK) {
        std::cerr << "{\"error\": \"Failed to start PAPI counters.\"}" << std::endl;
        return 1;
    }

    // 3. Execute the payload once to defeat branch prediction
    try {
        result = eval.evaluate(heavy_expression);
    } catch (const std::exception& e) {
        PAPI_stop_counters(values, 4);
        std::cerr << "{\"error\": \"Evaluation failed: " << e.what() << "\"}" << std::endl;
        return 1;
    }

    // --- PAPI STOP ---
    if (PAPI_stop_counters(values, 4) != PAPI_OK) {
        std::cerr << "{\"error\": \"Failed to stop PAPI counters.\"}" << std::endl;
        return 1;
    }
    
    // 4. Output a clean JSON string to stdout for the Python agent to parse
    std::cout << "{"
              << "\"status\": \"success\", "
              << "\"result\": " << result << ", "
              << "\"metrics\": {"
              << "\"cycles\": " << values[0] << ", "
              << "\"instructions\": " << values[1] << ", "
              << "\"branch_misses\": " << values[2] << ", "
              << "\"l3_misses\": " << values[3]
              << "}"
              << "}" << std::endl;

    return 0;
}