#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <algorithm>
#include <cmath>

std::vector<std::vector<double>> generateRandomMatrix(int rows, int cols, int minVal, int maxVal, unsigned int seed) {
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(minVal, maxVal);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = static_cast<double>(dist(rng));
        }
    }
    return matrix;
}

void printMatrix(const std::string& name, const std::vector<std::vector<double>>& matrix) {
    std::cout << name << " (" << matrix.size() << "x" << (matrix.empty() ? 0 : matrix[0].size()) << "):" << std::endl;
    if (matrix.empty() || matrix[0].empty()) {
        std::cout << "[Empty Matrix]" << std::endl;
        return;
    }
    int rows = matrix.size();
    int cols = matrix[0].size();
    const int MAX_PRINT_ROWS = 10;
    const int MAX_PRINT_COLS = 10;

    for (int i = 0; i < std::min(rows, MAX_PRINT_ROWS); ++i) {
        for (int j = 0; j < std::min(cols, MAX_PRINT_COLS); ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << matrix[i][j] << " ";
        }
        if (cols > MAX_PRINT_COLS) {
            std::cout << "...";
        }
        std::cout << std::endl;
    }
    if (rows > MAX_PRINT_ROWS) {
        std::cout << "..." << std::endl;
    }
    std::cout << std::endl;
}


int main(int argc, char* argv[]) {
    const int M = 800;
    const int K = 600;
    const int N = 700;

    const int MIN_VAL = 1;
    const int MAX_VAL = 10;
    const unsigned int SEED_A = 12345;
    const unsigned int SEED_B = 54321;

    const int NUM_DESIRED_THREADS = 4;
    omp_set_num_threads(NUM_DESIRED_THREADS);

    std::vector<std::vector<double>> matrix_A;
    std::vector<std::vector<double>> matrix_B;
    std::vector<std::vector<double>> matrix_C(M, std::vector<double>(N, 0.0));

    double start_time, end_time;



    matrix_A = generateRandomMatrix(M, K, MIN_VAL, MAX_VAL, SEED_A);
    matrix_B = generateRandomMatrix(K, N, MIN_VAL, MAX_VAL, SEED_B);

    if (M <= 20 && K <= 20 && N <= 20) {
        printMatrix("Matrix A", matrix_A);
        printMatrix("Matrix B", matrix_B);
    }

    start_time = omp_get_wtime();

    #pragma omp parallel for
        for (int i = 0; i < M; ++i) { // i - індекс рядка в C (і в A)
            for (int j = 0; j < N; ++j) { // j - індекс стовпця в C (і в B)
                double sum = 0.0; // Локальна сума для кожного елемента C[i][j]
                for (int l = 0; l < K; ++l) { // l - спільний індекс для стовпців A та рядків B
                    sum += matrix_A[i][l] * matrix_B[l][j];
                }
                matrix_C[i][j] = sum;
            }
        }

    end_time = omp_get_wtime();

    if (M <= 20 && N <= 20) {
        printMatrix("Matrix C (result)", matrix_C);
    }

    // Перевірка (послідовне множення)
    if (M * K * N < 100000) {
        std::cout << "Performing sequential multiplication for verification..." << std::endl;
        std::vector<std::vector<double>> matrix_C_sequential(M, std::vector<double>(N, 0.0));
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int l = 0; l < K; ++l) {
                    matrix_C_sequential[i][j] += matrix_A[i][l] * matrix_B[l][j];
                }
            }
        }
        bool match = true;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                if (std::abs(matrix_C[i][j] - matrix_C_sequential[i][j]) > 1e-6) {
                    match = false;
                    break;
                }
            }
            if (!match) break;
        }
        if (match) {
            std::cout << "Verification successful: OpenMP result matches sequential result." << std::endl;
        }
        else {
            std::cout << "Verification FAILED: OpenMP result DOES NOT match sequential result." << std::endl;
            if (M <= 20 && N <= 20) printMatrix("Matrix C (sequential for verification)", matrix_C_sequential);
        }
    }

    std::cout << "Matrix dimensions: A(" << M << "x" << K << "), B(" << K << "x" << N << "), C(" << M << "x" << N << ")" << std::endl;
    std::cout << "Execution time (OpenMP): " << std::fixed << std::setprecision(5) << end_time - start_time << " seconds." << std::endl;

    int num_threads_used = 0;
#pragma omp parallel
    {
#pragma omp master 
        num_threads_used = omp_get_num_threads();
    }
    std::cout << "Number of threads used (OpenMP): " << num_threads_used << std::endl;

    return 0;
}
