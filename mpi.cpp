#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <mpi.h>
#include <cassert>

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
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Параметри матриць та генерації
    const int M = 800; // Кількість рядків в A (і в C) - має ділитися на size для простоти
    const int K = 600;  // Кількість стовпців в A та рядків в B
    const int N = 700; // Кількість стовпців в B (і в C)

    const int MIN_VAL = 1;
    const int MAX_VAL = 10;
    const unsigned int SEED_A = 12345;
    const unsigned int SEED_B = 54321;

    if (M % size != 0 && rank == 0) {
        std::cerr << "Error: Number of rows in matrix A (" << M
            << ") must be divisible by the number of processes (" << size
            << ") for this simple implementation." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<std::vector<double>> matrix_A_full;
    std::vector<std::vector<double>> matrix_B_full(K, std::vector<double>(N)); // Всі процеси отримають повну B
    std::vector<std::vector<double>> matrix_C_full; // Тільки на процесі 0

    double start_time, end_time;

    // Генерація матриць A і B на процесі 0
    if (rank == 0) {
        matrix_A_full = generateRandomMatrix(M, K, MIN_VAL, MAX_VAL, SEED_A);
        matrix_B_full = generateRandomMatrix(K, N, MIN_VAL, MAX_VAL, SEED_B);

        if (M <= 20 && K <= 20 && N <= 20) {
            printMatrix("Matrix A (full)", matrix_A_full);
            printMatrix("Matrix B (full)", matrix_B_full);
        }
        matrix_C_full.assign(M, std::vector<double>(N, 0.0));
    }

    start_time = MPI_Wtime();

    // Розсилка матриці B всім процесам
    // Розсилаємо кожен рядок матриці B
    for (int i = 0; i < K; ++i) {
        MPI_Bcast(matrix_B_full[i].data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Розподіл рядків матриці A та обчислення
    int rows_per_process = M / size;
    std::vector<std::vector<double>> local_A_chunk(rows_per_process, std::vector<double>(K));
    std::vector<std::vector<double>> local_C_chunk(rows_per_process, std::vector<double>(N, 0.0));

    // Розсилка рядків матриці A.
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            for (int i = 0; i < rows_per_process; ++i) {
                int global_row_index = p * rows_per_process + i;
                if (p == 0) { // Процес 0 копіює свою частину
                    local_A_chunk[i] = matrix_A_full[global_row_index];
                }
                else {
                    MPI_Send(matrix_A_full[global_row_index].data(), K, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
                }
            }
        }
    }
    else {
        for (int i = 0; i < rows_per_process; ++i) {
            MPI_Recv(local_A_chunk[i].data(), K, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }


    // Обчислення локальної частини матриці C
    // local_C_chunk[i][j] = sum(local_A_chunk[i][l] * matrix_B_full[l][j]) for l = 0 to K-1
    for (int i = 0; i < rows_per_process; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int l = 0; l < K; ++l) {
                local_C_chunk[i][j] += local_A_chunk[i][l] * matrix_B_full[l][j];
            }
        }
    }

    // Збір результатів на процес 0
    // Кожен процес (крім 0) надсилає свою local_C_chunk. Процес 0 отримує.
    if (rank == 0) {
        // Копіюємо власну частину C
        for (int i = 0; i < rows_per_process; ++i) {
            matrix_C_full[i] = local_C_chunk[i]; // ранг 0 обробляє перші rows_per_process рядків
        }

        // Отримуємо частини від інших процесів
        for (int p = 1; p < size; ++p) {
            for (int i = 0; i < rows_per_process; ++i) {
                int global_row_index = p * rows_per_process + i;
                MPI_Recv(matrix_C_full[global_row_index].data(), N, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        end_time = MPI_Wtime();

        if (M <= 20 && N <= 20) {
            printMatrix("Matrix C (result)", matrix_C_full);
        }

        if (rank == 0 && M * K * N < 100000) { // Робимо послідовне множення для перевірки
            std::cout << "Performing sequential multiplication for verification..." << std::endl;
            std::vector<std::vector<double>> matrix_C_sequential(M, std::vector<double>(N, 0.0));
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    for (int l = 0; l < K; ++l) {
                        matrix_C_sequential[i][j] += matrix_A_full[i][l] * matrix_B_full[l][j];
                    }
                }
            }
            bool match = true;
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    if (std::abs(matrix_C_full[i][j] - matrix_C_sequential[i][j]) > 1e-6) {
                        match = false;
                        break;
                    }
                }
                if (!match) break;
            }
            if (match) {
                std::cout << "Verification successful: MPI result matches sequential result." << std::endl;
            }
            else {
                std::cout << "Verification FAILED: MPI result DOES NOT match sequential result." << std::endl;
                if (M <= 20 && N <= 20) printMatrix("Matrix C (sequential for verification)", matrix_C_sequential);
            }
        }


        std::cout << "Matrix dimensions: A(" << M << "x" << K << "), B(" << K << "x" << N << "), C(" << M << "x" << N << ")" << std::endl;
        std::cout << "Execution time: " << std::fixed << std::setprecision(5) << end_time - start_time << " seconds." << std::endl;
        std::cout << "Number of processes used: " << size << std::endl;

    }
    else { // Робітничі процеси надсилають свої результати
        for (int i = 0; i < rows_per_process; ++i) {
            MPI_Send(local_C_chunk[i].data(), N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}