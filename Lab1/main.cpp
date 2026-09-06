#include <cstdlib>
#include <cstring>
#include <random>
#include <ctime>
#include <immintrin.h>
#include <iostream>
#include <chrono>

#define K 4
#define L 1024
#define M 512
#define N 1024
#define ALIGNMENT 32
#define MIN_RNG 0.0
#define MAX_RNG 10.0
#define CALC_ERR 1e-9

/*void matrix_multiply_classic(double**** matrix1, double**** matrix2, double**** res_matrix)
{
    for(int i = 0; i < L; ++i)
    {
        for(int j = 0; j < M; ++j)
        {
            for(int k = 0; k < N; ++k)
            {
                for(int mini_i = 0; mini_i < K; ++mini_i)
                {
                    for(int mini_j = 0; mini_j < K; ++mini_j)
                    {
                        for(int mini_k = 0; mini_k < K; ++mini_k)
                        {
                            res_matrix[i][k][mini_i][mini_k] += matrix1[i][j][mini_i][mini_j] * matrix2[j][k][mini_j][mini_k];
                        }
                    }
                }
            }
        }
    }
}*/

/*double**** init_matrix(int x, int y)
{
    double**** matrix = (double****)std::aligned_alloc(ALIGNMENT, x * sizeof(double***));

    for(int i = 0; i < x; ++i)
    {
        matrix[i] = (double***)std::aligned_alloc(ALIGNMENT, y * sizeof(double**));

        for(int j = 0; j < y; ++j)
        {
            matrix[i][j] = (double**)std::aligned_alloc(ALIGNMENT, K * sizeof(double*));

            for(int k = 0; k < K; ++k)
            {
                matrix[i][j][k] = (double*)std::aligned_alloc(ALIGNMENT, K * sizeof(double));
            }
        }
    }

    return matrix;
}*/

/*void free_matrix(double**** matrix, int x, int y)
{
    for(int i = 0; i < x; ++i)
    {
        for(int j = 0; j < y; ++j)
        {
            for(int k = 0; k < K; ++k)
            {
                free(matrix[i][j][k]);
            }

            free(matrix[i][j]);
        }

        free(matrix[i]);
    }

    free(matrix);
}*/

/*void fill_matrix(double**** matrix, int x, int y)
{
    std::mt19937 rng(time(nullptr));
    std::uniform_real_distribution<double> distr(MIN_RNG, MAX_RNG);

    for(int i = 0; i < x; ++i)
    {
        for(int j = 0; j < y; ++j)
        {
            for(int mini_i = 0; mini_i < K; ++mini_i)
            {
                for(int mini_j = 0; mini_j < K; ++mini_j)
                {
                    matrix[i][j][mini_i][mini_j] = distr(rng);
                }
            }
        }
    }
}

void fill_matrix_zeros(double**** matrix, int x, int y) 
{
    for (int i = 0; i < x; ++i) 
    {
        for (int j = 0; j < y; ++j) 
        {
            for (int mini_i = 0; mini_i < K; ++mini_i) 
            {
                memset(matrix[i][j][mini_i], 0.0, sizeof(double) * K);
            }
        }
    }
}*/

void fill_matrix(double* matrix, int x, int y)
{
    std::mt19937 rng(time(nullptr));
    std::uniform_real_distribution<double> distr(MIN_RNG, MAX_RNG);

    int total_size = x * y * K * K;
    
    for (int i = 0; i < total_size; ++i)
    {
        matrix[i] = distr(rng);
    }
}

void fill_matrix_zeros(double* matrix, int x, int y) 
{
    std::memset(matrix, 0, x * y * K * K * sizeof(double));
}

double* init_matrix(int x, int y) 
{
    return static_cast<double*>(std::aligned_alloc(ALIGNMENT, x * y * K * K * sizeof(double)));
}

inline int idx(int i, int j, int mini_i, int mini_j, int max_j) 
{
    return ((i * max_j + j) * K + mini_i) * K + mini_j;
}

void matrix_multiply_asm(double* matrix1, double* matrix2, double* res_matrix)
{
    /*for(int i = 0; i < L; ++i)
    {
        for(int j = 0; j < M; ++j)
        {
            for(int k = 0; k < N; ++k)
            {
                for(int mini_i = 0; mini_i < K; ++mini_i)
                {
                    for(int mini_j = 0; mini_j < K; ++mini_j)
                    {
                        __m256d m1_val = _mm256_set1_pd(matrix1[i][j][mini_i][mini_j]);

                        for(int mini_k = 0; mini_k < K; mini_k += 4)
                        {
                            __m256d resm_val = _mm256_load_pd(&res_matrix[i][k][mini_i][mini_k]);
                            __m256d m2_val = _mm256_load_pd(&matrix2[j][k][mini_j][mini_k]);

                            resm_val = _mm256_fmadd_pd(m1_val, m2_val, resm_val);

                            _mm256_store_pd(&res_matrix[i][k][mini_i][mini_k], resm_val);
                        }
                    }
                }
            }
        }
    }*/

    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < M; ++j) {
            for (int k = 0; k < N; ++k) {
                for (int mini_i = 0; mini_i < K; ++mini_i) {
                    for (int mini_j = 0; mini_j < K; ++mini_j) {
                        
                        __m256d matrix1_val = _mm256_set1_pd(matrix1[idx(i, j, mini_i, mini_j, M)]);

                        int res_matrix_idx = idx(i, k, mini_i, 0, N);
                        int matrix2_idx = idx(j, k, mini_j, 0, N);

                        __m256d res_matrix_val = _mm256_load_pd(&res_matrix[res_matrix_idx]);
                        __m256d matrix2_val = _mm256_load_pd(&matrix2[matrix2_idx]);

                        res_matrix_val = _mm256_fmadd_pd(matrix1_val, matrix2_val, res_matrix_val);

                        _mm256_store_pd(&res_matrix[res_matrix_idx], res_matrix_val);
                    }
                }
            }
        }
    }
}

void matrix_multiply_classic(double* matrix1, double* matrix2, double* res_matrix)
{
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            for (int k = 0; k < N; ++k)
            {
                for (int mini_i = 0; mini_i < K; ++mini_i)
                {
                    int base_res = (i * N + k) * K * K + mini_i * K;
                    int base_m1  = (i * M + j) * K * K + mini_i * K;

                    for (int mini_j = 0; mini_j < K; ++mini_j)
                    {
                        double val1 = matrix1[base_m1 + mini_j];
                        int base_m2 = (j * N + k) * K * K + mini_j * K;

                        for (int mini_k = 0; mini_k < K; ++mini_k)
                        {
                            res_matrix[base_res + mini_k] += val1 * matrix2[base_m2 + mini_k];
                        }
                    }
                }
            }
        }
    }
}

/*bool compare_matrices(double**** matrix1, double**** matrix2)
{
    for(int i = 0; i < L; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            for(int mini_i = 0; mini_i < K; ++mini_i)
            {
                for(int mini_j = 0; mini_j < K; ++mini_j)
                {
                    double diff = fabs(matrix1[i][j][mini_i][mini_j] - matrix2[i][j][mini_i][mini_j]);

                    if (diff > CALC_ERR)
                    {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}*/

bool compare_matrices(const double* matrix1, const double* matrix2)
{
    int total_size = L * N * K * K;

    for (int i = 0; i < total_size; ++i)
    {
        double diff = std::fabs(matrix1[i] - matrix2[i]);

        if (diff > CALC_ERR)
        {
            return false;
        }
    }

    return true;
}

int main()
{
    double* matrix1 = init_matrix(L, M);
    double* matrix2 = init_matrix(M, N);
    double* matrix_res_diy = init_matrix(L, N);
    double* matrix_res_asm = init_matrix(L, N);

    fill_matrix(matrix1, L, M);
    fill_matrix(matrix2, M, N);
    fill_matrix_zeros(matrix_res_diy, L, N);
    fill_matrix_zeros(matrix_res_asm, L, N);

    std::cout << "Let's go\n";

    auto start = std::chrono::high_resolution_clock::now();
    matrix_multiply_classic(matrix1, matrix2, matrix_res_diy);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Calculation ended. Time: " << diff.count() << "\n";

    start = std::chrono::high_resolution_clock::now();
    matrix_multiply_asm(matrix1, matrix2, matrix_res_asm);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;

    std::cout << "Calculation ended. Time: " << diff.count() << "\n";

    if(compare_matrices(matrix_res_diy, matrix_res_asm))
    {
        std::cout << "Somehow, it works\n";
    }
    else
    {
        std::cout << "Obviously, skill issue\n";
    }

    free(matrix1);
    free(matrix2);
    free(matrix_res_asm);
    free(matrix_res_diy);
}