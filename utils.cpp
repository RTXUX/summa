//
// Created by WYF on 2021/12/7.
//

#include "utils.h"
#include <mpi.h>
#include <memory>
#include <functional>
#include <iostream>
#include <random>
#include <cmath>
#include <cstring>

template<typename T>
using deleted_unique_ptr = std::unique_ptr<T, std::function<void(T *)>>;

static void MPI_Datatype_Deleter(MPI_Datatype *p) {
    MPI_Type_free(p);
}

void matrix_gather(MPI_Comm comm, const double M[], const int n_row, const int n_col, double result[], int root) {
    int rank, size, pr, pc;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    if (rank == root) {
        int dims[2], periods[2], my_coords[2];
        MPI_Cart_get(comm, 2, dims, periods, my_coords);
        pr = dims[0];
        pc = dims[1];
        MPI_Datatype subarray, resized_subarray;
        int sizes[] = {pr * n_row, pc * n_col};
        int sub_sizes[] = {n_row, n_col};
        int starts[] = {0, 0};
        MPI_Type_create_subarray(2, sizes, sub_sizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray);
        MPI_Type_commit(&subarray);
        deleted_unique_ptr<MPI_Datatype> subarray_owner(&subarray, MPI_Datatype_Deleter);
        MPI_Type_create_resized(subarray, 0, sizeof(double) * n_col, &resized_subarray);
        MPI_Type_commit(&resized_subarray);
        deleted_unique_ptr<MPI_Datatype> resized_subarray_owner(&resized_subarray, MPI_Datatype_Deleter);
        int displs[size];
        int recvcounts[size];
        for (int i = 0; i < size; ++i) {
            int coords[2];
            MPI_Cart_coords(comm, i, 2, coords);
            displs[i] = coords[0] * pc * n_row + coords[1];
            recvcounts[i] = 1;
        }
        MPI_Gatherv(M, n_col * n_row, MPI_DOUBLE, result, recvcounts, displs, resized_subarray, root, comm);
        // MPI_Type_free(&resized_subarray);
        // MPI_Type_free(&subarray);
    } else {
        MPI_Gatherv(M, n_col * n_row, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, root, comm);
    }
}

void matrix_scatter(MPI_Comm comm, const double M[], const int n_row, const int n_col, double result[], int root) {
    int rank, size, pr, pc;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
    if (rank == root) {
        int dims[2], periods[2], my_coords[2];
        MPI_Cart_get(comm, 2, dims, periods, my_coords);
        pr = dims[0];
        pc = dims[1];
        MPI_Datatype subarray, resized_subarray;
        int sizes[] = {pr * n_row, pc * n_col};
        int sub_sizes[] = {n_row, n_col};
        int starts[] = {0, 0};
        MPI_Type_create_subarray(2, sizes, sub_sizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray);
        MPI_Type_commit(&subarray);
        deleted_unique_ptr<MPI_Datatype> subarray_owner(&subarray, MPI_Datatype_Deleter);
        MPI_Type_create_resized(subarray, 0, sizeof(double) * n_col, &resized_subarray);
        MPI_Type_commit(&resized_subarray);
        deleted_unique_ptr<MPI_Datatype> resized_subarray_owner(&resized_subarray, MPI_Datatype_Deleter);
        int displs[size];
        int sendcounts[size];
        for (int i = 0; i < size; ++i) {
            int coords[2];
            MPI_Cart_coords(comm, i, 2, coords);
            displs[i] = coords[0] * pc * n_row + coords[1];
            sendcounts[i] = 1;
        }
        MPI_Scatterv(M, sendcounts, displs, resized_subarray, result, n_row * n_col, MPI_DOUBLE, root, comm);
        // MPI_Type_free(&resized_subarray);
        // MPI_Type_free(&subarray);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNDEFINED, result, n_col * n_row, MPI_DOUBLE, root, comm);
    }
}

void print_matrix(const double M[], const int n_row, const int n_col) {
    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
            std::cout << M[i * n_col + j] << '\t';
        }
        std::cout << std::endl;
    }
}

void init_matrix(double M[], const int n_row, const int n_col) {
#ifdef DEBUG
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    std::default_random_engine re((std::random_device()) ());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    double n = 1;
#endif
    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
#ifdef DEBUG
            M[i * n_col + j] = rank;
#else
            M[i * n_col + j] = dist(re);
#endif
        }
    }
}

double validate_matrix(const double M[], const double N[], const int n_row, const int n_col) {
    double max_err = 0.0;
    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
            size_t idx = i * n_col + j;
            double err = std::abs(M[idx] - N[idx]);
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

void matmul(const double A[], const double B[], double C[], const int row_A, const int col_A, const int col_B) {
    memset(C, 0, row_A * col_B * sizeof(double));
    for (int i = 0; i < row_A; ++i) {
        for (int k = 0; k < col_A; ++k) {
            for (int j = 0; j < col_B; ++j) {
                C[i * col_B + j] += A[i * col_A + k] * B[k * col_B + j];
            }
        }
    }
}

