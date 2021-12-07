//
// Created by WYF on 2021/12/7.
//

#include "summa.h"
#include <mpi.h>
#include <iostream>
#include <cstring>
#include <memory>
#include <functional>

static void MPI_Comm_Deleter(MPI_Comm *comm) {
    MPI_Comm_free(comm);
}

static void MPI_Datatype_Deleter(MPI_Datatype *p) {
    MPI_Type_free(p);
}

template<typename T>
using deleted_unique_ptr = std::unique_ptr<T, std::function<void(T *)>>;

void summa(MPI_Comm comm_grid, const int row_A, const int col_A, const int row_B, const int col_B, const int nb,
           double A_local[], double B_local[], double C_local[]) {
    if (col_A % nb != 0 || row_B % nb != 0) {
        std::cerr << "k must be multiple of nb" << std::endl;
        MPI_Abort(comm_grid, 1);
        return;
    }
    // int n_block_per_process = k / nb;
    int nb_col_pp = col_A / nb;
    int nb_row_pp = row_B / nb;
    int rank, size, rank_row, rank_col, size_row, size_col;
    MPI_Comm_rank(comm_grid, &rank);
    MPI_Comm_size(comm_grid, &size);
    int dims[2];
    int periods[2];
    int my_coords[2];
    MPI_Cart_get(comm_grid, 2, dims, periods, my_coords);
    if (nb_col_pp * dims[1] != nb_row_pp * dims[0]) {
        std::cerr << "Not multiplicable matrices" << std::endl;
        MPI_Abort(comm_grid, 1);
        return;
    }
    int nb_global = nb_col_pp * dims[1];
    int remain_dims[2] = {0, 1};
    MPI_Comm row_comm;
    MPI_Comm col_comm;
    MPI_Cart_sub(comm_grid, remain_dims, &row_comm);
    deleted_unique_ptr<MPI_Comm> _row_comm(&row_comm, MPI_Comm_Deleter);
    MPI_Comm_rank(row_comm, &rank_row);
    remain_dims[0] = 1;
    remain_dims[1] = 0;
    MPI_Cart_sub(comm_grid, remain_dims, &col_comm);
    deleted_unique_ptr<MPI_Comm> _col_comm(&col_comm, MPI_Comm_Deleter);
    MPI_Comm_rank(col_comm, &rank_col);
    MPI_Comm_size(row_comm, &size_row);
    MPI_Comm_size(col_comm, &size_col);
    if (nb_col_pp * size_row != nb_row_pp * size_col) {
        std::cerr << "Not multiplicable matrices" << std::endl;
        MPI_Abort(comm_grid, 1);
        return;
    }
    memset(C_local, 0, row_A * col_B * sizeof(double));
    double *buffer_A = new double[nb * row_A];
    std::unique_ptr<double[]> _buffer_A(buffer_A);
    double *buffer_B = new double[nb * col_B];
    std::unique_ptr<double[]> _buffer_B(buffer_B);
    for (int k = 0; k < nb_global; ++k) {
        double *buff_ptr_A, *buff_ptr_B;
        int owner_A = k / nb_col_pp, owner_B = k / nb_row_pp;
        if (owner_A == rank_row) {
            // We still need to collect part of A to contiguous memory
            buff_ptr_A = buffer_A;
            int idx = 0;
            for (int i = 0; i < row_A; ++i) {
                for (int j = 0; j < nb; ++j) {
                    buff_ptr_A[idx++] = A_local[(k % nb_col_pp) * nb + i * col_A + j];
                }
            }
        } else {
            buff_ptr_A = buffer_A;
        }
        MPI_Bcast(buff_ptr_A, nb * row_A, MPI_DOUBLE, owner_A, row_comm);
        // Memory of B's part is already contiguous
        if (owner_B == rank_col) {
            buff_ptr_B = &B_local[(k % nb_row_pp) * nb * col_B];
        } else {
            buff_ptr_B = buffer_B;
        }
        MPI_Bcast(buff_ptr_B, nb * col_B, MPI_DOUBLE, owner_B, col_comm);
        for (int i = 0; i < row_A; ++i) {
            for (int l = 0; l < nb; ++l) {
                for (int j = 0; j < col_B; ++j) {
                    C_local[i * col_B + j] += buff_ptr_A[i * nb + l] * buff_ptr_B[l * col_B + j];
                }
            }
        }
    }
}
