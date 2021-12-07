//
// Created by WYF on 2021/12/7.
//

#ifndef SUMMA_UTILS_H
#define SUMMA_UTILS_H

#include <mpi.h>

void matrix_scatter(MPI_Comm comm, const double M[], int n_row, int n_col, double result[], int root);

void matrix_gather(MPI_Comm comm, const double M[], int n_row, int n_col, double result[], int root);

void print_matrix(const double M[], int n_row, int n_col);

void init_matrix(double M[], int n_row, int n_col);

double validate_matrix(const double M[], const double N[], int n_row, int n_col);

void matmul(const double A[], const double B[], double C[], int row_A, int col_A, int col_B);

template<typename T>
void safe_delete_array(T *&p) {
    if (p != nullptr) {
        delete[] p;
        p = nullptr;
    }
}

#endif //SUMMA_UTILS_H
