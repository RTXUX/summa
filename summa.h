//
// Created by WYF on 2021/12/7.
//

#ifndef SUMMA_SUMMA_H
#define SUMMA_SUMMA_H

#include <mpi.h>

void summa(MPI_Comm comm_grid, int row_A, int col_A, int row_B, int col_B, int nb, double A_local[], double B_local[],
           double C_local[]);

#endif //SUMMA_SUMMA_H
