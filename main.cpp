#include <iostream>
#include <mpi.h>
#include <memory>
#include "summa.h"
#include "utils.h"

int row_A, col_A, col_B, Pr, Pc, nb, repeats;
bool validate;
int periods[] = {0, 0};

static void parse_cmdline(int argc, char *argv[]) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 9) {
        if (rank == 0) {
            std::cerr << "We need 8 arguments: row_A, col_A, col_B, Pr, Pc, nb, repeats, validate" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    row_A = std::atoi(argv[1]);
    col_A = std::atoi(argv[2]);
    col_B = std::atoi(argv[3]);
    Pr = std::atoi(argv[4]);
    Pc = std::atoi(argv[5]);
    nb = std::atoi(argv[6]);
    repeats = std::atoi(argv[7]);
    validate = std::atoi(argv[8]) != 0;
    if (rank == 0) {
        if (row_A % Pr != 0) {
            std::cerr << "row_A must be divisible by Pr" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (col_B % Pc != 0) {
            std::cerr << "col_B must be divisible by Pc" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (col_A % Pc != 0) {
            std::cerr << "col_A must be divisible by Pc" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (col_A % Pr != 0) {
            std::cerr << "col_A must be divisible by Pr" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (Pr * Pc > size) {
            std::cerr << "We need at least " << Pr * Pc << " Processors, but we have only " << size << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;

    parse_cmdline(argc, argv);
    MPI_Comm comm_grid;
    int dims[] = {Pr, Pc};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_grid);
    if (comm_grid == MPI_COMM_NULL) {
        MPI_Finalize();
        return 0;
    }
    MPI_Comm_rank(comm_grid, &rank);
    MPI_Comm_size(comm_grid, &size);
    int row_A_local = row_A / Pr;
    int col_A_local = col_A / Pc;
    int row_B_local = col_A / Pr;
    int col_B_local = col_B / Pc;
    std::unique_ptr<double[]> A_local(new double[row_A_local * col_A_local]);
    std::unique_ptr<double[]> B_local(new double[row_B_local * col_B_local]);
    std::unique_ptr<double[]> C_local(new double[row_A_local * col_B_local]);
    init_matrix(A_local.get(), row_A_local, col_A_local);
    init_matrix(B_local.get(), row_B_local, col_B_local);
    double start, end;
    start = MPI_Wtime();
    for (int i = 0; i < repeats; ++i) {
        summa(comm_grid, row_A_local, col_A_local, row_B_local, col_B_local, nb, A_local.get(), B_local.get(),
              C_local.get());
    }
    end = MPI_Wtime();
    double duration = end - start;
    double max_duration;
    MPI_Reduce(&duration, &max_duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm_grid);
    if (rank == 0) {
        std::cout << "SUMMA MPI Impl: " << row_A << "*" << col_A << ", " << col_A << "*" << col_B << ", nb=" << nb
                  << std::endl;
        std::cout << "Finished " << repeats << " operations in " << max_duration << " secs" << std::endl;
        std::cout << "AvgTime/Op: " << max_duration / repeats << std::endl;
        std::cout << "Speed: " << repeats / max_duration << " op/s" << std::endl;
    }
    if (validate) {
        std::unique_ptr<double[]> A, B, C, C_summa;

        if (rank == 0) {
            std::cout << "Validating results" << std::endl;
            A = std::unique_ptr<double[]>(new double[row_A * col_A]);
            B = std::unique_ptr<double[]>(new double[col_A * col_B]);
            C = std::unique_ptr<double[]>(new double[row_A * col_B]);
            C_summa = std::unique_ptr<double[]>(new double[row_A * col_B]);
        }
        matrix_gather(comm_grid, A_local.get(), row_A_local, col_A_local, A.get(), 0);
        matrix_gather(comm_grid, B_local.get(), row_B_local, col_B_local, B.get(), 0);
        matrix_gather(comm_grid, C_local.get(), row_A_local, col_B_local, C_summa.get(), 0);
        if (rank == 0) {
            start = MPI_Wtime();
            matmul(A.get(), B.get(), C.get(), row_A, col_A, col_B);
            end = MPI_Wtime();
            duration = end - start;
            std::cout << "Serial Impl: " << duration << "s, " << "Speed: " << 1 / duration << " op/s" << std::endl;
#ifdef DEBUG
            std::cout << "A:" << std::endl;
            print_matrix(A.get(), row_A, col_A);
            std::cout << "B:" << std::endl;
            print_matrix(B.get(), col_A, col_B);
            std::cout << "C:" << std::endl;
            print_matrix(C.get(), row_A, col_B);
            std::cout << "C summa:" << std::endl;
            print_matrix(C_summa.get(), row_A, col_B);
#endif
            double err = validate_matrix(C.get(), C_summa.get(), row_A, col_B);
            std::cout << "Max Error: " << err << std::endl;
        }
    }
    MPI_Comm_free(&comm_grid);
    MPI_Finalize();
    return 0;
}
