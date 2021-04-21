#include <stdio.h>
#include <mpi.h>

// mpicc -g -O3 -o int_ring int_ring.c

int main( int argc, char** argv){
  
  MPI_Init(&argc, &argv);

  // how many times around the loop
  int loopN = atoi(argv[1]);

  // getting process rank
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // getting number of ranks
  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  int memo = 0;
  // start timer
  double tt = MPI_Wtime();
  for (int i = 1 ; i <= loopN ; i++){
    MPI_Status  status;

    //  getting processor name
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    //printf("Rank %d/%d running on %s.\n", rank, num_ranks, processor_name);
    if (rank == 0){
      // send message to rank + 1
      memo += 1;
      MPI_Send(&memo, 1, MPI_INT, rank+1, 666, MPI_COMM_WORLD);

      // rank[0] processor recv from rank[size-1] processor
      MPI_Recv(&memo, 1, MPI_INT, num_ranks-1, 666, MPI_COMM_WORLD, &status);
      printf("The loop is on iteration %d/%d with %d nodes, memo = %d \n", i, loopN, num_ranks, memo);
    }
    else if (rank == num_ranks - 1){
      // recv from rank-1 processor
      MPI_Recv(&memo, 1, MPI_INT, rank-1, 666, MPI_COMM_WORLD, &status);

      // send to rank 0 processor
      memo += 1;
      MPI_Send(&memo, 1, MPI_INT, 0, 666, MPI_COMM_WORLD);
    }
    else{
      // recv from rank - 1 processor
      MPI_Recv(&memo, 1, MPI_INT, rank-1, 666, MPI_COMM_WORLD, &status);

      // send to rank 0 processor
      memo += 1; 
      MPI_Send(&memo, 1, MPI_INT, rank+1, 666, MPI_COMM_WORLD);
    }
  }
  // print time once memo is back at rank 0
  if (rank == 0){
    tt = MPI_Wtime() - tt; 
    printf("integer ring time = %e \n", tt/ (((double) loopN)*num_ranks));
  }
  MPI_Finalize();
  return 0;
}
