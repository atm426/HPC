CPP=g++-10 -std=c++11
CC=gcc-10
FLAGS= -g -O3 -fopenmp

all: omp_bug2.c omp_bug3.c omp_bug4.c omp_bug5.c omp_bug6.c omp-scan.cpp jacobi2D-omp.cpp
	$(CC) $(FLAGS) -o omp-solved2 omp_bug2.c 
	$(CC) $(FLAGS) -o omp-solved3 omp_bug3.c 
	$(CC) $(FLAGS) -o omp-solved4 omp_bug4.c 
	$(CC) $(FLAGS) -o omp-solved5 omp_bug5.c 
	$(CC) $(FLAGS) -o omp-solved6 omp_bug6.c 
	$(CPP) $(FLAGS) -o omp-scan omp-scan.cpp
	$(CPP) $(FLAGS) -o jacobi2D-omp jacobi2D-omp.cpp