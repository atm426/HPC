CC=g++
FLAGS= -g -std=c++11

all: val_test01_solved.cpp val_test02_solved.cpp MMult1.cpp fast-sin.cpp
	$(CC) $(FLAGS) -o val_test01_solved val_test01_solved.cpp 
	$(CC) $(FLAGS) -o val_test02_solved val_test02_solved.cpp
	g++-10 $(FLAGS) -O3 -march=native -fopenmp -o MMult1 MMult1.cpp
	$(CC) $(FLAGS) -O3 -o fast-sin fast-sin.cpp