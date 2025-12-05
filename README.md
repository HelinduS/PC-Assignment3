# N-body Simulation - Compilation & Execution Instructions

## Directory Structure
- Serial/serial.c
- OpenMP/openmp.c
- MPI/mpi.c
- cuda/nbody.cu

## Compilation Instructions

### Serial
```
gcc -O2 serial.c -o serial -lm
```

### OpenMP
```
gcc -O2 -fopenmp openmp.c -o openmp -lm
```

### MPI
```
mpicc -O2 mpi.c -o mpi -lm
```

### CUDA
```
nvcc -O3 -arch=sm_75 nbody.cu -o nbody
```

## Execution Instructions

### Serial
```
./serial [N] [steps]
```

### OpenMP
```
OMP_NUM_THREADS=4 ./openmp [N] [steps]
```

### MPI
```
mpirun -np 4 ./mpi [N] [steps]
```

### CUDA
```
./nbody [N] [steps]
```

- `[N]` = number of bodies (default: 1024)
- `[steps]` = number of simulation steps (default: 100)

---

