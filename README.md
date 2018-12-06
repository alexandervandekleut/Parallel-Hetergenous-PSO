# Parallel-Hetergenous-PSO
Particle Swarm Optimizer (PSO) with 4 parameter modification behaviours implemented in C++. Uses OpenMPI for parallelization.

### Compiling
Compiling can be different on different systems. Any of the following may work:

```mpicc -lc++ -o <output name> PSO.cpp```

```mpicc -std=c++11 -o <output name> PSO.cpp```

```mpiCC -o <output name> PSO.cpp```
  
### Running
Running is simple.

```mpiexec -n <number of processes> <output name> <number of swarms per process>```
  
Run with 4 or more processes to guarantee exposure to all behaviours.
