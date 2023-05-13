# Multigrid Monte Carlo
C++ implementation of multigrid Monte Carlo algorithm

## Dependencies
The code requires the [Eigen library](https://eigen.tuxfamily.org/index.php?title=Main_Page) for linear algebra as well as [libconfig](https://hyperrealm.github.io/libconfig/) for parsing configuration files. To install libconfig, clone the [libconfig repository](https://github.com/hyperrealm/libconfig) and build/install it with CMake.

CholMod is an optional dependency, if it is not found the code falls back to using the Simplicial Cholesky factorisation in Eigen, which is not necessarily slower. Cholmod is available as part of [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html). To prevent the use of CholMod even if it has been installed, set the `USE_CHOLMOD` flag to `Off` during the CMake configure stage.

## Building the code
To compile, create a new directory called `build`. Change to this directory and run

```
cmake ..
```

to configure, followed by

```
make
```

to build the code.

## Testing the code
To run the unit tests, use

```
./bin/test
```

This can take quite long (several minutes to an hour), to build a simplified version of the tests (which essentially generated less samples when testing statistical properties) set the flag `USE_THOROUGH_TESTS` to `Off` when configuring CMake.

## Running the code
The executable is `driver` in the `bin` subdirectory. To run the code, use

```
./bin/driver CONFIG_FILE
```

where `CONFIG_FILE` is the name of the file that contains the runtime configuration; an example can be found in [parameters_template.cfg](parameters_template.cfg).