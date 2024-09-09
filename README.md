# mole_math
![Static Badge](https://img.shields.io/badge/Linux-black?style=flat&logo=linux&labelColor=black&color=red)   ![Static Badge](https://img.shields.io/badge/C-black?style=flat&logo=C&labelColor=black&color=blue) ![Static Badge](https://img.shields.io/badge/OpenMP-black?style=flat&logo=C&labelColor=black&color=blue) ![Static Badge](https://img.shields.io/badge/CUnit-black?style=flat&logo=C&labelColor=black&color=blue)
## Contents:
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [The basics](#the-basics)

## Overview
**mole_math** is an easy to use and easy to setup matrix-math library written entirely in C. It makes use of OpenMP for additional speed-up by parallelization. CUnit framework is used for unit-testing of all functions. The library was also manually tested with *valgrind* to avoid memory leaks. 

## Requirements
- [GCC](https://gcc.gnu.org/) compiler (with OpenMP)
- [Make](https://www.gnu.org/software/make/)
- [CUnit](https://cunit.sourceforge.net/)
## Installation
First, clone the repository.
```sh
git clone https://github.com/Griger5/mole_math.git
```
Then enter the cloned directory and run the *setup* script.
```sh
cd mole_math
sudo ./setup.sh
```
or
```
sudo ./setup.sh install
```
The script takes care of compiling all required files into a shared library with *Make*. It also as compiles the tests. **sudo** command is needed, because the script copies the headers and compiled library into */usr/local/include/* and */usr/local/lib/* directories respectively.
<br>

To run the tests, enter the *test* directory and execute *run_tests* script:
```sh
cd test
./run_tests.sh
```

To uninstall the library, simply execute:

```sh
sudo ./setup.sh uninstall
```

## The basics
All functions can be accessed with just one header.
```c
#include <mole_math/matrix.h>
```
Additionally, during compilation you should add a *-lmolemath* flag, like so:
```sh
gcc foo.c -o foo -lmolemath
```
<br>

Matrices can be initialized with *matrix_init(rows, cols)*, but also with any function returning a *Matrix* type. All matrices are dynamically allocated and should be freed.

The *Matrix* is composed of 3 fields: *size_t rows*, *size_t cols*, *double \*\*values*. *rows* and *cols* hold the number of rows and columns of the matrix, while *values* contains the actual matrix. *Matrix.values* is allocated as a contiguous block of memory in such a way, that regular indexing approach works just fine.
```c
some_matrix.values[i][j] // returns the value of an entry at i-th row and j-th column
```
<br>

If a Matrix-returning function fails, due to allocation failure or otherwise (f.e. *matrix_inverse* was called with a non-square matrix), it will return a Matrix whose *values* field is equal to *NULL*.  
<br>

Example program:
```c
// example.c
#include <mole_math/matrix.h>

int main(void) {
	Matrix matrix_a = matrix_init(2,2);
	matrix_a.values[0][0] = 2;
	matrix_a.values[0][1] = 3;
	matrix_a.values[1][0] = -5;
	matrix_a.values[1][1] = 10.5;

	Matrix matrix_b = matrix_random(2,2);

	Matrix result = matrix_multiply(matrix_a, matrix_b);

	if (result.values != NULL) {
		matrix_print(result);
	}
	
	matrix_free(&matrix_a);
	matrix_free(&matrix_b);
	matrix_free(&result); // matrix_free is safe to be used on "nulled" matrices

	return 0;
}
```
```sh
>>> gcc example.c -o example -lmolemath
>>> ./example
4.029673 3.184086 
4.021603 6.411706 
```
<br>

For convenience, a macro was added to simplify freeing, so you can just do:
```c
MFREE(some_matrix)
```
<br>

Most functions can be called with an additional parameter specifying if a sequential or a parallelized version should be used. All function calls below are valid and won't cause warning or errors.

```c
matrix_inverse(some_matrix);
matrix_inverse(some_matrix, 's') // specify sequential version
matrix_inverse(some_matrix, 'o') // specify openmp version
```
If a function is called with no additional parameter, it will, somewhat heuristically, decide which version to use on its own, depending on the problem size and the number of CPU cores on the device it's running on.

<br>

Regular assignment operator should be used with caution when dealing with matrices, as it can cause memory leaks. If you want to replace the old values of an already existing matrix, use *matrix_replace*.
```c
#include <mole_math.h>

int main(void) {
	Matrix matrix_a = matrix_random(2,2);
	Matrix matrix_b = matrix_random(2,2);
	Matrix matrix_c = matrix_random(2,2);
	
	matrix_b = matrix_a; // causes memory leak, access to matrix_b.value
	                     // pointer is lost, but the pointer was not freed

	matrix_replace(&matrix_c, matrix_a); // memory associated with matrix_c is properly freed
	                                     // and the contents of matrix_a are safely copied 

	return 0;
}
```
