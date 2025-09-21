# CVEC

A simple **header-only library** for vector and matrix operations in C.

## Installation

Just copy `cvec.h` into your project and include it.  
There is no separate `.c` file — everything is in the header.

## Usage

To use this library work like a [nothings/stb](https://github.com/nothings/stb) lib, you must define the implementation if necessary. Additionally, you have the option to allow the library to make parallel operations with the macro `CVEC_ALLOW_PARALLEL_OPS`.

## Configuration macros

| Macro                     | Description                                                           |
| ------------------------- | --------------------------------------------------------------------- |
| `CVEC_IMPLEMENTATION`     | Must be defined in **one** source file to compile the implementation. |
| `CVEC_ALLOW_PARALLEL_OPS` | Enables parallel operations (requires OpenMP).                        |

## Example

```c
#define CVEC_IMPLEMENTATION
#define CVEC_ALLOW_PARALLEL_OPS
#include "cvec.h"

int main()
{
    int shape_a[3] = {2, 2, 3};
    int shape_b[3] = {2, 3, 4};

    NDArray *a = ndarray_create(3, shape_a);
    NDArray *b = ndarray_create(3, shape_b);

    for (int batch = 0; batch < 2; batch++)
        for (int i = 0; i < 2; i++)
            for (int k = 0; k < 3; k++)
                a->data[batch * 6 + i * 3 + k] = batch + i + k;

    printf("a = ");
    ndarray_print(a);
    printf("\n");

    for (int batch = 0; batch < 2; batch++)
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < 4; j++)
                b->data[batch * 12 + k * 4 + j] = batch + k + j;

    printf("b = ");
    ndarray_print(b);
    printf("\n");

    NDArray *res = ndarray_matmul(a, b);

    printf("result = ");
    ndarray_print(res);

    ndarray_free(a);
    ndarray_free(b);
    ndarray_free(res);

    return 0;
}
```

Or just run:

```shell
make run
```

and this will compile the `main.c` example.

## Running tests

```bash
make test
```
