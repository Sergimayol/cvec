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
| `CVEC_DTYPE`              | Element type, `float` by default (e.g. `#define CVEC_DTYPE double`).  |

## Features

| Area             | Functions                                                                                          |
| ---------------- | -------------------------------------------------------------------------------------------------- |
| Creation         | `create`, `zeros`, `ones`, `full`, `from_buffer`, `copy`, `free`                                   |
| Element access   | `get`, `set`, `get_index`, `size`, `print`                                                         |
| Views (no copy)  | `transpose`, `permute`, `reshape`, `slice`                                                         |
| Elementwise      | `add`, `sub`, `mul`, `div` (with broadcasting) and `add_scalar`, `sub_scalar`, `mul_scalar`, `div_scalar` |
| Matrix product   | `matmul_2d`, `matmul` (batched, batch dims are broadcast)                                          |
| Reductions       | `sum`, `mean`, `min`, `max`, `norm_l1`, `norm_l2`, and per axis: `sum_axis`, `mean_axis`, `min_axis`, `max_axis` |
| Vector functions | `dot`, `euclidean_distance`, `manhattan_distance`, `cosine_similarity`, `cosine_distance`          |

All functions are prefixed with `cvec_ndarray_`. Views share their data with the original array: free them with `cvec_ndarray_free` (it never frees data a view does not own) and don't use them after the original has been freed.

## Naming

Everything the library exposes is prefixed with `cvec_` (types, functions) or `CVEC_` (macros and enum values), e.g. `cvec_NDArray`, `cvec_ndarray_create`, `CVEC_OK`. Internal helpers use `cvec__` and are `static`.

## Error handling

Functions never abort on bad input (NULL pointers, incompatible shapes, indices out of range, failed allocations). Instead:

| Return type        | On error                                                         |
| ------------------ | ---------------------------------------------------------------- |
| pointer            | `NULL` (`cvec_ndarray_create`, `cvec_ndarray_matmul`, ...)       |
| `float`            | `NAN` (`cvec_ndarray_get`, `cvec_ndarray_euclidean_distance`)    |
| `cvec_status`      | `CVEC_ERR_NULL`, `CVEC_ERR_INDEX` or `CVEC_ERR_ALLOC`            |
| `ptrdiff_t` index  | `-1` (`cvec_ndarray_get_index`)                                  |

## Example

```c
#define CVEC_IMPLEMENTATION
#define CVEC_ALLOW_PARALLEL_OPS
#include "cvec.h"

int main()
{
    int shape_a[3] = {2, 2, 3};
    int shape_b[3] = {2, 3, 4};

    cvec_NDArray *a = cvec_ndarray_create(3, shape_a);
    cvec_NDArray *b = cvec_ndarray_create(3, shape_b);

    for (int batch = 0; batch < 2; batch++)
        for (int i = 0; i < 2; i++)
            for (int k = 0; k < 3; k++)
                a->data[batch * 6 + i * 3 + k] = batch + i + k;

    printf("a = ");
    cvec_ndarray_print(a);
    printf("\n");

    for (int batch = 0; batch < 2; batch++)
        for (int k = 0; k < 3; k++)
            for (int j = 0; j < 4; j++)
                b->data[batch * 12 + k * 4 + j] = batch + k + j;

    printf("b = ");
    cvec_ndarray_print(b);
    printf("\n");

    cvec_NDArray *res = cvec_ndarray_matmul(a, b);

    printf("result = ");
    cvec_ndarray_print(res);

    cvec_ndarray_free(a);
    cvec_ndarray_free(b);
    cvec_ndarray_free(res);

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
