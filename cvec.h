#ifndef CVEC_H_
#define CVEC_H_

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>

// Error handling convention
// -------------------------
// No function in this library aborts on bad input. Instead:
//   - functions returning a pointer return NULL,
//   - functions returning a float return NAN,
//   - functions that only report success return a `cvec_status`,
//   - `cvec_ndarray_get_index` returns -1.
// "Bad input" means a NULL pointer, incompatible shapes, an index out of
// range or a failed memory allocation.

typedef enum
{
    CVEC_OK = 0,
    CVEC_ERR_NULL = -1,  // a required pointer argument was NULL
    CVEC_ERR_INDEX = -2, // an index was out of range
    CVEC_ERR_ALLOC = -3, // a memory allocation failed
} cvec_status;

typedef struct
{
    size_t ndim;  // num of dims
    int *shape;   // size of each dim
    int *strides; // how to move in memory for each dim
    // TODO: Make `data` customizable depending on some dtype
    float *data; // data in contiguous form
} cvec_NDArray;

// Creates a zero-filled C-contiguous array. Returns NULL if `ndim` < 1, if
// `shape` is NULL or has a negative entry, if the number of elements does not
// fit in an `int` (strides are `int`), or if allocation fails.
cvec_NDArray *cvec_ndarray_create(int ndim, const int *shape);
void cvec_ndarray_free(cvec_NDArray *arr);

// `indices` must have `arr->ndim` entries. Returns -1 if `arr` or `indices`
// is NULL or any index is out of range.
ptrdiff_t cvec_ndarray_get_index(const cvec_NDArray *arr, const int *indices);
// Returns NAN if the index is invalid.
float cvec_ndarray_get(const cvec_NDArray *arr, const int *indices);
cvec_status cvec_ndarray_set(cvec_NDArray *arr, const int *indices, float value);
void cvec_ndarray_print(const cvec_NDArray *arr);

// Returns NULL if the arguments are invalid or allocation fails.
// Same as cvec_ndarray_matmul for 2D inputs
cvec_NDArray *cvec_ndarray_matmul_2d(const cvec_NDArray *a, const cvec_NDArray *b);
cvec_NDArray *cvec_ndarray_matmul(const cvec_NDArray *a, const cvec_NDArray *b);

// Returns NAN if the arguments are invalid or allocation fails.
float cvec_ndarray_euclidean_distance(const cvec_NDArray *a, const cvec_NDArray *b);

#endif // CVEC_H_

#ifdef CVEC_IMPLEMENTATION

cvec_NDArray *cvec_ndarray_create(int ndim, const int *shape)
{
    if (ndim < 1 || !shape)
    {
        return NULL;
    }

    // Strides are `int`, so the element count (and every partial product used
    // to build the strides) must fit in an `int`.
    size_t total = 1;
    for (int i = 0; i < ndim; i++)
    {
        if (shape[i] < 0)
        {
            return NULL;
        }
        if (shape[i] != 0 && total > (size_t)INT_MAX / (size_t)shape[i])
        {
            return NULL;
        }
        total *= (size_t)shape[i];
    }
    if (total > SIZE_MAX / sizeof(float))
    {
        return NULL;
    }

    cvec_NDArray *arr = malloc(sizeof(cvec_NDArray));
    if (!arr)
    {
        return NULL;
    }

    arr->ndim = (size_t)ndim;
    arr->shape = malloc((size_t)ndim * sizeof(int));
    arr->strides = malloc((size_t)ndim * sizeof(int));
    // calloc(0) may return NULL, so always ask for at least one element
    arr->data = calloc(total > 0 ? total : 1, sizeof(float));
    if (!arr->shape || !arr->strides || !arr->data)
    {
        cvec_ndarray_free(arr);
        return NULL;
    }

    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--)
    {
        arr->shape[i] = shape[i];
        arr->strides[i] = stride;
        stride *= shape[i];
    }

    return arr;
}

void cvec_ndarray_free(cvec_NDArray *arr)
{
    if (!arr)
    {
        return;
    }
    free(arr->shape);
    free(arr->strides);
    free(arr->data);
    free(arr);
}

ptrdiff_t cvec_ndarray_get_index(const cvec_NDArray *arr, const int *indices)
{
    if (!arr || !indices)
    {
        return -1;
    }

    ptrdiff_t idx = 0;
    for (size_t i = 0; i < arr->ndim; i++)
    {
        if (indices[i] < 0 || indices[i] >= arr->shape[i])
        {
            return -1;
        }
        idx += (ptrdiff_t)indices[i] * arr->strides[i];
    }
    return idx;
}

float cvec_ndarray_get(const cvec_NDArray *arr, const int *indices)
{
    ptrdiff_t idx = cvec_ndarray_get_index(arr, indices);
    if (idx < 0)
    {
        return NAN;
    }
    return arr->data[idx];
}

cvec_status cvec_ndarray_set(cvec_NDArray *arr, const int *indices, float value)
{
    if (!arr || !indices)
    {
        return CVEC_ERR_NULL;
    }
    ptrdiff_t idx = cvec_ndarray_get_index(arr, indices);
    if (idx < 0)
    {
        return CVEC_ERR_INDEX;
    }
    arr->data[idx] = value;
    return CVEC_OK;
}

cvec_NDArray *cvec_ndarray_matmul_2d(const cvec_NDArray *a, const cvec_NDArray *b)
{
    if (!a || !b || a->ndim != 2 || b->ndim != 2 || a->shape[1] != b->shape[0])
    {
        return NULL;
    }

    int result_shape[2] = {a->shape[0], b->shape[1]};
    cvec_NDArray *result = cvec_ndarray_create(2, result_shape);
    if (!result)
    {
        return NULL;
    }

    for (int i = 0; i < a->shape[0]; i++)
    {
        for (int j = 0; j < b->shape[1]; j++)
        {
            float c_val = 0;
            for (int k = 0; k < a->shape[1]; k++)
            {
                int a_idx = i * a->strides[0] + k * a->strides[1];
                int b_idx = k * b->strides[0] + j * b->strides[1];
                c_val += a->data[a_idx] * b->data[b_idx];
            }
            int c_idx = i * result->strides[0] + j * result->strides[1];
            result->data[c_idx] = c_val;
        }
    }

    return result;
}

static void cvec__matmul_2d_batch(const cvec_NDArray *a, const cvec_NDArray *b, cvec_NDArray *res,
                                  const int *batch_indices, int ndim_batch)
{
    int M = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int N = b->shape[b->ndim - 1];

    int a_offset = 0, b_offset = 0, res_offset = 0;
    for (int d = 0; d < ndim_batch; d++)
    {
        a_offset += batch_indices[d] * a->strides[d];
        b_offset += batch_indices[d] * b->strides[d];
        res_offset += batch_indices[d] * res->strides[d];
    }

    for (int i = 0; i < M; i++)
    {
        int a_row_offset = a_offset + i * a->strides[a->ndim - 2];
        int res_row_offset = res_offset + i * res->strides[res->ndim - 2];

        for (int j = 0; j < N; j++)
        {
            int b_col_offset = b_offset + j * b->strides[b->ndim - 1];
            int c_idx = res_row_offset + j * res->strides[res->ndim - 1];

            float sum = 0;
            const float *a_ptr = a->data + a_row_offset;
            const float *b_ptr = b->data + b_col_offset;
            for (int k = 0; k < K; k++)
            {
                sum += a_ptr[k * a->strides[a->ndim - 1]] * b_ptr[k * b->strides[b->ndim - 2]];
            }

            res->data[c_idx] = sum;
        }
    }
}

// Returns CVEC_ERR_ALLOC if a per-thread buffer could not be allocated
static cvec_status cvec__matmul_nd_iterative(const cvec_NDArray *a, const cvec_NDArray *b, cvec_NDArray *res)
{
    int ndim_batch = (int)a->ndim - 2;

    int total_batches = 1;
    for (int i = 0; i < ndim_batch; i++)
    {
        total_batches *= a->shape[i];
    }

    int failed = 0;

#ifdef CVEC_ALLOW_PARALLEL_OPS
#pragma omp parallel
#endif // CVEC_ALLOW_PARALLEL_OPS
    {
        // multidimensional indices from batch, one buffer per thread so
        // threads don't overwrite each other's indices
        // (calloc(0) may return NULL, so always ask for at least one element)
        int *batch_indices = calloc(ndim_batch > 0 ? (size_t)ndim_batch : 1, sizeof(int));
        if (!batch_indices)
        {
#ifdef CVEC_ALLOW_PARALLEL_OPS
#pragma omp atomic write
#endif // CVEC_ALLOW_PARALLEL_OPS
            failed = 1;
        }

#ifdef CVEC_ALLOW_PARALLEL_OPS
#pragma omp for schedule(static)
#endif // CVEC_ALLOW_PARALLEL_OPS
        for (int batch = 0; batch < total_batches; batch++)
        {
            // can't leave an `omp for` early, so skip the work instead
            if (!batch_indices)
            {
                continue;
            }

            // lineal index to multidimensional index
            int rem = batch;
            for (int d = ndim_batch - 1; d >= 0; d--)
            {
                batch_indices[d] = rem % a->shape[d];
                rem /= a->shape[d];
            }

            cvec__matmul_2d_batch(a, b, res, batch_indices, ndim_batch);
        }

        free(batch_indices);
    }

    return failed ? CVEC_ERR_ALLOC : CVEC_OK;
}

cvec_NDArray *cvec_ndarray_matmul(const cvec_NDArray *a, const cvec_NDArray *b)
{
    if (!a || !b || a->ndim < 2 || a->ndim != b->ndim)
    {
        return NULL;
    }

    int ndim_batch = (int)a->ndim - 2;
    for (int i = 0; i < ndim_batch; i++)
    {
        // cvec__matmul_2d_batch uses the same batch indices for a, b and res
        if (a->shape[i] != b->shape[i])
        {
            return NULL;
        }
    }
    if (a->shape[a->ndim - 1] != b->shape[b->ndim - 2])
    {
        return NULL;
    }

    int result_ndim = (int)a->ndim;

    int *result_shape = malloc((size_t)result_ndim * sizeof(int));
    if (!result_shape)
    {
        return NULL;
    }

    for (int i = 0; i < ndim_batch; i++)
        result_shape[i] = a->shape[i];

    result_shape[result_ndim - 2] = a->shape[a->ndim - 2];
    result_shape[result_ndim - 1] = b->shape[b->ndim - 1];

    cvec_NDArray *res = cvec_ndarray_create(result_ndim, result_shape);
    free(result_shape);
    if (!res)
    {
        return NULL;
    }

    if (cvec__matmul_nd_iterative(a, b, res) != CVEC_OK)
    {
        cvec_ndarray_free(res);
        return NULL;
    }

    return res;
}

// True if `arr` is laid out in C-contiguous (row-major) order.
static int cvec__is_contiguous(const cvec_NDArray *arr)
{
    ptrdiff_t expected = 1;
    for (size_t i = arr->ndim; i-- > 0;)
    {
        // The stride of a dim of size 1 is never used, so it can be anything
        if (arr->shape[i] != 1 && arr->strides[i] != expected)
        {
            return 0;
        }
        expected *= arr->shape[i];
    }
    return 1;
}

float cvec_ndarray_euclidean_distance(const cvec_NDArray *a, const cvec_NDArray *b)
{
    if (!a || !b || a->ndim != b->ndim)
    {
        return NAN;
    }

    size_t ndim = a->ndim;
    size_t total = 1;
    for (size_t i = 0; i < ndim; i++)
    {
        if (a->shape[i] != b->shape[i])
        {
            return NAN;
        }
        total *= (size_t)a->shape[i];
    }

    if (total == 0)
    {
        return 0.0f;
    }
    if (!a->data || !b->data)
    {
        return NAN;
    }

    // Accumulate in double: summing many squares in float loses precision fast
    double sum = 0.0;

    // Fast path: both contiguous, so a single flat loop (easy to vectorize)
    if (cvec__is_contiguous(a) && cvec__is_contiguous(b))
    {
        for (size_t i = 0; i < total; i++)
        {
            double diff = (double)a->data[i] - (double)b->data[i];
            sum += diff * diff;
        }
        return (float)sqrt(sum);
    }

    // General path: walk the N-d index and keep the offsets up to date
    // incrementally instead of recomputing them for every element.
    int *index = (int *)calloc(ndim, sizeof(int));
    if (!index)
    {
        return NAN;
    }

    ptrdiff_t offset_a = 0, offset_b = 0;
    for (size_t count = 0; count < total; count++)
    {
        double diff = (double)a->data[offset_a] - (double)b->data[offset_b];
        sum += diff * diff;

        for (size_t d = ndim; d-- > 0;)
        {
            index[d]++;
            offset_a += a->strides[d];
            offset_b += b->strides[d];
            if (index[d] < a->shape[d])
            {
                break;
            }
            // wrap this dim back to 0 and carry into the previous one
            offset_a -= (ptrdiff_t)a->shape[d] * a->strides[d];
            offset_b -= (ptrdiff_t)b->shape[d] * b->strides[d];
            index[d] = 0;
        }
    }

    free(index);
    return (float)sqrt(sum);
}

static void cvec__print_recursive(const cvec_NDArray *arr, int *indices, size_t dim, int indent)
{
    if (dim == arr->ndim)
    {
        printf("%.2f", cvec_ndarray_get(arr, indices));
        return;
    }

    printf("[");
    for (int i = 0; i < arr->shape[dim]; i++)
    {
        indices[dim] = i;

        if (dim < arr->ndim - 1)
        {
            printf("\n");
            for (int s = 0; s < indent + 2; s++)
            {
                printf(" ");
            }
        }

        cvec__print_recursive(arr, indices, dim + 1, indent + 2);

        if (i != arr->shape[dim] - 1)
            printf(", ");
    }

    if (dim < arr->ndim - 1)
    {
        printf("\n");
        for (int s = 0; s < indent; s++)
            printf(" ");
    }
    printf("]");
}

void cvec_ndarray_print(const cvec_NDArray *arr)
{
    if (!arr)
    {
        printf("(null)\n");
        return;
    }

    int *indices = calloc(arr->ndim > 0 ? arr->ndim : 1, sizeof(int));
    if (!indices)
    {
        return;
    }
    cvec__print_recursive(arr, indices, 0, 0);
    printf(", shape: (");
    for (size_t i = 0; i < arr->ndim; i++)
    {
        if (i == 0)
        {
            printf("%d", arr->shape[i]);
        }
        else
        {
            printf(", %d", arr->shape[i]);
        }
    }
    printf(") -> %zu dims\n", arr->ndim);
    free(indices);
}

#endif // CVEC_IMPLEMENTATION
