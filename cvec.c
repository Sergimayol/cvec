#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

typedef struct
{
    int ndim;     // num of dims
    int *shape;   // size of each dim
    int *strides; // how to move in memory for each dim
    float *data;  // data in contiguous form
} NDArray;

NDArray *ndarray_create(int ndim, int *shape)
{
    NDArray *arr = malloc(sizeof(NDArray));
    arr->ndim = ndim;

    arr->shape = malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; i++)
    {
        arr->shape[i] = shape[i];
    }

    arr->strides = malloc(ndim * sizeof(int));
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--)
    {
        arr->strides[i] = stride;
        stride *= shape[i];
    }

    arr->data = calloc(stride, sizeof(float));

    return arr;
}

void ndarray_free(NDArray *arr)
{
    free(arr->shape);
    free(arr->strides);
    free(arr->data);
    free(arr);
}

int ndarray_get_index(NDArray *arr, int *indices)
{
    int idx = 0;
    for (int i = 0; i < arr->ndim; i++)
    {
        idx += indices[i] * arr->strides[i];
    }
    return idx;
}

float ndarray_get(NDArray *arr, int *indices)
{
    int idx = ndarray_get_index(arr, indices);
    return arr->data[idx];
}

void ndarray_set(NDArray *arr, int *indices, float value)
{
    int idx = ndarray_get_index(arr, indices);
    arr->data[idx] = value;
}

void ndarray_print_pretty_recursive(NDArray *arr, int *indices, int dim, int indent)
{
    if (dim == arr->ndim)
    {
        printf("%.2f", ndarray_get(arr, indices));
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

        ndarray_print_pretty_recursive(arr, indices, dim + 1, indent + 2);

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

void ndarray_print(NDArray *arr)
{
    int *indices = calloc(arr->ndim, sizeof(int));
    ndarray_print_pretty_recursive(arr, indices, 0, 0);
    printf(", shape: (");
    for (int i = 0; i < arr->ndim; i++)
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
    printf(") -> %d dims\n", arr->ndim);
    free(indices);
}

NDArray *ndarray_matmul_2d(NDArray *a, NDArray *b)
{
    assert(a->ndim == 2);
    assert(b->ndim == 2);
    assert(a->shape[1] == b->shape[0]);

    int result_shape[2] = {a->shape[0], b->shape[1]};
    NDArray *result = ndarray_create(2, result_shape);

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

void matmul_2d_batch(NDArray *a, NDArray *b, NDArray *res, int *batch_indices, int ndim_batch)
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
            for (int k = 0; k < K; k++)
            {
                int a_idx = a_row_offset + k * a->strides[a->ndim - 1];
                int b_idx = b_col_offset + k * b->strides[b->ndim - 2];
                sum += a->data[a_idx] * b->data[b_idx];
            }

            res->data[c_idx] = sum;
        }
    }
}

void matmul_nd_recursive(NDArray *a, NDArray *b, NDArray *res, int *batch_indices, int dim, int ndim_batch)
{
    if (dim == ndim_batch)
    {
        matmul_2d_batch(a, b, res, batch_indices, ndim_batch);
        return;
    }

    for (int i = 0; i < a->shape[dim]; i++)
    {
        batch_indices[dim] = i;
        matmul_nd_recursive(a, b, res, batch_indices, dim + 1, ndim_batch);
    }
}

NDArray *ndarray_matmul_nd(NDArray *a, NDArray *b)
{
    assert(a->ndim >= 2 && b->ndim >= 2);
    assert(a->shape[a->ndim - 1] == b->shape[b->ndim - 2]);

    int ndim_batch = a->ndim - 2;
    int result_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    int *result_shape = malloc(result_ndim * sizeof(int));

    for (int i = 0; i < ndim_batch; i++)
    {
        result_shape[i] = a->shape[i];
    }

    result_shape[result_ndim - 2] = a->shape[a->ndim - 2];
    result_shape[result_ndim - 1] = b->shape[b->ndim - 1];

    NDArray *res = ndarray_create(result_ndim, result_shape);
    free(result_shape);

    int *batch_indices = calloc(ndim_batch, sizeof(int));
    matmul_nd_recursive(a, b, res, batch_indices, 0, ndim_batch);
    free(batch_indices);

    return res;
}

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

    NDArray *res = ndarray_matmul_nd(a, b);

    printf("result = ");
    ndarray_print(res);

    ndarray_free(a);
    ndarray_free(b);
    ndarray_free(res);

    return 0;
}
