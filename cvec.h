#ifndef CVEC_H_
#define CVEC_H_

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <math.h>

// Element type
// ------------
// Arrays store elements of type `cvec_scalar`, `float` by default. Define
// `CVEC_DTYPE` before including this header to use another floating point type:
//
//     #define CVEC_DTYPE double
//     #include "cvec.h"
//
// The type is chosen at compile time and must be the same everywhere the header
// is included. Reductions and distances accumulate in `double` internally.
#ifndef CVEC_DTYPE
#define CVEC_DTYPE float
#endif
typedef CVEC_DTYPE cvec_scalar;

// Error handling convention
// -------------------------
// No function in this library aborts on bad input. Instead:
//   - functions returning a pointer return NULL,
//   - functions returning a `cvec_scalar` return NAN,
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
    size_t ndim;         // num of dims
    int *shape;          // size of each dim
    int *strides;        // how to move in memory for each dim (in elements)
    cvec_scalar *data;   // pointer to the first element
    int owns_data;       // non-zero if `data` must be freed with the array (0 for views)
} cvec_NDArray;

// Creation
// --------
// Creates a zero-filled C-contiguous array. Returns NULL if `ndim` < 1, if
// `shape` is NULL or has a negative entry, if the number of elements does not
// fit in an `int` (strides are `int`), or if allocation fails.
cvec_NDArray *cvec_ndarray_create(int ndim, const int *shape);
cvec_NDArray *cvec_ndarray_zeros(int ndim, const int *shape);
cvec_NDArray *cvec_ndarray_ones(int ndim, const int *shape);
cvec_NDArray *cvec_ndarray_full(int ndim, const int *shape, cvec_scalar value);
// Copies `prod(shape)` elements from `buffer` (row-major) into a new array.
cvec_NDArray *cvec_ndarray_from_buffer(int ndim, const int *shape, const cvec_scalar *buffer);
// Deep copy into a new C-contiguous array (also works for non-contiguous views).
cvec_NDArray *cvec_ndarray_copy(const cvec_NDArray *arr);
// Frees an array. For a view only the view itself is freed, never the data.
void cvec_ndarray_free(cvec_NDArray *arr);

// Number of elements (0 if `arr` is NULL).
size_t cvec_ndarray_size(const cvec_NDArray *arr);

// Element access
// --------------
// `indices` must have `arr->ndim` entries. Returns -1 if `arr` or `indices`
// is NULL or any index is out of range.
ptrdiff_t cvec_ndarray_get_index(const cvec_NDArray *arr, const int *indices);
// Returns NAN if the index is invalid.
cvec_scalar cvec_ndarray_get(const cvec_NDArray *arr, const int *indices);
cvec_status cvec_ndarray_set(cvec_NDArray *arr, const int *indices, cvec_scalar value);
void cvec_ndarray_print(const cvec_NDArray *arr);

// Views
// -----
// Views share their data with the original array (nothing is copied), so
// writes through a view are visible in the original, and a view must not be
// used after the original has been freed. Free each view with
// `cvec_ndarray_free`. All of them return NULL if the arguments are invalid.
//
// Reverses the order of the axes.
cvec_NDArray *cvec_ndarray_transpose(const cvec_NDArray *arr);
// `axes` must be a permutation of 0..ndim-1; new axis i is old axis axes[i].
cvec_NDArray *cvec_ndarray_permute(const cvec_NDArray *arr, const int *axes);
// Only for C-contiguous arrays (use cvec_ndarray_copy first otherwise). At
// most one entry of `shape` may be -1 and it is inferred from the others.
cvec_NDArray *cvec_ndarray_reshape(const cvec_NDArray *arr, int ndim, const int *shape);
// Elements start, start + step, ... < stop along `axis`. Requires
// 0 <= start <= stop <= shape[axis], step >= 1.
cvec_NDArray *cvec_ndarray_slice(const cvec_NDArray *arr, int axis, int start, int stop, int step);

// Elementwise operations
// ----------------------
// Return a new array, or NULL on invalid arguments. The two operands are
// broadcast following the NumPy rules: shapes are aligned on the right and
// each pair of dims must be equal or one of them must be 1.
cvec_NDArray *cvec_ndarray_add(const cvec_NDArray *a, const cvec_NDArray *b);
cvec_NDArray *cvec_ndarray_sub(const cvec_NDArray *a, const cvec_NDArray *b);
cvec_NDArray *cvec_ndarray_mul(const cvec_NDArray *a, const cvec_NDArray *b);
cvec_NDArray *cvec_ndarray_div(const cvec_NDArray *a, const cvec_NDArray *b);

cvec_NDArray *cvec_ndarray_add_scalar(const cvec_NDArray *a, cvec_scalar s);
cvec_NDArray *cvec_ndarray_sub_scalar(const cvec_NDArray *a, cvec_scalar s);
cvec_NDArray *cvec_ndarray_mul_scalar(const cvec_NDArray *a, cvec_scalar s);
cvec_NDArray *cvec_ndarray_div_scalar(const cvec_NDArray *a, cvec_scalar s);

// Matrix multiplication
// ---------------------
// Returns NULL if the arguments are invalid or allocation fails.
// Same as cvec_ndarray_matmul for 2D inputs
cvec_NDArray *cvec_ndarray_matmul_2d(const cvec_NDArray *a, const cvec_NDArray *b);
// Operands need ndim >= 2; the leading (batch) dims are broadcast, so
// (4, 2, 3) x (3, 5) and (4, 1, 2, 3) x (5, 3, 4) are valid.
cvec_NDArray *cvec_ndarray_matmul(const cvec_NDArray *a, const cvec_NDArray *b);

// Reductions
// ----------
// Over all the elements. Return NAN if `arr` is NULL (and, for mean, min and
// max, if it is empty).
cvec_scalar cvec_ndarray_sum(const cvec_NDArray *arr);
cvec_scalar cvec_ndarray_mean(const cvec_NDArray *arr);
cvec_scalar cvec_ndarray_min(const cvec_NDArray *arr);
cvec_scalar cvec_ndarray_max(const cvec_NDArray *arr);
cvec_scalar cvec_ndarray_norm_l1(const cvec_NDArray *arr);
cvec_scalar cvec_ndarray_norm_l2(const cvec_NDArray *arr);

// Distances and dot product
// -------------------------
// `a` and `b` must have the same shape (any number of dims: the arrays are
// treated as flat vectors). Return NAN if the arguments are invalid or
// allocation fails.
cvec_scalar cvec_ndarray_dot(const cvec_NDArray *a, const cvec_NDArray *b);
cvec_scalar cvec_ndarray_euclidean_distance(const cvec_NDArray *a, const cvec_NDArray *b);
cvec_scalar cvec_ndarray_manhattan_distance(const cvec_NDArray *a, const cvec_NDArray *b);
// dot(a, b) / (|a| * |b|); NAN if either vector is empty or has zero norm.
cvec_scalar cvec_ndarray_cosine_similarity(const cvec_NDArray *a, const cvec_NDArray *b);
// 1 - cosine_similarity
cvec_scalar cvec_ndarray_cosine_distance(const cvec_NDArray *a, const cvec_NDArray *b);

#endif // CVEC_H_

#ifdef CVEC_IMPLEMENTATION

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Checks that `shape` describes a valid array and returns its element count.
// Strides are `int`, so the element count (and every partial product used to
// build the strides) must fit in an `int`.
static int cvec__shape_total(int ndim, const int *shape, size_t *total)
{
    if (ndim < 1 || !shape)
    {
        return -1;
    }

    size_t t = 1;
    for (int i = 0; i < ndim; i++)
    {
        if (shape[i] < 0)
        {
            return -1;
        }
        if (shape[i] != 0 && t > (size_t)INT_MAX / (size_t)shape[i])
        {
            return -1;
        }
        t *= (size_t)shape[i];
    }
    if (t > SIZE_MAX / sizeof(cvec_scalar))
    {
        return -1;
    }

    *total = t;
    return 0;
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

// Walks every index of an N-d shape in row-major order, keeping the offsets
// into one or two arrays up to date incrementally (instead of recomputing them
// from the full index for every element).
typedef struct
{
    size_t ndim;
    const int *shape;
    const int *strides_a;
    const int *strides_b; // may be NULL when only one array is walked
    int *index;
    ptrdiff_t offset_a, offset_b;
} cvec__iter;

static int cvec__iter_init(cvec__iter *it, size_t ndim, const int *shape,
                           const int *strides_a, const int *strides_b)
{
    // calloc(0) may return NULL, so always ask for at least one element
    it->index = (int *)calloc(ndim > 0 ? ndim : 1, sizeof(int));
    if (!it->index)
    {
        return -1;
    }
    it->ndim = ndim;
    it->shape = shape;
    it->strides_a = strides_a;
    it->strides_b = strides_b;
    it->offset_a = 0;
    it->offset_b = 0;
    return 0;
}

static void cvec__iter_next(cvec__iter *it)
{
    for (size_t d = it->ndim; d-- > 0;)
    {
        it->index[d]++;
        it->offset_a += it->strides_a[d];
        if (it->strides_b)
        {
            it->offset_b += it->strides_b[d];
        }
        if (it->index[d] < it->shape[d])
        {
            return;
        }
        // wrap this dim back to 0 and carry into the previous one
        it->offset_a -= (ptrdiff_t)it->shape[d] * it->strides_a[d];
        if (it->strides_b)
        {
            it->offset_b -= (ptrdiff_t)it->shape[d] * it->strides_b[d];
        }
        it->index[d] = 0;
    }
}

static void cvec__iter_free(cvec__iter *it)
{
    free(it->index);
    it->index = NULL;
}

// Allocates a view: a new struct with its own shape/strides that points into
// (does not own) `data`. The caller fills in shape and strides.
static cvec_NDArray *cvec__view_new(size_t ndim, cvec_scalar *data)
{
    cvec_NDArray *view = (cvec_NDArray *)malloc(sizeof(cvec_NDArray));
    if (!view)
    {
        return NULL;
    }
    view->ndim = ndim;
    view->shape = (int *)malloc(ndim * sizeof(int));
    view->strides = (int *)malloc(ndim * sizeof(int));
    view->data = data;
    view->owns_data = 0;
    if (!view->shape || !view->strides)
    {
        cvec_ndarray_free(view);
        return NULL;
    }
    return view;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

cvec_NDArray *cvec_ndarray_create(int ndim, const int *shape)
{
    size_t total;
    if (cvec__shape_total(ndim, shape, &total) != 0)
    {
        return NULL;
    }

    cvec_NDArray *arr = (cvec_NDArray *)malloc(sizeof(cvec_NDArray));
    if (!arr)
    {
        return NULL;
    }

    arr->ndim = (size_t)ndim;
    arr->shape = (int *)malloc((size_t)ndim * sizeof(int));
    arr->strides = (int *)malloc((size_t)ndim * sizeof(int));
    // calloc(0) may return NULL, so always ask for at least one element
    arr->data = (cvec_scalar *)calloc(total > 0 ? total : 1, sizeof(cvec_scalar));
    arr->owns_data = 1;
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

cvec_NDArray *cvec_ndarray_zeros(int ndim, const int *shape)
{
    return cvec_ndarray_create(ndim, shape);
}

cvec_NDArray *cvec_ndarray_full(int ndim, const int *shape, cvec_scalar value)
{
    cvec_NDArray *arr = cvec_ndarray_create(ndim, shape);
    if (!arr)
    {
        return NULL;
    }
    size_t total = cvec_ndarray_size(arr);
    for (size_t i = 0; i < total; i++)
    {
        arr->data[i] = value;
    }
    return arr;
}

cvec_NDArray *cvec_ndarray_ones(int ndim, const int *shape)
{
    return cvec_ndarray_full(ndim, shape, (cvec_scalar)1);
}

cvec_NDArray *cvec_ndarray_from_buffer(int ndim, const int *shape, const cvec_scalar *buffer)
{
    if (!buffer)
    {
        return NULL;
    }
    cvec_NDArray *arr = cvec_ndarray_create(ndim, shape);
    if (!arr)
    {
        return NULL;
    }
    memcpy(arr->data, buffer, cvec_ndarray_size(arr) * sizeof(cvec_scalar));
    return arr;
}

cvec_NDArray *cvec_ndarray_copy(const cvec_NDArray *arr)
{
    if (!arr)
    {
        return NULL;
    }
    cvec_NDArray *copy = cvec_ndarray_create((int)arr->ndim, arr->shape);
    if (!copy)
    {
        return NULL;
    }

    size_t total = cvec_ndarray_size(arr);
    if (total == 0)
    {
        return copy;
    }
    if (!arr->data)
    {
        cvec_ndarray_free(copy);
        return NULL;
    }

    if (cvec__is_contiguous(arr))
    {
        memcpy(copy->data, arr->data, total * sizeof(cvec_scalar));
        return copy;
    }

    cvec__iter it;
    if (cvec__iter_init(&it, arr->ndim, arr->shape, arr->strides, NULL) != 0)
    {
        cvec_ndarray_free(copy);
        return NULL;
    }
    for (size_t i = 0; i < total; i++)
    {
        copy->data[i] = arr->data[it.offset_a];
        cvec__iter_next(&it);
    }
    cvec__iter_free(&it);
    return copy;
}

void cvec_ndarray_free(cvec_NDArray *arr)
{
    if (!arr)
    {
        return;
    }
    free(arr->shape);
    free(arr->strides);
    if (arr->owns_data)
    {
        free(arr->data);
    }
    free(arr);
}

size_t cvec_ndarray_size(const cvec_NDArray *arr)
{
    if (!arr)
    {
        return 0;
    }
    size_t total = 1;
    for (size_t i = 0; i < arr->ndim; i++)
    {
        total *= (size_t)arr->shape[i];
    }
    return total;
}

// ---------------------------------------------------------------------------
// Element access
// ---------------------------------------------------------------------------

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

cvec_scalar cvec_ndarray_get(const cvec_NDArray *arr, const int *indices)
{
    ptrdiff_t idx = cvec_ndarray_get_index(arr, indices);
    if (idx < 0)
    {
        return NAN;
    }
    return arr->data[idx];
}

cvec_status cvec_ndarray_set(cvec_NDArray *arr, const int *indices, cvec_scalar value)
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

static void cvec__print_recursive(const cvec_NDArray *arr, int *indices, size_t dim, int indent)
{
    if (dim == arr->ndim)
    {
        printf("%.2f", (double)cvec_ndarray_get(arr, indices));
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

    int *indices = (int *)calloc(arr->ndim > 0 ? arr->ndim : 1, sizeof(int));
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

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

cvec_NDArray *cvec_ndarray_permute(const cvec_NDArray *arr, const int *axes)
{
    if (!arr || !axes || arr->ndim < 1)
    {
        return NULL;
    }

    size_t ndim = arr->ndim;
    char *seen = (char *)calloc(ndim, sizeof(char));
    if (!seen)
    {
        return NULL;
    }
    for (size_t i = 0; i < ndim; i++)
    {
        if (axes[i] < 0 || (size_t)axes[i] >= ndim || seen[axes[i]])
        {
            free(seen);
            return NULL;
        }
        seen[axes[i]] = 1;
    }
    free(seen);

    cvec_NDArray *view = cvec__view_new(ndim, arr->data);
    if (!view)
    {
        return NULL;
    }
    for (size_t i = 0; i < ndim; i++)
    {
        view->shape[i] = arr->shape[axes[i]];
        view->strides[i] = arr->strides[axes[i]];
    }
    return view;
}

cvec_NDArray *cvec_ndarray_transpose(const cvec_NDArray *arr)
{
    if (!arr || arr->ndim < 1)
    {
        return NULL;
    }

    int *axes = (int *)malloc(arr->ndim * sizeof(int));
    if (!axes)
    {
        return NULL;
    }
    for (size_t i = 0; i < arr->ndim; i++)
    {
        axes[i] = (int)(arr->ndim - 1 - i);
    }
    cvec_NDArray *view = cvec_ndarray_permute(arr, axes);
    free(axes);
    return view;
}

cvec_NDArray *cvec_ndarray_reshape(const cvec_NDArray *arr, int ndim, const int *shape)
{
    if (!arr || !shape || ndim < 1 || !cvec__is_contiguous(arr))
    {
        return NULL;
    }

    size_t total = cvec_ndarray_size(arr);

    // resolve the (optional) -1 entry
    int inferred = -1;
    size_t known = 1;
    for (int i = 0; i < ndim; i++)
    {
        if (shape[i] == -1)
        {
            if (inferred != -1)
            {
                return NULL;
            }
            inferred = i;
        }
        else if (shape[i] < 0)
        {
            return NULL;
        }
        else
        {
            known *= (size_t)shape[i];
        }
    }

    int *new_shape = (int *)malloc((size_t)ndim * sizeof(int));
    if (!new_shape)
    {
        return NULL;
    }
    for (int i = 0; i < ndim; i++)
    {
        new_shape[i] = shape[i];
    }
    if (inferred != -1)
    {
        if (known == 0 || total % known != 0 || total / known > (size_t)INT_MAX)
        {
            free(new_shape);
            return NULL;
        }
        new_shape[inferred] = (int)(total / known);
    }

    size_t new_total;
    if (cvec__shape_total(ndim, new_shape, &new_total) != 0 || new_total != total)
    {
        free(new_shape);
        return NULL;
    }

    cvec_NDArray *view = cvec__view_new((size_t)ndim, arr->data);
    if (!view)
    {
        free(new_shape);
        return NULL;
    }
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--)
    {
        view->shape[i] = new_shape[i];
        view->strides[i] = stride;
        stride *= new_shape[i];
    }
    free(new_shape);
    return view;
}

cvec_NDArray *cvec_ndarray_slice(const cvec_NDArray *arr, int axis, int start, int stop, int step)
{
    if (!arr || axis < 0 || (size_t)axis >= arr->ndim || step < 1)
    {
        return NULL;
    }
    if (start < 0 || start > stop || stop > arr->shape[axis])
    {
        return NULL;
    }
    if ((long long)arr->strides[axis] * step > INT_MAX)
    {
        return NULL;
    }

    int len = (stop - start + step - 1) / step;

    // don't move the pointer for an empty slice, it could land past the end
    cvec_scalar *data = arr->data;
    if (len > 0)
    {
        data += (ptrdiff_t)start * arr->strides[axis];
    }

    cvec_NDArray *view = cvec__view_new(arr->ndim, data);
    if (!view)
    {
        return NULL;
    }
    for (size_t i = 0; i < arr->ndim; i++)
    {
        view->shape[i] = arr->shape[i];
        view->strides[i] = arr->strides[i];
    }
    view->shape[axis] = len;
    view->strides[axis] = arr->strides[axis] * step;
    return view;
}

// ---------------------------------------------------------------------------
// Elementwise operations
// ---------------------------------------------------------------------------

typedef enum
{
    CVEC__OP_ADD,
    CVEC__OP_SUB,
    CVEC__OP_MUL,
    CVEC__OP_DIV,
} cvec__op;

static inline cvec_scalar cvec__apply(cvec__op op, cvec_scalar x, cvec_scalar y)
{
    switch (op)
    {
    case CVEC__OP_ADD:
        return x + y;
    case CVEC__OP_SUB:
        return x - y;
    case CVEC__OP_MUL:
        return x * y;
    case CVEC__OP_DIV:
        return x / y;
    }
    return NAN;
}

// Broadcast shape of `a` and `b` (NumPy rules) written to `out_shape`, which
// has `ndim` entries (the max of both ndims). Returns -1 if incompatible.
static int cvec__broadcast_shape(const cvec_NDArray *a, const cvec_NDArray *b, size_t ndim, int *out_shape)
{
    for (size_t i = 0; i < ndim; i++)
    {
        // dims missing on the left count as 1
        int da = i + a->ndim >= ndim ? a->shape[i + a->ndim - ndim] : 1;
        int db = i + b->ndim >= ndim ? b->shape[i + b->ndim - ndim] : 1;
        if (da == db || db == 1)
        {
            out_shape[i] = da;
        }
        else if (da == 1)
        {
            out_shape[i] = db;
        }
        else
        {
            return -1;
        }
    }
    return 0;
}

// Strides to walk `arr` as if it had shape `out_shape`: 0 for the dims that
// are broadcast (missing or of size 1), so the same element is read again.
// `n_out` is the number of dims to fill, aligned on the right.
static void cvec__broadcast_strides(const cvec_NDArray *arr, size_t n_out,
                                    const int *out_shape, int *out_strides)
{
    for (size_t i = 0; i < n_out; i++)
    {
        if (i + arr->ndim < n_out)
        {
            out_strides[i] = 0;
            continue;
        }
        size_t j = i + arr->ndim - n_out;
        out_strides[i] = (arr->shape[j] == 1 && out_shape[i] != 1) ? 0 : arr->strides[j];
    }
}

static cvec_NDArray *cvec__binary_op(const cvec_NDArray *a, const cvec_NDArray *b, cvec__op op)
{
    if (!a || !b || a->ndim < 1 || b->ndim < 1)
    {
        return NULL;
    }

    size_t ndim = a->ndim > b->ndim ? a->ndim : b->ndim;

    // one block for: result shape, strides of a, strides of b
    int *buf = (int *)malloc(3 * ndim * sizeof(int));
    if (!buf)
    {
        return NULL;
    }
    int *shape = buf, *strides_a = buf + ndim, *strides_b = buf + 2 * ndim;

    if (cvec__broadcast_shape(a, b, ndim, shape) != 0)
    {
        free(buf);
        return NULL;
    }
    cvec__broadcast_strides(a, ndim, shape, strides_a);
    cvec__broadcast_strides(b, ndim, shape, strides_b);

    cvec_NDArray *res = cvec_ndarray_create((int)ndim, shape);
    if (!res)
    {
        free(buf);
        return NULL;
    }

    size_t total = cvec_ndarray_size(res);
    if (total > 0)
    {
        cvec__iter it;
        if (!a->data || !b->data || cvec__iter_init(&it, ndim, res->shape, strides_a, strides_b) != 0)
        {
            cvec_ndarray_free(res);
            free(buf);
            return NULL;
        }
        for (size_t i = 0; i < total; i++)
        {
            res->data[i] = cvec__apply(op, a->data[it.offset_a], b->data[it.offset_b]);
            cvec__iter_next(&it);
        }
        cvec__iter_free(&it);
    }

    free(buf);
    return res;
}

static cvec_NDArray *cvec__scalar_op(const cvec_NDArray *a, cvec_scalar s, cvec__op op)
{
    if (!a || a->ndim < 1)
    {
        return NULL;
    }

    cvec_NDArray *res = cvec_ndarray_create((int)a->ndim, a->shape);
    if (!res)
    {
        return NULL;
    }

    size_t total = cvec_ndarray_size(res);
    if (total > 0)
    {
        cvec__iter it;
        if (!a->data || cvec__iter_init(&it, a->ndim, a->shape, a->strides, NULL) != 0)
        {
            cvec_ndarray_free(res);
            return NULL;
        }
        for (size_t i = 0; i < total; i++)
        {
            res->data[i] = cvec__apply(op, a->data[it.offset_a], s);
            cvec__iter_next(&it);
        }
        cvec__iter_free(&it);
    }
    return res;
}

cvec_NDArray *cvec_ndarray_add(const cvec_NDArray *a, const cvec_NDArray *b)
{
    return cvec__binary_op(a, b, CVEC__OP_ADD);
}

cvec_NDArray *cvec_ndarray_sub(const cvec_NDArray *a, const cvec_NDArray *b)
{
    return cvec__binary_op(a, b, CVEC__OP_SUB);
}

cvec_NDArray *cvec_ndarray_mul(const cvec_NDArray *a, const cvec_NDArray *b)
{
    return cvec__binary_op(a, b, CVEC__OP_MUL);
}

cvec_NDArray *cvec_ndarray_div(const cvec_NDArray *a, const cvec_NDArray *b)
{
    return cvec__binary_op(a, b, CVEC__OP_DIV);
}

cvec_NDArray *cvec_ndarray_add_scalar(const cvec_NDArray *a, cvec_scalar s)
{
    return cvec__scalar_op(a, s, CVEC__OP_ADD);
}

cvec_NDArray *cvec_ndarray_sub_scalar(const cvec_NDArray *a, cvec_scalar s)
{
    return cvec__scalar_op(a, s, CVEC__OP_SUB);
}

cvec_NDArray *cvec_ndarray_mul_scalar(const cvec_NDArray *a, cvec_scalar s)
{
    return cvec__scalar_op(a, s, CVEC__OP_MUL);
}

cvec_NDArray *cvec_ndarray_div_scalar(const cvec_NDArray *a, cvec_scalar s)
{
    return cvec__scalar_op(a, s, CVEC__OP_DIV);
}

// ---------------------------------------------------------------------------
// Matrix multiplication
// ---------------------------------------------------------------------------

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
            cvec_scalar c_val = 0;
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

// Multiplies the two matrices found at `a_offset` / `b_offset` (the last two
// dims of each array) and stores the result at `res_offset`.
static void cvec__matmul_2d_batch(const cvec_NDArray *a, const cvec_NDArray *b, cvec_NDArray *res,
                                  ptrdiff_t a_offset, ptrdiff_t b_offset, ptrdiff_t res_offset)
{
    int M = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int N = b->shape[b->ndim - 1];

    for (int i = 0; i < M; i++)
    {
        ptrdiff_t a_row_offset = a_offset + (ptrdiff_t)i * a->strides[a->ndim - 2];
        ptrdiff_t res_row_offset = res_offset + (ptrdiff_t)i * res->strides[res->ndim - 2];

        for (int j = 0; j < N; j++)
        {
            ptrdiff_t b_col_offset = b_offset + (ptrdiff_t)j * b->strides[b->ndim - 1];
            ptrdiff_t c_idx = res_row_offset + (ptrdiff_t)j * res->strides[res->ndim - 1];

            cvec_scalar sum = 0;
            const cvec_scalar *a_ptr = a->data + a_row_offset;
            const cvec_scalar *b_ptr = b->data + b_col_offset;
            for (int k = 0; k < K; k++)
            {
                sum += a_ptr[(ptrdiff_t)k * a->strides[a->ndim - 1]] * b_ptr[(ptrdiff_t)k * b->strides[b->ndim - 2]];
            }

            res->data[c_idx] = sum;
        }
    }
}

// `a_batch_strides` / `b_batch_strides` have one entry per batch dim of `res`
// (0 for the dims that are broadcast).
// Returns CVEC_ERR_ALLOC if a per-thread buffer could not be allocated.
static cvec_status cvec__matmul_nd_iterative(const cvec_NDArray *a, const cvec_NDArray *b, cvec_NDArray *res,
                                             const int *a_batch_strides, const int *b_batch_strides)
{
    int ndim_batch = (int)res->ndim - 2;

    int total_batches = 1;
    for (int i = 0; i < ndim_batch; i++)
    {
        total_batches *= res->shape[i];
    }

    int failed = 0;

#ifdef CVEC_ALLOW_PARALLEL_OPS
#pragma omp parallel
#endif // CVEC_ALLOW_PARALLEL_OPS
    {
        // multidimensional indices from batch, one buffer per thread so
        // threads don't overwrite each other's indices
        // (calloc(0) may return NULL, so always ask for at least one element)
        int *batch_indices = (int *)calloc(ndim_batch > 0 ? (size_t)ndim_batch : 1, sizeof(int));
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
                batch_indices[d] = rem % res->shape[d];
                rem /= res->shape[d];
            }

            ptrdiff_t a_offset = 0, b_offset = 0, res_offset = 0;
            for (int d = 0; d < ndim_batch; d++)
            {
                a_offset += (ptrdiff_t)batch_indices[d] * a_batch_strides[d];
                b_offset += (ptrdiff_t)batch_indices[d] * b_batch_strides[d];
                res_offset += (ptrdiff_t)batch_indices[d] * res->strides[d];
            }

            cvec__matmul_2d_batch(a, b, res, a_offset, b_offset, res_offset);
        }

        free(batch_indices);
    }

    return failed ? CVEC_ERR_ALLOC : CVEC_OK;
}

cvec_NDArray *cvec_ndarray_matmul(const cvec_NDArray *a, const cvec_NDArray *b)
{
    if (!a || !b || a->ndim < 2 || b->ndim < 2)
    {
        return NULL;
    }
    if (a->shape[a->ndim - 1] != b->shape[b->ndim - 2])
    {
        return NULL;
    }

    size_t ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    size_t nb = ndim - 2; // number of batch dims

    // one block for: result shape, batch strides of a, batch strides of b
    int *buf = (int *)malloc((ndim + 2 * nb + 1) * sizeof(int));
    if (!buf)
    {
        return NULL;
    }
    int *result_shape = buf, *a_bs = buf + ndim, *b_bs = buf + ndim + nb;

    // broadcast the batch dims (all the dims but the last two) of a and b
    for (size_t i = 0; i < nb; i++)
    {
        int da = i + a->ndim >= ndim ? a->shape[i + a->ndim - ndim] : 1;
        int db = i + b->ndim >= ndim ? b->shape[i + b->ndim - ndim] : 1;
        if (da == db || db == 1)
        {
            result_shape[i] = da;
        }
        else if (da == 1)
        {
            result_shape[i] = db;
        }
        else
        {
            free(buf);
            return NULL;
        }
    }
    result_shape[ndim - 2] = a->shape[a->ndim - 2];
    result_shape[ndim - 1] = b->shape[b->ndim - 1];

    // batch strides (0 where a dim is broadcast). Batch dim i of the result
    // is dim i - (ndim - a->ndim) of a, or missing if that is negative.
    for (size_t i = 0; i < nb; i++)
    {
        if (i + a->ndim < ndim)
            a_bs[i] = 0;
        else
        {
            size_t j = i + a->ndim - ndim;
            a_bs[i] = (a->shape[j] == 1 && result_shape[i] != 1) ? 0 : a->strides[j];
        }

        if (i + b->ndim < ndim)
            b_bs[i] = 0;
        else
        {
            size_t j = i + b->ndim - ndim;
            b_bs[i] = (b->shape[j] == 1 && result_shape[i] != 1) ? 0 : b->strides[j];
        }
    }

    cvec_NDArray *res = cvec_ndarray_create((int)ndim, result_shape);
    if (!res)
    {
        free(buf);
        return NULL;
    }

    if (cvec_ndarray_size(res) > 0 && cvec__matmul_nd_iterative(a, b, res, a_bs, b_bs) != CVEC_OK)
    {
        cvec_ndarray_free(res);
        res = NULL;
    }

    free(buf);
    return res;
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

typedef enum
{
    CVEC__R1_SUM,
    CVEC__R1_ABS_SUM,
    CVEC__R1_SQ_SUM,
    CVEC__R1_MIN,
    CVEC__R1_MAX,
} cvec__reduce1_mode;

static inline double cvec__reduce1_step(cvec__reduce1_mode mode, double acc, double x)
{
    switch (mode)
    {
    case CVEC__R1_SUM:
        return acc + x;
    case CVEC__R1_ABS_SUM:
        return acc + fabs(x);
    case CVEC__R1_SQ_SUM:
        return acc + x * x;
    case CVEC__R1_MIN:
        return x < acc ? x : acc;
    case CVEC__R1_MAX:
        return x > acc ? x : acc;
    }
    return acc;
}

// Reduces all the elements of `arr` (non empty) into `*out`, accumulating in double.
static cvec_status cvec__reduce1(const cvec_NDArray *arr, cvec__reduce1_mode mode, double *out)
{
    size_t total = cvec_ndarray_size(arr);
    if (!arr || !arr->data)
    {
        return CVEC_ERR_NULL;
    }

    double acc = 0.0;
    if (mode == CVEC__R1_MIN)
        acc = INFINITY;
    else if (mode == CVEC__R1_MAX)
        acc = -INFINITY;

    if (cvec__is_contiguous(arr))
    {
        for (size_t i = 0; i < total; i++)
        {
            acc = cvec__reduce1_step(mode, acc, (double)arr->data[i]);
        }
    }
    else
    {
        cvec__iter it;
        if (cvec__iter_init(&it, arr->ndim, arr->shape, arr->strides, NULL) != 0)
        {
            return CVEC_ERR_ALLOC;
        }
        for (size_t i = 0; i < total; i++)
        {
            acc = cvec__reduce1_step(mode, acc, (double)arr->data[it.offset_a]);
            cvec__iter_next(&it);
        }
        cvec__iter_free(&it);
    }

    *out = acc;
    return CVEC_OK;
}

// `empty_value` is what to return for an empty array
static cvec_scalar cvec__reduce1_scalar(const cvec_NDArray *arr, cvec__reduce1_mode mode, double empty_value)
{
    if (!arr)
    {
        return NAN;
    }
    if (cvec_ndarray_size(arr) == 0)
    {
        return (cvec_scalar)empty_value;
    }
    double out;
    if (cvec__reduce1(arr, mode, &out) != CVEC_OK)
    {
        return NAN;
    }
    return (cvec_scalar)out;
}

cvec_scalar cvec_ndarray_sum(const cvec_NDArray *arr)
{
    return cvec__reduce1_scalar(arr, CVEC__R1_SUM, 0.0);
}

cvec_scalar cvec_ndarray_mean(const cvec_NDArray *arr)
{
    size_t total = cvec_ndarray_size(arr);
    if (!arr || total == 0)
    {
        return NAN;
    }
    double out;
    if (cvec__reduce1(arr, CVEC__R1_SUM, &out) != CVEC_OK)
    {
        return NAN;
    }
    return (cvec_scalar)(out / (double)total);
}

cvec_scalar cvec_ndarray_min(const cvec_NDArray *arr)
{
    return cvec__reduce1_scalar(arr, CVEC__R1_MIN, (double)NAN);
}

cvec_scalar cvec_ndarray_max(const cvec_NDArray *arr)
{
    return cvec__reduce1_scalar(arr, CVEC__R1_MAX, (double)NAN);
}

cvec_scalar cvec_ndarray_norm_l1(const cvec_NDArray *arr)
{
    return cvec__reduce1_scalar(arr, CVEC__R1_ABS_SUM, 0.0);
}

cvec_scalar cvec_ndarray_norm_l2(const cvec_NDArray *arr)
{
    if (!arr)
    {
        return NAN;
    }
    if (cvec_ndarray_size(arr) == 0)
    {
        return 0;
    }
    double out;
    if (cvec__reduce1(arr, CVEC__R1_SQ_SUM, &out) != CVEC_OK)
    {
        return NAN;
    }
    return (cvec_scalar)sqrt(out);
}

// ---------------------------------------------------------------------------
// Distances and dot product
// ---------------------------------------------------------------------------

typedef enum
{
    CVEC__R2_EUCLIDEAN, // acc[0] = sum((x - y)^2)
    CVEC__R2_MANHATTAN, // acc[0] = sum(|x - y|)
    CVEC__R2_DOT,       // acc[0] = sum(x * y)
    CVEC__R2_COSINE,    // acc[0] = sum(x * y), acc[1] = sum(x^2), acc[2] = sum(y^2)
} cvec__reduce2_mode;

static inline void cvec__reduce2_step(cvec__reduce2_mode mode, double x, double y, double *acc)
{
    switch (mode)
    {
    case CVEC__R2_EUCLIDEAN:
        acc[0] += (x - y) * (x - y);
        break;
    case CVEC__R2_MANHATTAN:
        acc[0] += fabs(x - y);
        break;
    case CVEC__R2_DOT:
        acc[0] += x * y;
        break;
    case CVEC__R2_COSINE:
        acc[0] += x * y;
        acc[1] += x * x;
        acc[2] += y * y;
        break;
    }
}

// True if both arrays are non NULL and have the same ndim and shape;
// `*total` is their element count.
static int cvec__same_shape(const cvec_NDArray *a, const cvec_NDArray *b, size_t *total)
{
    if (!a || !b || a->ndim != b->ndim)
    {
        return 0;
    }
    for (size_t i = 0; i < a->ndim; i++)
    {
        if (a->shape[i] != b->shape[i])
        {
            return 0;
        }
    }
    *total = cvec_ndarray_size(a);
    return 1;
}

// Reduces the pairs of elements of `a` and `b` (same shape, non empty) into acc[0..2].
static cvec_status cvec__reduce2(const cvec_NDArray *a, const cvec_NDArray *b, size_t total,
                                 cvec__reduce2_mode mode, double acc[3])
{
    acc[0] = acc[1] = acc[2] = 0.0;
    if (!a->data || !b->data)
    {
        return CVEC_ERR_NULL;
    }

    // Fast path: both contiguous, so a single flat loop (easy to vectorize)
    if (cvec__is_contiguous(a) && cvec__is_contiguous(b))
    {
        for (size_t i = 0; i < total; i++)
        {
            cvec__reduce2_step(mode, (double)a->data[i], (double)b->data[i], acc);
        }
        return CVEC_OK;
    }

    cvec__iter it;
    if (cvec__iter_init(&it, a->ndim, a->shape, a->strides, b->strides) != 0)
    {
        return CVEC_ERR_ALLOC;
    }
    for (size_t i = 0; i < total; i++)
    {
        cvec__reduce2_step(mode, (double)a->data[it.offset_a], (double)b->data[it.offset_b], acc);
        cvec__iter_next(&it);
    }
    cvec__iter_free(&it);
    return CVEC_OK;
}

cvec_scalar cvec_ndarray_dot(const cvec_NDArray *a, const cvec_NDArray *b)
{
    size_t total;
    if (!cvec__same_shape(a, b, &total))
    {
        return NAN;
    }
    if (total == 0)
    {
        return 0;
    }
    double acc[3];
    if (cvec__reduce2(a, b, total, CVEC__R2_DOT, acc) != CVEC_OK)
    {
        return NAN;
    }
    return (cvec_scalar)acc[0];
}

cvec_scalar cvec_ndarray_euclidean_distance(const cvec_NDArray *a, const cvec_NDArray *b)
{
    size_t total;
    if (!cvec__same_shape(a, b, &total))
    {
        return NAN;
    }
    if (total == 0)
    {
        return 0;
    }
    double acc[3];
    if (cvec__reduce2(a, b, total, CVEC__R2_EUCLIDEAN, acc) != CVEC_OK)
    {
        return NAN;
    }
    return (cvec_scalar)sqrt(acc[0]);
}

cvec_scalar cvec_ndarray_manhattan_distance(const cvec_NDArray *a, const cvec_NDArray *b)
{
    size_t total;
    if (!cvec__same_shape(a, b, &total))
    {
        return NAN;
    }
    if (total == 0)
    {
        return 0;
    }
    double acc[3];
    if (cvec__reduce2(a, b, total, CVEC__R2_MANHATTAN, acc) != CVEC_OK)
    {
        return NAN;
    }
    return (cvec_scalar)acc[0];
}

cvec_scalar cvec_ndarray_cosine_similarity(const cvec_NDArray *a, const cvec_NDArray *b)
{
    size_t total;
    if (!cvec__same_shape(a, b, &total) || total == 0)
    {
        return NAN;
    }
    double acc[3];
    if (cvec__reduce2(a, b, total, CVEC__R2_COSINE, acc) != CVEC_OK)
    {
        return NAN;
    }
    if (acc[1] == 0.0 || acc[2] == 0.0)
    {
        return NAN;
    }
    return (cvec_scalar)(acc[0] / (sqrt(acc[1]) * sqrt(acc[2])));
}

cvec_scalar cvec_ndarray_cosine_distance(const cvec_NDArray *a, const cvec_NDArray *b)
{
    return (cvec_scalar)1 - cvec_ndarray_cosine_similarity(a, b);
}

#endif // CVEC_IMPLEMENTATION
