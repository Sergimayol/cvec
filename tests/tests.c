#define CVEC_IMPLEMENTATION
#define CVEC_ALLOW_PARALLEL_OPS
#include "cvec.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CHECKMARK "[OK]"
#define CROSS "[X]"

#define CHECK_EQ(actual, expected)                                                 \
    do                                                                             \
    {                                                                              \
        if ((actual) != (expected))                                                \
        {                                                                          \
            fprintf(stderr, CROSS " %s:%d: CHECK_EQ failed: %s=%d, expected %d\n", \
                    __FILE__, __LINE__, #actual, (actual), (expected));            \
            exit(1);                                                               \
        }                                                                          \
    } while (0)

#define CHECK_FLOAT_NEAR(actual, expected, tol)                                                       \
    do                                                                                                \
    {                                                                                                 \
        if (fabs((actual) - (expected)) > (tol))                                                      \
        {                                                                                             \
            fprintf(stderr, CROSS " %s:%d: CHECK_FLOAT_NEAR failed: %s=%.6f, expected %.6f ± %.6f\n", \
                    __FILE__, __LINE__, #actual, (actual), (expected), (tol));                        \
            exit(1);                                                                                  \
        }                                                                                             \
    } while (0)

#define PASS(msg) \
    printf(CHECKMARK " %s passed\n", msg)

#define EPS (1e-6)

void test_create_and_set()
{
    int shape[2] = {2, 2};
    cvec_NDArray *a = cvec_ndarray_create(2, shape);

    int idx1[2] = {0, 0};
    int idx2[2] = {1, 1};

    cvec_ndarray_set(a, idx1, 42.0f);
    cvec_ndarray_set(a, idx2, 3.14f);

    CHECK_FLOAT_NEAR(cvec_ndarray_get(a, idx1), 42.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_get(a, idx2), 3.14f, EPS);

    cvec_ndarray_free(a);
    PASS("test_create_and_set");
}

void test_matmul_2d()
{
    int shapeA[2] = {2, 3};
    int shapeB[2] = {3, 2};
    cvec_NDArray *A = cvec_ndarray_create(2, shapeA);
    cvec_NDArray *B = cvec_ndarray_create(2, shapeB);

    int idx[2];
    float valsA[6] = {1, 2, 3, 4, 5, 6};
    float valsB[6] = {7, 8, 9, 10, 11, 12};

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            idx[0] = i;
            idx[1] = j;
            cvec_ndarray_set(A, idx, valsA[i * 3 + j]);
        }
    }
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            idx[0] = i;
            idx[1] = j;
            cvec_ndarray_set(B, idx, valsB[i * 2 + j]);
        }
    }

    cvec_NDArray *C = cvec_ndarray_matmul_2d(A, B);

    int idxC[2] = {0, 0};
    CHECK_FLOAT_NEAR(cvec_ndarray_get(C, idxC), 58.0f, EPS);

    idxC[0] = 0;
    idxC[1] = 1;
    CHECK_FLOAT_NEAR(cvec_ndarray_get(C, idxC), 64.0f, EPS);

    idxC[0] = 1;
    idxC[1] = 0;
    CHECK_FLOAT_NEAR(cvec_ndarray_get(C, idxC), 139.0f, EPS);

    idxC[0] = 1;
    idxC[1] = 1;
    CHECK_FLOAT_NEAR(cvec_ndarray_get(C, idxC), 154.0f, EPS);

    PASS("test_matmul_2d");

    cvec_ndarray_free(A);
    cvec_ndarray_free(B);
    cvec_ndarray_free(C);
}

void test_matmul_batched()
{
    // Many batches so several threads run at once: batch `n` of A is filled
    // with (n + 1) and B with ones, so every entry of C[n] is 3 * (n + 1).
    // With a shared index buffer between threads the batches get mixed up.
    const int batches = 256;
    int shape_a[3] = {batches, 2, 3};
    int shape_b[3] = {batches, 3, 2};

    cvec_NDArray *A = cvec_ndarray_create(3, shape_a);
    cvec_NDArray *B = cvec_ndarray_create(3, shape_b);

    for (int n = 0; n < batches; n++)
    {
        for (int i = 0; i < 6; i++)
        {
            A->data[n * 6 + i] = (float)(n + 1);
            B->data[n * 6 + i] = 1.0f;
        }
    }

    cvec_NDArray *C = cvec_ndarray_matmul(A, B);

    for (int n = 0; n < batches; n++)
    {
        for (int i = 0; i < 4; i++)
        {
            CHECK_FLOAT_NEAR(C->data[n * 4 + i], 3.0f * (n + 1), EPS);
        }
    }

    PASS("test_matmul_batched");

    cvec_ndarray_free(A);
    cvec_ndarray_free(B);
    cvec_ndarray_free(C);
}

void test_matmul_invalid()
{
    int shape_a[3] = {4, 2, 3};
    int shape_b[3] = {5, 3, 2}; // different batch dim
    int shape_c[3] = {4, 2, 2}; // inner dims 3 vs 2
    int shape_d[2] = {2, 2};    // fewer dims, but inner dims 3 vs 2

    cvec_NDArray *A = cvec_ndarray_create(3, shape_a);
    cvec_NDArray *B = cvec_ndarray_create(3, shape_b);
    cvec_NDArray *C = cvec_ndarray_create(3, shape_c);
    cvec_NDArray *D = cvec_ndarray_create(2, shape_d);

    if (cvec_ndarray_matmul(A, B) || cvec_ndarray_matmul(A, C) || cvec_ndarray_matmul(A, D) || cvec_ndarray_matmul(NULL, A))
    {
        fprintf(stderr, CROSS " invalid matmul should return NULL\n");
        exit(1);
    }

    PASS("test_matmul_invalid");

    cvec_ndarray_free(A);
    cvec_ndarray_free(B);
    cvec_ndarray_free(C);
    cvec_ndarray_free(D);
}

void test_create_invalid()
{
    int ok_shape[2] = {2, 3};
    int neg_shape[2] = {2, -1};
    int huge_shape[3] = {100000, 100000, 100000}; // overflows int
    int empty_shape[2] = {0, 3};

    if (cvec_ndarray_create(0, ok_shape) || cvec_ndarray_create(-1, ok_shape))
    {
        fprintf(stderr, CROSS " ndim < 1 should return NULL\n");
        exit(1);
    }
    if (cvec_ndarray_create(2, NULL))
    {
        fprintf(stderr, CROSS " NULL shape should return NULL\n");
        exit(1);
    }
    if (cvec_ndarray_create(2, neg_shape))
    {
        fprintf(stderr, CROSS " negative shape should return NULL\n");
        exit(1);
    }
    if (cvec_ndarray_create(3, huge_shape))
    {
        fprintf(stderr, CROSS " overflowing shape should return NULL\n");
        exit(1);
    }

    // empty arrays are valid
    cvec_NDArray *empty = cvec_ndarray_create(2, empty_shape);
    if (!empty)
    {
        fprintf(stderr, CROSS " empty array should be created\n");
        exit(1);
    }
    cvec_ndarray_free(empty);

    PASS("test_create_invalid");
}

void test_get_set_out_of_range()
{
    int shape[2] = {2, 3};
    cvec_NDArray *arr = cvec_ndarray_create(2, shape);

    int ok[2] = {1, 2};
    int too_big[2] = {2, 0};
    int negative[2] = {0, -1};

    CHECK_EQ(cvec_ndarray_set(arr, ok, 7.0f), CVEC_OK);
    CHECK_FLOAT_NEAR(cvec_ndarray_get(arr, ok), 7.0f, EPS);
    CHECK_EQ(cvec_ndarray_set(arr, too_big, 1.0f), CVEC_ERR_INDEX);
    CHECK_EQ(cvec_ndarray_set(arr, negative, 1.0f), CVEC_ERR_INDEX);
    CHECK_EQ(cvec_ndarray_set(NULL, ok, 1.0f), CVEC_ERR_NULL);
    CHECK_EQ(cvec_ndarray_set(arr, NULL, 1.0f), CVEC_ERR_NULL);

    if (!isnan(cvec_ndarray_get(arr, too_big)) || !isnan(cvec_ndarray_get(arr, negative)) ||
        !isnan(cvec_ndarray_get(NULL, ok)))
    {
        fprintf(stderr, CROSS " get with invalid index should return NAN\n");
        exit(1);
    }
    CHECK_EQ((int)cvec_ndarray_get_index(arr, too_big), -1);

    PASS("test_get_set_out_of_range");

    cvec_ndarray_free(arr);
}

void test_matmul_2d_invalid()
{
    int shape_a[2] = {2, 3};
    int shape_b[2] = {2, 3}; // inner dims 3 vs 2
    int shape_c[3] = {1, 3, 2};

    cvec_NDArray *A = cvec_ndarray_create(2, shape_a);
    cvec_NDArray *B = cvec_ndarray_create(2, shape_b);
    cvec_NDArray *C = cvec_ndarray_create(3, shape_c);

    if (cvec_ndarray_matmul_2d(A, B) || cvec_ndarray_matmul_2d(A, C) || cvec_ndarray_matmul_2d(NULL, A))
    {
        fprintf(stderr, CROSS " invalid matmul_2d should return NULL\n");
        exit(1);
    }

    PASS("test_matmul_2d_invalid");

    cvec_ndarray_free(A);
    cvec_ndarray_free(B);
    cvec_ndarray_free(C);
}

void test_ndarray_euclidean_distance_1d()
{
    int shape[1] = {3};
    int strides[1] = {1};

    cvec_scalar data_a[3] = {1.0f, 2.0f, 3.0f};
    cvec_scalar data_b[3] = {4.0f, 6.0f, 3.0f};

    cvec_NDArray a = {1, shape, strides, data_a, 0};
    cvec_NDArray b = {1, shape, strides, data_b, 0};

    float dist = cvec_ndarray_euclidean_distance(&a, &b);

    CHECK_FLOAT_NEAR(dist, 5.0f, EPS);

    PASS("test_ndarray_euclidean_distance_1d");
}

void test_ndarray_euclidean_distance_2d()
{
    int shape[2] = {2, 2};
    int strides[2] = {2, 1};

    cvec_scalar data_a[4] = {1, 2,
                       3, 4};

    cvec_scalar data_b[4] = {1, 0,
                       3, 0};

    cvec_NDArray a = {2, shape, strides, data_a, 0};
    cvec_NDArray b = {2, shape, strides, data_b, 0};

    float dist = cvec_ndarray_euclidean_distance(&a, &b);

    CHECK_FLOAT_NEAR(dist, sqrtf(20.0f), EPS);

    PASS("test_ndarray_euclidean_distance_2d");
}

void test_ndarray_euclidean_distance_3d()
{
    int shape[3] = {2, 2, 2}; // 3D array: 2x2x2 (8)

    cvec_NDArray *a = cvec_ndarray_create(3, shape);
    cvec_NDArray *b = cvec_ndarray_create(3, shape);

    // a = [ [ [1,2], [3,4] ],
    //       [ [5,6], [7,8] ] ]
    for (int i = 0; i < 8; i++)
    {
        a->data[i] = (float)(i + 1);
    }

    for (int i = 0; i < 8; i++)
    {
        b->data[i] = (float)(i + 2);
    }

    float dist = cvec_ndarray_euclidean_distance(a, b);

    // sqrt( sum( (1)^2 * 8 ) ) = sqrt(8)
    float expected = sqrtf(8.0f);

    CHECK_FLOAT_NEAR(dist, expected, EPS);

    PASS("test_ndarray_euclidean_distance_3d");

    cvec_ndarray_free(a);
    cvec_ndarray_free(b);
}

void test_ndarray_euclidean_distance_non_contiguous()
{
    // a is the transpose view of a 2x3 row-major buffer (shape 3x2, strides 1,3)
    // a = [[1, 4],
    //      [2, 5],
    //      [3, 6]]
    int shape[2] = {3, 2};
    int strides_a[2] = {1, 3};
    int strides_b[2] = {2, 1};

    cvec_scalar data_a[6] = {1, 2, 3, 4, 5, 6};
    cvec_scalar data_b[6] = {1, 4,
                       2, 5,
                       3, 9};

    cvec_NDArray a = {2, shape, strides_a, data_a, 0};
    cvec_NDArray b = {2, shape, strides_b, data_b, 0};

    // only the last element differs: 6 vs 9
    float dist = cvec_ndarray_euclidean_distance(&a, &b);
    CHECK_FLOAT_NEAR(dist, 3.0f, EPS);

    // both non-contiguous
    dist = cvec_ndarray_euclidean_distance(&a, &a);
    CHECK_FLOAT_NEAR(dist, 0.0f, EPS);

    PASS("test_ndarray_euclidean_distance_non_contiguous");
}

void test_ndarray_euclidean_distance_invalid()
{
    int shape_a[2] = {2, 2};
    int shape_b[2] = {2, 3};
    int shape_c[1] = {4};
    int strides[2] = {2, 1};
    int strides_c[1] = {1};
    cvec_scalar data[6] = {0};

    cvec_NDArray a = {2, shape_a, strides, data, 0};
    cvec_NDArray b = {2, shape_b, strides, data, 0};
    cvec_NDArray c = {1, shape_c, strides_c, data, 0};

    if (!isnan(cvec_ndarray_euclidean_distance(&a, &b)))
    {
        fprintf(stderr, CROSS " different shapes should return NAN\n");
        exit(1);
    }
    if (!isnan(cvec_ndarray_euclidean_distance(&a, &c)))
    {
        fprintf(stderr, CROSS " different ndim should return NAN\n");
        exit(1);
    }
    if (!isnan(cvec_ndarray_euclidean_distance(NULL, &a)))
    {
        fprintf(stderr, CROSS " NULL should return NAN\n");
        exit(1);
    }

    PASS("test_ndarray_euclidean_distance_invalid");
}

void test_ndarray_euclidean_distance_empty()
{
    int shape[2] = {0, 3};
    int strides[2] = {3, 1};
    cvec_NDArray a = {2, shape, strides, NULL, 0};
    cvec_NDArray b = {2, shape, strides, NULL, 0};

    CHECK_FLOAT_NEAR(cvec_ndarray_euclidean_distance(&a, &b), 0.0f, EPS);

    PASS("test_ndarray_euclidean_distance_empty");
}

#define CHECK_NULL(ptr)                                                     \
    do                                                                      \
    {                                                                       \
        if ((ptr) != NULL)                                                  \
        {                                                                   \
            fprintf(stderr, CROSS " %s:%d: %s should be NULL\n",            \
                    __FILE__, __LINE__, #ptr);                              \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#define CHECK_NOT_NULL(ptr)                                                 \
    do                                                                      \
    {                                                                       \
        if ((ptr) == NULL)                                                  \
        {                                                                   \
            fprintf(stderr, CROSS " %s:%d: %s should not be NULL\n",        \
                    __FILE__, __LINE__, #ptr);                              \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#define CHECK_NAN(x)                                                        \
    do                                                                      \
    {                                                                       \
        if (!isnan(x))                                                      \
        {                                                                   \
            fprintf(stderr, CROSS " %s:%d: %s should be NAN\n",             \
                    __FILE__, __LINE__, #x);                                \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// checks that the (contiguous or not) array holds `expected` in row-major order
static void check_values(const cvec_NDArray *arr, const cvec_scalar *expected, size_t n, int line)
{
    CHECK_NOT_NULL(arr);
    CHECK_EQ((int)cvec_ndarray_size(arr), (int)n);
    cvec_NDArray *flat = cvec_ndarray_copy(arr);
    CHECK_NOT_NULL(flat);
    for (size_t i = 0; i < n; i++)
    {
        if (fabs(flat->data[i] - expected[i]) > EPS)
        {
            fprintf(stderr, CROSS " tests.c:%d: element %zu is %.6f, expected %.6f\n",
                    line, i, (double)flat->data[i], (double)expected[i]);
            exit(1);
        }
    }
    cvec_ndarray_free(flat);
}
#define CHECK_VALUES(arr, ...)                                 \
    do                                                         \
    {                                                          \
        const cvec_scalar exp_[] = {__VA_ARGS__};                    \
        check_values((arr), exp_, sizeof(exp_) / sizeof(exp_[0]), __LINE__); \
    } while (0)

void test_constructors()
{
    int shape[2] = {2, 3};

    cvec_NDArray *z = cvec_ndarray_zeros(2, shape);
    CHECK_VALUES(z, 0, 0, 0, 0, 0, 0);

    cvec_NDArray *o = cvec_ndarray_ones(2, shape);
    CHECK_VALUES(o, 1, 1, 1, 1, 1, 1);

    cvec_NDArray *f = cvec_ndarray_full(2, shape, 2.5f);
    CHECK_VALUES(f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f);

    cvec_scalar buf[6] = {1, 2, 3, 4, 5, 6};
    cvec_NDArray *b = cvec_ndarray_from_buffer(2, shape, buf);
    CHECK_VALUES(b, 1, 2, 3, 4, 5, 6);
    buf[0] = 99; // the array owns a copy
    CHECK_FLOAT_NEAR(b->data[0], 1.0f, EPS);
    CHECK_NULL(cvec_ndarray_from_buffer(2, shape, NULL));
    CHECK_NULL(cvec_ndarray_ones(0, shape));

    // copy of a strided view is contiguous and independent of the original
    cvec_NDArray *t = cvec_ndarray_transpose(b);
    cvec_NDArray *c = cvec_ndarray_copy(t);
    CHECK_VALUES(c, 1, 4, 2, 5, 3, 6);
    CHECK_EQ(c->owns_data, 1);
    c->data[0] = 100;
    CHECK_FLOAT_NEAR(b->data[0], 1.0f, EPS);
    CHECK_NULL(cvec_ndarray_copy(NULL));

    PASS("test_constructors");

    cvec_ndarray_free(z);
    cvec_ndarray_free(o);
    cvec_ndarray_free(f);
    cvec_ndarray_free(b);
    cvec_ndarray_free(t);
    cvec_ndarray_free(c);
}

void test_views()
{
    int shape[2] = {2, 3};
    cvec_scalar buf[6] = {1, 2, 3, 4, 5, 6};
    cvec_NDArray *a = cvec_ndarray_from_buffer(2, shape, buf);

    // transpose: shape (3, 2), shares data
    cvec_NDArray *t = cvec_ndarray_transpose(a);
    CHECK_NOT_NULL(t);
    CHECK_EQ(t->shape[0], 3);
    CHECK_EQ(t->shape[1], 2);
    CHECK_EQ(t->owns_data, 0);
    int idx[2] = {2, 1};
    CHECK_FLOAT_NEAR(cvec_ndarray_get(t, idx), 6.0f, EPS);
    CHECK_EQ(cvec_ndarray_set(t, idx, 60.0f), CVEC_OK);
    int idx_a[2] = {1, 2};
    CHECK_FLOAT_NEAR(cvec_ndarray_get(a, idx_a), 60.0f, EPS);
    cvec_ndarray_set(t, idx, 6.0f);

    // permute
    int axes[2] = {1, 0};
    cvec_NDArray *p = cvec_ndarray_permute(a, axes);
    CHECK_VALUES(p, 1, 4, 2, 5, 3, 6);
    int repeated[2] = {0, 0};
    int out_of_range[2] = {0, 2};
    CHECK_NULL(cvec_ndarray_permute(a, repeated));
    CHECK_NULL(cvec_ndarray_permute(a, out_of_range));
    CHECK_NULL(cvec_ndarray_permute(a, NULL));

    // reshape: contiguous only, shares data, -1 is inferred
    int s3[3] = {3, 1, 2};
    cvec_NDArray *r = cvec_ndarray_reshape(a, 3, s3);
    CHECK_NOT_NULL(r);
    CHECK_EQ((int)r->ndim, 3);
    CHECK_VALUES(r, 1, 2, 3, 4, 5, 6);
    int infer[2] = {-1, 2};
    cvec_NDArray *r2 = cvec_ndarray_reshape(a, 2, infer);
    CHECK_NOT_NULL(r2);
    CHECK_EQ(r2->shape[0], 3);
    int flat[1] = {-1};
    cvec_NDArray *r3 = cvec_ndarray_reshape(a, 1, flat);
    CHECK_EQ(r3->shape[0], 6);
    int wrong[2] = {4, 2};
    int two_inferred[2] = {-1, -1};
    int not_divisible[2] = {-1, 4};
    CHECK_NULL(cvec_ndarray_reshape(a, 2, wrong));
    CHECK_NULL(cvec_ndarray_reshape(a, 2, two_inferred));
    CHECK_NULL(cvec_ndarray_reshape(a, 2, not_divisible));
    CHECK_NULL(cvec_ndarray_reshape(t, 2, wrong)); // not contiguous
    cvec_NDArray *t_copy = cvec_ndarray_copy(t);
    int s_back[2] = {2, 3};
    cvec_NDArray *r4 = cvec_ndarray_reshape(t_copy, 2, s_back);
    CHECK_VALUES(r4, 1, 4, 2, 5, 3, 6);

    // slice
    cvec_NDArray *rows = cvec_ndarray_slice(a, 0, 1, 2, 1); // second row
    CHECK_VALUES(rows, 4, 5, 6);
    cvec_NDArray *cols = cvec_ndarray_slice(a, 1, 1, 3, 1); // columns 1, 2
    CHECK_VALUES(cols, 2, 3, 5, 6);
    cvec_NDArray *stepped = cvec_ndarray_slice(a, 1, 0, 3, 2); // columns 0, 2
    CHECK_VALUES(stepped, 1, 3, 4, 6);
    cvec_NDArray *empty = cvec_ndarray_slice(a, 0, 2, 2, 1);
    CHECK_NOT_NULL(empty);
    CHECK_EQ((int)cvec_ndarray_size(empty), 0);
    // slice of a slice keeps the right offset
    cvec_NDArray *nested = cvec_ndarray_slice(cols, 0, 1, 2, 1); // [5, 6]
    CHECK_VALUES(nested, 5, 6);
    CHECK_NULL(cvec_ndarray_slice(a, 2, 0, 1, 1));  // axis out of range
    CHECK_NULL(cvec_ndarray_slice(a, 0, 0, 3, 1));  // stop out of range
    CHECK_NULL(cvec_ndarray_slice(a, 0, 2, 1, 1));  // start > stop
    CHECK_NULL(cvec_ndarray_slice(a, 0, 0, 2, 0));  // step < 1
    CHECK_NULL(cvec_ndarray_slice(NULL, 0, 0, 1, 1));

    PASS("test_views");

    // views are freed without touching the data of the original
    cvec_ndarray_free(t);
    cvec_ndarray_free(p);
    cvec_ndarray_free(r);
    cvec_ndarray_free(r2);
    cvec_ndarray_free(r3);
    cvec_ndarray_free(r4);
    cvec_ndarray_free(t_copy);
    cvec_ndarray_free(rows);
    cvec_ndarray_free(cols);
    cvec_ndarray_free(stepped);
    cvec_ndarray_free(empty);
    cvec_ndarray_free(nested);
    CHECK_FLOAT_NEAR(a->data[5], 6.0f, EPS); // still alive
    cvec_ndarray_free(a);
}

void test_elementwise()
{
    int shape[2] = {2, 3};
    cvec_scalar fa[6] = {1, 2, 3, 4, 5, 6};
    cvec_scalar fb[6] = {6, 5, 4, 3, 2, 1};
    cvec_NDArray *a = cvec_ndarray_from_buffer(2, shape, fa);
    cvec_NDArray *b = cvec_ndarray_from_buffer(2, shape, fb);

    cvec_NDArray *r = cvec_ndarray_add(a, b);
    CHECK_VALUES(r, 7, 7, 7, 7, 7, 7);
    cvec_ndarray_free(r);
    r = cvec_ndarray_sub(a, b);
    CHECK_VALUES(r, -5, -3, -1, 1, 3, 5);
    cvec_ndarray_free(r);
    r = cvec_ndarray_mul(a, b);
    CHECK_VALUES(r, 6, 10, 12, 12, 10, 6);
    cvec_ndarray_free(r);
    r = cvec_ndarray_div(a, a);
    CHECK_VALUES(r, 1, 1, 1, 1, 1, 1);
    cvec_ndarray_free(r);

    // scalars
    r = cvec_ndarray_add_scalar(a, 1.0f);
    CHECK_VALUES(r, 2, 3, 4, 5, 6, 7);
    cvec_ndarray_free(r);
    r = cvec_ndarray_sub_scalar(a, 1.0f);
    CHECK_VALUES(r, 0, 1, 2, 3, 4, 5);
    cvec_ndarray_free(r);
    r = cvec_ndarray_mul_scalar(a, 2.0f);
    CHECK_VALUES(r, 2, 4, 6, 8, 10, 12);
    cvec_ndarray_free(r);
    r = cvec_ndarray_div_scalar(a, 2.0f);
    CHECK_VALUES(r, 0.5f, 1, 1.5f, 2, 2.5f, 3);
    cvec_ndarray_free(r);

    // broadcasting: (2, 3) + (3,)
    int s1[1] = {3};
    cvec_scalar row[3] = {10, 20, 30};
    cvec_NDArray *v = cvec_ndarray_from_buffer(1, s1, row);
    r = cvec_ndarray_add(a, v);
    CHECK_VALUES(r, 11, 22, 33, 14, 25, 36);
    CHECK_EQ((int)r->ndim, 2);
    cvec_ndarray_free(r);
    r = cvec_ndarray_add(v, a); // order doesn't matter
    CHECK_VALUES(r, 11, 22, 33, 14, 25, 36);
    cvec_ndarray_free(r);

    // broadcasting: (2, 1) * (1, 3) -> (2, 3)
    int s_col[2] = {2, 1};
    int s_row[2] = {1, 3};
    cvec_scalar col[2] = {1, 2};
    cvec_scalar row2[3] = {1, 10, 100};
    cvec_NDArray *c = cvec_ndarray_from_buffer(2, s_col, col);
    cvec_NDArray *w = cvec_ndarray_from_buffer(2, s_row, row2);
    r = cvec_ndarray_mul(c, w);
    CHECK_EQ(r->shape[0], 2);
    CHECK_EQ(r->shape[1], 3);
    CHECK_VALUES(r, 1, 10, 100, 2, 20, 200);
    cvec_ndarray_free(r);

    // strided operands: a + a^T^T
    cvec_NDArray *t = cvec_ndarray_transpose(a);
    cvec_NDArray *tt = cvec_ndarray_transpose(t);
    r = cvec_ndarray_add(a, tt);
    CHECK_VALUES(r, 2, 4, 6, 8, 10, 12);
    cvec_ndarray_free(r);
    // (3, 2) with (2, 3) does not broadcast
    CHECK_NULL(cvec_ndarray_add(a, t));
    r = cvec_ndarray_sub_scalar(t, 1.0f); // scalar op on a strided view
    CHECK_VALUES(r, 0, 3, 1, 4, 2, 5);
    cvec_ndarray_free(r);

    CHECK_NULL(cvec_ndarray_add(NULL, a));
    CHECK_NULL(cvec_ndarray_mul_scalar(NULL, 1.0f));

    PASS("test_elementwise");

    cvec_ndarray_free(a);
    cvec_ndarray_free(b);
    cvec_ndarray_free(v);
    cvec_ndarray_free(c);
    cvec_ndarray_free(w);
    cvec_ndarray_free(t);
    cvec_ndarray_free(tt);
}

void test_matmul_broadcast()
{
    // (2, 2, 3) x (3, 2): b is used for every batch of a
    int sa[3] = {2, 2, 3};
    int sb[2] = {3, 2};
    cvec_scalar fa[12], fb[6];
    for (int i = 0; i < 12; i++)
        fa[i] = (float)(i + 1);
    for (int i = 0; i < 6; i++)
        fb[i] = (float)(i % 3);
    cvec_NDArray *a = cvec_ndarray_from_buffer(3, sa, fa);
    cvec_NDArray *b = cvec_ndarray_from_buffer(2, sb, fb);

    cvec_NDArray *r = cvec_ndarray_matmul(a, b);
    CHECK_NOT_NULL(r);
    CHECK_EQ((int)r->ndim, 3);
    for (int n = 0; n < 2; n++)
    {
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                float expected = 0;
                for (int k = 0; k < 3; k++)
                    expected += fa[n * 6 + i * 3 + k] * fb[k * 2 + j];
                int idx[3] = {n, i, j};
                CHECK_FLOAT_NEAR(cvec_ndarray_get(r, idx), expected, EPS);
            }
        }
    }
    cvec_ndarray_free(r);

    // (2, 1, 2, 2) x (3, 2, 2) -> (2, 3, 2, 2): batch dims broadcast both ways
    int sc[4] = {2, 1, 2, 2};
    int sd[3] = {3, 2, 2};
    cvec_scalar fc[8], fd[12];
    for (int i = 0; i < 8; i++)
        fc[i] = (float)(i + 1);
    for (int i = 0; i < 12; i++)
        fd[i] = (float)(i % 5) - 1.0f;
    cvec_NDArray *c = cvec_ndarray_from_buffer(4, sc, fc);
    cvec_NDArray *d = cvec_ndarray_from_buffer(3, sd, fd);
    r = cvec_ndarray_matmul(c, d);
    CHECK_NOT_NULL(r);
    CHECK_EQ(r->shape[0], 2);
    CHECK_EQ(r->shape[1], 3);
    for (int n = 0; n < 2; n++)
        for (int m = 0; m < 3; m++)
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                {
                    float expected = 0;
                    for (int k = 0; k < 2; k++)
                        expected += fc[n * 4 + i * 2 + k] * fd[m * 4 + k * 2 + j];
                    int idx[4] = {n, m, i, j};
                    CHECK_FLOAT_NEAR(cvec_ndarray_get(r, idx), expected, EPS);
                }
    cvec_ndarray_free(r);

    // batch dims (3) vs (2, 5): aligned on the right, 3 vs 5 can't broadcast
    int se[3] = {3, 2, 2};
    int sf[4] = {2, 5, 2, 2};
    cvec_NDArray *e = cvec_ndarray_create(3, se);
    cvec_NDArray *f = cvec_ndarray_create(4, sf);
    CHECK_NULL(cvec_ndarray_matmul(e, f));
    cvec_ndarray_free(e);
    cvec_ndarray_free(f);

    PASS("test_matmul_broadcast");

    cvec_ndarray_free(a);
    cvec_ndarray_free(b);
    cvec_ndarray_free(c);
    cvec_ndarray_free(d);
}

void test_reductions()
{
    int shape[2] = {2, 3};
    cvec_scalar buf[6] = {1, -2, 3, -4, 5, -6};
    cvec_NDArray *a = cvec_ndarray_from_buffer(2, shape, buf);
    cvec_NDArray *t = cvec_ndarray_transpose(a); // strided

    CHECK_FLOAT_NEAR(cvec_ndarray_sum(a), -3.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_sum(t), -3.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_mean(a), -0.5f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_mean(t), -0.5f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_min(a), -6.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_min(t), -6.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_max(a), 5.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_max(t), 5.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_norm_l1(a), 21.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_norm_l1(t), 21.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_norm_l2(a), sqrtf(91.0f), 1e-5);
    CHECK_FLOAT_NEAR(cvec_ndarray_norm_l2(t), sqrtf(91.0f), 1e-5);

    // empty and NULL
    int empty_shape[2] = {0, 3};
    cvec_NDArray *e = cvec_ndarray_create(2, empty_shape);
    CHECK_FLOAT_NEAR(cvec_ndarray_sum(e), 0.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_norm_l2(e), 0.0f, EPS);
    CHECK_NAN(cvec_ndarray_mean(e));
    CHECK_NAN(cvec_ndarray_min(e));
    CHECK_NAN(cvec_ndarray_max(e));
    CHECK_NAN(cvec_ndarray_sum(NULL));
    CHECK_NAN(cvec_ndarray_mean(NULL));
    CHECK_NAN(cvec_ndarray_norm_l2(NULL));

    PASS("test_reductions");

    cvec_ndarray_free(a);
    cvec_ndarray_free(t);
    cvec_ndarray_free(e);
}

void test_dot_and_metrics()
{
    int shape[1] = {3};
    cvec_scalar fa[3] = {1, 2, 3};
    cvec_scalar fb[3] = {4, 5, 6};
    cvec_NDArray *a = cvec_ndarray_from_buffer(1, shape, fa);
    cvec_NDArray *b = cvec_ndarray_from_buffer(1, shape, fb);

    CHECK_FLOAT_NEAR(cvec_ndarray_dot(a, b), 32.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_manhattan_distance(a, b), 9.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_cosine_similarity(a, b), 32.0f / (sqrtf(14.0f) * sqrtf(77.0f)), 1e-5);
    CHECK_FLOAT_NEAR(cvec_ndarray_cosine_distance(a, a), 0.0f, 1e-5);

    // orthogonal vectors
    cvec_scalar fx[2] = {1, 0};
    cvec_scalar fy[2] = {0, 1};
    int s2[1] = {2};
    cvec_NDArray *x = cvec_ndarray_from_buffer(1, s2, fx);
    cvec_NDArray *y = cvec_ndarray_from_buffer(1, s2, fy);
    CHECK_FLOAT_NEAR(cvec_ndarray_cosine_similarity(x, y), 0.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_cosine_distance(x, y), 1.0f, EPS);

    // zero vector has no direction
    cvec_NDArray *zero = cvec_ndarray_zeros(1, s2);
    CHECK_NAN(cvec_ndarray_cosine_similarity(x, zero));
    CHECK_NAN(cvec_ndarray_cosine_distance(x, zero));

    // n-d arrays on non-contiguous views are treated as flat vectors
    int s22[2] = {2, 3};
    cvec_scalar fm[6] = {1, 2, 3, 4, 5, 6};
    cvec_NDArray *m = cvec_ndarray_from_buffer(2, s22, fm);
    cvec_NDArray *t = cvec_ndarray_transpose(m);
    cvec_NDArray *tt = cvec_ndarray_transpose(t);
    CHECK_FLOAT_NEAR(cvec_ndarray_dot(m, tt), 91.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_manhattan_distance(m, tt), 0.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_manhattan_distance(t, t), 0.0f, EPS);
    cvec_NDArray *m2 = cvec_ndarray_add_scalar(m, 1.0f);
    cvec_NDArray *t2 = cvec_ndarray_transpose(m2);
    CHECK_FLOAT_NEAR(cvec_ndarray_manhattan_distance(t, t2), 6.0f, EPS);
    CHECK_FLOAT_NEAR(cvec_ndarray_dot(t, t2), 112.0f, EPS);

    // invalid arguments
    CHECK_NAN(cvec_ndarray_dot(a, m));
    CHECK_NAN(cvec_ndarray_dot(NULL, a));
    CHECK_NAN(cvec_ndarray_manhattan_distance(a, x));
    CHECK_NAN(cvec_ndarray_cosine_similarity(a, x));

    PASS("test_dot_and_metrics");

    cvec_ndarray_free(a);
    cvec_ndarray_free(b);
    cvec_ndarray_free(x);
    cvec_ndarray_free(y);
    cvec_ndarray_free(zero);
    cvec_ndarray_free(m);
    cvec_ndarray_free(t);
    cvec_ndarray_free(tt);
    cvec_ndarray_free(m2);
    cvec_ndarray_free(t2);
}

int main()
{
    test_create_and_set();
    test_matmul_2d();
    test_matmul_batched();
    test_matmul_invalid();
    test_matmul_2d_invalid();
    test_create_invalid();
    test_get_set_out_of_range();

    test_ndarray_euclidean_distance_1d();
    test_ndarray_euclidean_distance_2d();
    test_ndarray_euclidean_distance_3d();
    test_ndarray_euclidean_distance_non_contiguous();
    test_ndarray_euclidean_distance_invalid();
    test_ndarray_euclidean_distance_empty();

    test_constructors();
    test_views();
    test_elementwise();
    test_matmul_broadcast();
    test_reductions();
    test_dot_and_metrics();

    printf("All tests passed!\n");
    return 0;
}