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
    int shape_d[2] = {3, 2};    // different ndim

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

    float data_a[3] = {1.0f, 2.0f, 3.0f};
    float data_b[3] = {4.0f, 6.0f, 3.0f};

    cvec_NDArray a = {1, shape, strides, data_a};
    cvec_NDArray b = {1, shape, strides, data_b};

    float dist = cvec_ndarray_euclidean_distance(&a, &b);

    CHECK_FLOAT_NEAR(dist, 5.0f, EPS);

    PASS("test_ndarray_euclidean_distance_1d");
}

void test_ndarray_euclidean_distance_2d()
{
    int shape[2] = {2, 2};
    int strides[2] = {2, 1};

    float data_a[4] = {1, 2,
                       3, 4};

    float data_b[4] = {1, 0,
                       3, 0};

    cvec_NDArray a = {2, shape, strides, data_a};
    cvec_NDArray b = {2, shape, strides, data_b};

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

    float data_a[6] = {1, 2, 3, 4, 5, 6};
    float data_b[6] = {1, 4,
                       2, 5,
                       3, 9};

    cvec_NDArray a = {2, shape, strides_a, data_a};
    cvec_NDArray b = {2, shape, strides_b, data_b};

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
    float data[6] = {0};

    cvec_NDArray a = {2, shape_a, strides, data};
    cvec_NDArray b = {2, shape_b, strides, data};
    cvec_NDArray c = {1, shape_c, strides_c, data};

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
    cvec_NDArray a = {2, shape, strides, NULL};
    cvec_NDArray b = {2, shape, strides, NULL};

    CHECK_FLOAT_NEAR(cvec_ndarray_euclidean_distance(&a, &b), 0.0f, EPS);

    PASS("test_ndarray_euclidean_distance_empty");
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

    printf("All tests passed!\n");
    return 0;
}