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
    NDArray *a = ndarray_create(2, shape);

    int idx1[2] = {0, 0};
    int idx2[2] = {1, 1};

    ndarray_set(a, idx1, 42.0f);
    ndarray_set(a, idx2, 3.14f);

    CHECK_FLOAT_NEAR(ndarray_get(a, idx1), 42.0f, EPS);
    CHECK_FLOAT_NEAR(ndarray_get(a, idx2), 3.14f, EPS);

    ndarray_free(a);
    PASS("test_create_and_set");
}

void test_matmul_2d()
{
    int shapeA[2] = {2, 3};
    int shapeB[2] = {3, 2};
    NDArray *A = ndarray_create(2, shapeA);
    NDArray *B = ndarray_create(2, shapeB);

    int idx[2];
    float valsA[6] = {1, 2, 3, 4, 5, 6};
    float valsB[6] = {7, 8, 9, 10, 11, 12};

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            idx[0] = i;
            idx[1] = j;
            ndarray_set(A, idx, valsA[i * 3 + j]);
        }
    }
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            idx[0] = i;
            idx[1] = j;
            ndarray_set(B, idx, valsB[i * 2 + j]);
        }
    }

    NDArray *C = ndarray_matmul_2d(A, B);

    int idxC[2] = {0, 0};
    CHECK_FLOAT_NEAR(ndarray_get(C, idxC), 58.0f, EPS);

    idxC[0] = 0;
    idxC[1] = 1;
    CHECK_FLOAT_NEAR(ndarray_get(C, idxC), 64.0f, EPS);

    idxC[0] = 1;
    idxC[1] = 0;
    CHECK_FLOAT_NEAR(ndarray_get(C, idxC), 139.0f, EPS);

    idxC[0] = 1;
    idxC[1] = 1;
    CHECK_FLOAT_NEAR(ndarray_get(C, idxC), 154.0f, EPS);

    PASS("test_matmul_2d");

    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_matmul_batched()
{
    // Many batches so several threads run at once: batch `n` of A is filled
    // with (n + 1) and B with ones, so every entry of C[n] is 3 * (n + 1).
    // With a shared index buffer between threads the batches get mixed up.
    const int batches = 256;
    int shape_a[3] = {batches, 2, 3};
    int shape_b[3] = {batches, 3, 2};

    NDArray *A = ndarray_create(3, shape_a);
    NDArray *B = ndarray_create(3, shape_b);

    for (int n = 0; n < batches; n++)
    {
        for (int i = 0; i < 6; i++)
        {
            A->data[n * 6 + i] = (float)(n + 1);
            B->data[n * 6 + i] = 1.0f;
        }
    }

    NDArray *C = ndarray_matmul(A, B);

    for (int n = 0; n < batches; n++)
    {
        for (int i = 0; i < 4; i++)
        {
            CHECK_FLOAT_NEAR(C->data[n * 4 + i], 3.0f * (n + 1), EPS);
        }
    }

    PASS("test_matmul_batched");

    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_matmul_invalid()
{
    int shape_a[3] = {4, 2, 3};
    int shape_b[3] = {5, 3, 2}; // different batch dim
    int shape_c[3] = {4, 2, 2}; // inner dims 3 vs 2
    int shape_d[2] = {3, 2};    // different ndim

    NDArray *A = ndarray_create(3, shape_a);
    NDArray *B = ndarray_create(3, shape_b);
    NDArray *C = ndarray_create(3, shape_c);
    NDArray *D = ndarray_create(2, shape_d);

    if (ndarray_matmul(A, B) || ndarray_matmul(A, C) || ndarray_matmul(A, D) || ndarray_matmul(NULL, A))
    {
        fprintf(stderr, CROSS " invalid matmul should return NULL\n");
        exit(1);
    }

    PASS("test_matmul_invalid");

    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
    ndarray_free(D);
}

void test_ndarray_euclidean_distance_1d()
{
    int shape[1] = {3};
    int strides[1] = {1};

    float data_a[3] = {1.0f, 2.0f, 3.0f};
    float data_b[3] = {4.0f, 6.0f, 3.0f};

    NDArray a = {1, shape, strides, data_a};
    NDArray b = {1, shape, strides, data_b};

    float dist = ndarray_euclidean_distance(&a, &b);

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

    NDArray a = {2, shape, strides, data_a};
    NDArray b = {2, shape, strides, data_b};

    float dist = ndarray_euclidean_distance(&a, &b);

    CHECK_FLOAT_NEAR(dist, sqrtf(20.0f), EPS);

    PASS("test_ndarray_euclidean_distance_2d");
}

void test_ndarray_euclidean_distance_3d()
{
    int shape[3] = {2, 2, 2}; // 3D array: 2x2x2 (8)

    NDArray *a = ndarray_create(3, shape);
    NDArray *b = ndarray_create(3, shape);

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

    float dist = ndarray_euclidean_distance(a, b);

    // sqrt( sum( (1)^2 * 8 ) ) = sqrt(8)
    float expected = sqrtf(8.0f);

    CHECK_FLOAT_NEAR(dist, expected, EPS);

    PASS("test_ndarray_euclidean_distance_3d");

    ndarray_free(a);
    ndarray_free(b);
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

    NDArray a = {2, shape, strides_a, data_a};
    NDArray b = {2, shape, strides_b, data_b};

    // only the last element differs: 6 vs 9
    float dist = ndarray_euclidean_distance(&a, &b);
    CHECK_FLOAT_NEAR(dist, 3.0f, EPS);

    // both non-contiguous
    dist = ndarray_euclidean_distance(&a, &a);
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

    NDArray a = {2, shape_a, strides, data};
    NDArray b = {2, shape_b, strides, data};
    NDArray c = {1, shape_c, strides_c, data};

    if (!isnan(ndarray_euclidean_distance(&a, &b)))
    {
        fprintf(stderr, CROSS " different shapes should return NAN\n");
        exit(1);
    }
    if (!isnan(ndarray_euclidean_distance(&a, &c)))
    {
        fprintf(stderr, CROSS " different ndim should return NAN\n");
        exit(1);
    }
    if (!isnan(ndarray_euclidean_distance(NULL, &a)))
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
    NDArray a = {2, shape, strides, NULL};
    NDArray b = {2, shape, strides, NULL};

    CHECK_FLOAT_NEAR(ndarray_euclidean_distance(&a, &b), 0.0f, EPS);

    PASS("test_ndarray_euclidean_distance_empty");
}

int main()
{
    test_create_and_set();
    test_matmul_2d();
    test_matmul_batched();
    test_matmul_invalid();

    test_ndarray_euclidean_distance_1d();
    test_ndarray_euclidean_distance_2d();
    test_ndarray_euclidean_distance_3d();
    test_ndarray_euclidean_distance_non_contiguous();
    test_ndarray_euclidean_distance_invalid();
    test_ndarray_euclidean_distance_empty();

    printf("All tests passed!\n");
    return 0;
}