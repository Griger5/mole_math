#include <CUnit/Basic.h>
#include <omp.h>

#include "../include/mole_math/matrix_define.h"
#include "../include/mole_math/omp_matrix_utils.h"

static Matrix matrix_a;
static Matrix matrix_b;

int init_suite_util(void) {
    matrix_a = matrix_init(1,3);

    if (matrix_a.values == NULL) return -1;

    matrix_a.values[0][0] = 1.0;
    matrix_a.values[0][1] = 2.0;
    matrix_a.values[0][2] = 3.0;

    return 0;
}

int clean_suite_util(void) {
    MFREE(matrix_a);
    MFREE(matrix_b);

    return 0;
}

void test_matrix_identity(void) {
    matrix_b = omp_matrix_identity(3);

    for (int i = 0; i < 3; i++) {
        CU_ASSERT_EQUAL(matrix_b.values[i][i], 1.0);
    }
}

void test_matrix_nulled(void) {
    omp_matrix_replace(&matrix_b, omp_matrix_nulled(3,3));

    CU_ASSERT_PTR_NULL(matrix_b.values);
}

void test_matrix_copy(void) {
    Matrix temp = omp_matrix_copy(matrix_a);
    omp_matrix_replace(&matrix_b, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(matrix_b.rows, 1);
    CU_ASSERT_EQUAL(matrix_b.cols, 3)
    CU_ASSERT_EQUAL(matrix_b.values[0][0], 1.0);
    CU_ASSERT_EQUAL(matrix_b.values[0][1], 2.0);
    CU_ASSERT_EQUAL(matrix_b.values[0][2], 3.0);
}

void test_matrix_replace(void) {
    Matrix temp = matrix_init(2,2);
    omp_matrix_replace(&matrix_b, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(matrix_b.rows, 2);
    CU_ASSERT_EQUAL(matrix_b.cols, 2)
    CU_ASSERT_EQUAL(matrix_b.values[0][0], 0.0);
    CU_ASSERT_EQUAL(matrix_b.values[0][1], 0.0);
    CU_ASSERT_EQUAL(matrix_b.values[1][0], 0.0);
    CU_ASSERT_EQUAL(matrix_b.values[1][1], 0.0);
}

void test_matrix_array_to_matrix(void) {
    double array[3];
    array[0] = 1.0;
    array[1] = 2.0;
    array[2] = 3.0;

    Matrix temp = omp_matrix_array_to_matrix(array, 3);
    omp_matrix_replace(&matrix_b, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(matrix_b.rows, 1);
    CU_ASSERT_EQUAL(matrix_b.cols, 3)
    CU_ASSERT_EQUAL(matrix_b.values[0][0], 1.0);
    CU_ASSERT_EQUAL(matrix_b.values[0][1], 2.0);
    CU_ASSERT_EQUAL(matrix_b.values[0][2], 3.0);
}

int main() {
    omp_set_num_threads(16*omp_get_num_procs());

    if (CUE_SUCCESS != CU_initialize_registry()) return CU_get_error();

    CU_pSuite mat_util_Suite = NULL;
    mat_util_Suite = CU_add_suite("omp_matrix_transform", init_suite_util, clean_suite_util);

    if (NULL == mat_util_Suite) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    if (NULL == CU_add_test(mat_util_Suite, "test 1 of omp_matrix_identity", test_matrix_identity)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_util_Suite, "test 1 of omp_matrix_nulled", test_matrix_nulled)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_util_Suite, "test 1 of omp_matrix_copy", test_matrix_copy)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_util_Suite, "test 1 of omp_matrix_replace", test_matrix_replace)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_util_Suite, "test 1 of omp_matrix_array_to_matrix", test_matrix_array_to_matrix)) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();
   
    return CU_get_error();
}