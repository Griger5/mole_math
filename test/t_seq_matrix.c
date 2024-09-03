#include <CUnit/Basic.h>

#include <../include/mole_math/matrix_define.h>
#include "../include/mole_math/seq_matrix_operations.h"
#include "../include/mole_math/seq_matrix_properties.h"
#include "../include/mole_math/seq_matrix_scalars.h"
#include "../include/mole_math/seq_matrix_transform.h"
#include "../include/mole_math/seq_matrix_utils.h"

static Matrix matrix_a;
static Matrix matrix_b;
static Matrix matrix_c;
static Matrix matrix_d;
static Matrix result;

int init_suite(void) {
    matrix_a = matrix_init(2,2);
    matrix_b = matrix_init(2,2);

    matrix_c = matrix_init(1,2);
    matrix_d = matrix_init(2,1);

    result = seq_matrix_nulled(1,1);

    if (matrix_a.values == NULL || matrix_b.values == NULL || matrix_c.values == NULL || matrix_d.values == NULL) return -1;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            matrix_a.values[i][j] = i + j/2 + 1;
            matrix_b.values[i][j] = (i+2) * (j+1) - 1;
        }
    }

    matrix_c.values[0][0] = 2.0;
    matrix_c.values[0][1] = -1.0;
    matrix_d.values[0][0] = 3.0;
    matrix_d.values[1][0] = 5.0;

    return 0;
}

int clean_suite(void) {
    MFREE(matrix_a);
    MFREE(matrix_b);
    MFREE(matrix_c);
    MFREE(matrix_d);
    MFREE(result);

    return 0;
}

void test_matrix_multiply1(void) {
    Matrix temp = seq_matrix_multiply(matrix_a, matrix_b);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(result.values[0][0], 3.0);
    CU_ASSERT_EQUAL(result.values[0][1], 8.0);
    CU_ASSERT_EQUAL(result.values[1][0], 6.0);
    CU_ASSERT_EQUAL(result.values[1][1], 16.0);
}

void test_matrix_multiply2(void) {
    Matrix temp = seq_matrix_multiply(matrix_c, matrix_d);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(result.values[0][0], 1.0);
    CU_ASSERT_EQUAL(result.rows, 1);
    CU_ASSERT_EQUAL(result.cols, 1);
}

void test_matrix_subtract_elements1(void) {
    Matrix temp = seq_matrix_subtract_elements(matrix_a, matrix_b);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(result.values[0][0], 0.0);
    CU_ASSERT_EQUAL(result.values[0][1], -2.0);
    CU_ASSERT_EQUAL(result.values[1][0], 0.0);
    CU_ASSERT_EQUAL(result.values[1][1], -3.0);
}

void test_matrix_subtract_elements2(void) {
    Matrix temp = seq_matrix_subtract_elements(matrix_c, matrix_d);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_PTR_NULL(result.values);
}

void test_matrix_multiply_elements1(void) {
    Matrix temp = seq_matrix_multiply_elements(matrix_a, matrix_b);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(result.values[0][0], 1.0);
    CU_ASSERT_EQUAL(result.values[0][1], 3.0);
    CU_ASSERT_EQUAL(result.values[1][0], 4.0);
    CU_ASSERT_EQUAL(result.values[1][1], 10.0);
}

void test_matrix_multiply_elements2(void) {
    Matrix temp = seq_matrix_multiply_elements(matrix_c, matrix_d);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_PTR_NULL(result.values);
}

int main() {
    if (CUE_SUCCESS != CU_initialize_registry()) return CU_get_error();

    CU_pSuite mat_oper_Suite = NULL;
    mat_oper_Suite = CU_add_suite("seq_matrix_operations", init_suite, clean_suite);
   
    if (NULL == mat_oper_Suite) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    if (NULL == CU_add_test(mat_oper_Suite, "test 1 of seq_matrix_multiply", test_matrix_multiply1)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_oper_Suite, "test 2 of seq_matrix_multiply", test_matrix_multiply2)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_oper_Suite, "test 1 of seq_matrix_subtract_elements", test_matrix_subtract_elements1)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_oper_Suite, "test 2 of seq_matrix_subtract_elements", test_matrix_subtract_elements2)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_oper_Suite, "test 1 of seq_matrix_multiply_elements", test_matrix_multiply_elements1)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_oper_Suite, "test 2 of seq_matrix_multiply_elements", test_matrix_multiply_elements2)) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();
   
    return CU_get_error();
}