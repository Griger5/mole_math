#include <CUnit/Basic.h>
#include <omp.h>

#define PRIVATE_MAT
#include "../include/mole_math/matrix_define.h"
#include "../include/mole_math/seq_matrix_utils.h"
#include "../include/mole_math/omp_matrix_funcs.h"
#include "../include/mole_math/omp_matrix_operations.h"
#include "../include/mole_math/omp_matrix_properties.h"
#include "../include/mole_math/omp_matrix_scalars.h"
#include "../include/mole_math/omp_matrix_transform.h"
#include "../include/mole_math/omp_matrix_utils.h"

static Matrix matrix_a;
static Matrix matrix_b;
static Matrix matrix_c;
static Matrix matrix_d;
static Matrix result;

int init_suite_func(void) {
    matrix_a = matrix_init(2,2);

    if (matrix_a.values == NULL) return -1;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            matrix_a.values[i][j] = i + j + 1;
        }
    }

    return 0;
}

int clean_suite_func(void) {
    MFREE(matrix_a);

    return 0;
}

void test_matrix_sum_row1(void) {
    CU_ASSERT_EQUAL(omp_matrix_sum_row(matrix_a, 0), 3.0);
}

void test_matrix_sum_row2(void) {
    CU_ASSERT(isnan(omp_matrix_sum_row(matrix_a, 2)));
}

int init_suite_oper(void) {
    matrix_a = matrix_init(2,2);
    matrix_b = matrix_init(2,2);

    matrix_c = matrix_init(1,2);
    matrix_d = matrix_init(2,1);

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

int clean_suite_oper(void) {
    MFREE(matrix_a);
    MFREE(matrix_b);
    MFREE(matrix_c);
    MFREE(matrix_d);
    MFREE(result);

    return 0;
}

void test_matrix_multiply1(void) {
    Matrix temp = omp_matrix_multiply(matrix_a, matrix_b);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(result.values[0][0], 3.0);
    CU_ASSERT_EQUAL(result.values[0][1], 8.0);
    CU_ASSERT_EQUAL(result.values[1][0], 6.0);
    CU_ASSERT_EQUAL(result.values[1][1], 16.0);
}

void test_matrix_multiply2(void) {
    Matrix temp = omp_matrix_multiply(matrix_c, matrix_d);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(result.values[0][0], 1.0);
    CU_ASSERT_EQUAL(result.rows, 1);
    CU_ASSERT_EQUAL(result.cols, 1);
}

void test_matrix_subtract_elements1(void) {
    Matrix temp = omp_matrix_subtract_elements(matrix_a, matrix_b);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(result.values[0][0], 0.0);
    CU_ASSERT_EQUAL(result.values[0][1], -2.0);
    CU_ASSERT_EQUAL(result.values[1][0], 0.0);
    CU_ASSERT_EQUAL(result.values[1][1], -3.0);
}

void test_matrix_subtract_elements2(void) {
    Matrix temp = omp_matrix_subtract_elements(matrix_c, matrix_d);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_PTR_NULL(result.values);
}

void test_matrix_multiply_elements1(void) {
    Matrix temp = omp_matrix_multiply_elements(matrix_a, matrix_b);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(result.values[0][0], 1.0);
    CU_ASSERT_EQUAL(result.values[0][1], 3.0);
    CU_ASSERT_EQUAL(result.values[1][0], 4.0);
    CU_ASSERT_EQUAL(result.values[1][1], 10.0);
}

void test_matrix_multiply_elements2(void) {
    Matrix temp = omp_matrix_multiply_elements(matrix_c, matrix_d);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_PTR_NULL(result.values);
}

int init_suite_prop(void) {
    matrix_a = matrix_init(2,2);
    matrix_b = matrix_init(2,2);
    matrix_c = matrix_init(1,2);

    if (matrix_a.values == NULL || matrix_b.values == NULL || matrix_c.values == NULL) return -1;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            matrix_a.values[i][j] = i + j/2 + 1;
            matrix_b.values[i][j] = (i+2) * (j+1) - 1;
        }
    }

    matrix_c.values[0][0] = 2.0;
    matrix_c.values[0][1] = -1.0;

    return 0;
}

int clean_suite_prop(void) {
    MFREE(matrix_a);
    MFREE(matrix_b);
    MFREE(matrix_c);

    return 0;
}

void test_matrix_determinant1(void) {
    CU_ASSERT_EQUAL(omp_matrix_determinant(matrix_a), 0.0);
}

void test_matrix_determinant2(void) {
    CU_ASSERT_EQUAL(omp_matrix_determinant(matrix_b), -1.0);
}

void test_matrix_determinant3(void) {
    CU_ASSERT(isnan(omp_matrix_determinant(matrix_c)));
}

void test_matrix_ij_minor1(void) {
    CU_ASSERT_EQUAL(omp_matrix_ij_minor(matrix_a, 0, 0), 2.0);
}

void test_matrix_ij_minor2(void) {
    CU_ASSERT(isnan(omp_matrix_ij_minor(matrix_a, 2, 2)));
}

void test_matrix_ij_minor3(void) {
    CU_ASSERT(isnan(omp_matrix_ij_minor(matrix_c, 0, 0)));
}

void test_matrix_cofactor(void) {
    Matrix temp = omp_matrix_cofactor(matrix_a);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(result.values[0][0], 2.0);
    CU_ASSERT_EQUAL(result.values[0][1], -2.0);
    CU_ASSERT_EQUAL(result.values[1][0], -1.0);
    CU_ASSERT_EQUAL(result.values[1][1], 1.0);
}

int init_suite_scal(void) {
    matrix_a = matrix_init(2,2);

    if (matrix_a.values == NULL) return -1;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            matrix_a.values[i][j] = i + j + 1;
        }
    }

    return 0;
}

int clean_suite_scal(void) {
    MFREE(matrix_a);

    return 0;
}

void test_matrix_subtract_scalar(void) {
    omp_matrix_subtract_scalar(&matrix_a, 0.5);

    CU_ASSERT_EQUAL(matrix_a.values[0][0], 0.5);
    CU_ASSERT_EQUAL(matrix_a.values[0][1], 1.5);
    CU_ASSERT_EQUAL(matrix_a.values[1][0], 1.5);
    CU_ASSERT_EQUAL(matrix_a.values[1][1], 2.5);
}

void test_matrix_multiply_row_scalar(void) {
    omp_matrix_multiply_row_scalar(&matrix_a, 1, -4);

    CU_ASSERT_EQUAL(matrix_a.values[1][0], -6.0);
    CU_ASSERT_EQUAL(matrix_a.values[1][1], -10.0);
}

int init_suite_tran(void) {
    matrix_a = matrix_init(2,2);
    matrix_b = matrix_init(2,2);
    matrix_c = matrix_init(1,2);

    if (matrix_a.values == NULL || matrix_b.values == NULL || matrix_c.values == NULL) return -1;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            matrix_a.values[i][j] = i + j/2 + 1;
            matrix_b.values[i][j] = (i+2) * (j+1) - 1;
        }
    }

    matrix_c.values[0][0] = 2.0;
    matrix_c.values[0][1] = -1.0;

    return 0;
}

int clean_suite_tran(void) {
    MFREE(matrix_a);
    MFREE(matrix_b);
    MFREE(matrix_c);
    MFREE(result);

    return 0;
}

void test_matrix_subtract_rows1(void) {
    omp_matrix_subtract_rows(&matrix_b, 2, 1, 2);

    CU_ASSERT_EQUAL(matrix_b.values[0][0], 1.0);
    CU_ASSERT_EQUAL(matrix_b.values[0][1], 3.0);
    CU_ASSERT_EQUAL(matrix_b.values[1][0], 2.0);
    CU_ASSERT_EQUAL(matrix_b.values[1][1], 5.0);
}

void test_matrix_subtract_rows2(void) {
    omp_matrix_subtract_rows(&matrix_b, 0, 1, 2);

    CU_ASSERT_EQUAL(matrix_b.values[0][0], -3.0);
    CU_ASSERT_EQUAL(matrix_b.values[0][1], -7.0);
    CU_ASSERT_EQUAL(matrix_b.values[1][0], 2.0);
    CU_ASSERT_EQUAL(matrix_b.values[1][1], 5.0);
}

void test_matrix_transpose1(void) {
    Matrix temp = omp_matrix_transpose(matrix_a);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(result.values[0][0], 1.0);
    CU_ASSERT_EQUAL(result.values[0][1], 2.0);
    CU_ASSERT_EQUAL(result.values[1][0], 1.0);
    CU_ASSERT_EQUAL(result.values[1][1], 2.0);
}

void test_matrix_transpose2(void) {
    Matrix temp = omp_matrix_transpose(matrix_c);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(result.rows, 2);
    CU_ASSERT_EQUAL(result.cols, 1);
    CU_ASSERT_EQUAL(result.values[0][0], 2.0);
    CU_ASSERT_EQUAL(result.values[1][0], -1.0);
}

void test_matrix_ij_minor_matrix1(void) {
    Matrix temp = omp_matrix_ij_minor_matrix(matrix_a, 0, 0);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(result.values[0][0], 2.0);
}

void test_matrix_ij_minor_matrix2(void) {
    Matrix temp = omp_matrix_ij_minor_matrix(matrix_a, 2, 2);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_PTR_NULL(result.values);
}

void test_matrix_inverse1(void) {
    Matrix temp = omp_matrix_inverse(matrix_a);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_PTR_NULL(result.values);
}

void test_matrix_inverse2(void) {
    Matrix temp = omp_matrix_inverse(matrix_b);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_DOUBLE_EQUAL(result.values[0][0], -5.0, 0.00001);
    CU_ASSERT_DOUBLE_EQUAL(result.values[0][1], -7.0, 0.00001);
    CU_ASSERT_DOUBLE_EQUAL(result.values[1][0], 2.0, 0.00001);
    CU_ASSERT_DOUBLE_EQUAL(result.values[1][1], 3.0, 0.00001);
}

void test_matrix_inverse3(void) {
    Matrix temp = omp_matrix_inverse(matrix_c);
    seq_matrix_replace(&result, temp);
    MFREE(temp);

    CU_ASSERT_PTR_NULL(result.values);
}

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
    matrix_b = seq_matrix_identity(3);

    for (int i = 0; i < 3; i++) {
        CU_ASSERT_EQUAL(matrix_b.values[i][i], 1.0);
    }
}

void test_matrix_init_integers(void) {
    Matrix temp = omp_matrix_init_integers(2,2);
    seq_matrix_replace(&matrix_b, temp);
    MFREE(temp);

    CU_ASSERT_EQUAL(matrix_b.rows, 2);
    CU_ASSERT_EQUAL(matrix_b.cols, 2)
    CU_ASSERT_EQUAL(matrix_b.values[0][0], 1.0);
    CU_ASSERT_EQUAL(matrix_b.values[0][1], 2.0);
    CU_ASSERT_EQUAL(matrix_b.values[1][0], 3.0);
    CU_ASSERT_EQUAL(matrix_b.values[1][1], 4.0);
}

int main() {
    omp_set_num_threads(16*omp_get_num_procs());

    if (CUE_SUCCESS != CU_initialize_registry()) return CU_get_error();

     CU_pSuite mat_func_Suite = NULL;
    mat_func_Suite = CU_add_suite("omp_matrix_funcs", init_suite_func, clean_suite_func);

    if (NULL == mat_func_Suite) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    if (NULL == CU_add_test(mat_func_Suite, "test 1 of omp_matrix_sum_row", test_matrix_sum_row1)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_func_Suite, "test 1 of omp_matrix_sum_row", test_matrix_sum_row2)) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    CU_pSuite mat_oper_Suite = NULL;
    mat_oper_Suite = CU_add_suite("omp_matrix_operations", init_suite_oper, clean_suite_oper);
   
    if (NULL == mat_oper_Suite) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    if (NULL == CU_add_test(mat_oper_Suite, "test 1 of omp_matrix_multiply", test_matrix_multiply1)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_oper_Suite, "test 2 of omp_matrix_multiply", test_matrix_multiply2)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_oper_Suite, "test 1 of omp_matrix_subtract_elements", test_matrix_subtract_elements1)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_oper_Suite, "test 2 of omp_matrix_subtract_elements", test_matrix_subtract_elements2)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_oper_Suite, "test 1 of omp_matrix_multiply_elements", test_matrix_multiply_elements1)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_oper_Suite, "test 2 of omp_matrix_multiply_elements", test_matrix_multiply_elements2)) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    CU_pSuite mat_prop_Suite = NULL;
    mat_prop_Suite = CU_add_suite("omp_matrix_properties", init_suite_prop, clean_suite_prop);

    if (NULL == mat_prop_Suite) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    if (NULL == CU_add_test(mat_prop_Suite, "test 1 of omp_matrix_determinant", test_matrix_determinant1)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_prop_Suite, "test 2 of omp_matrix_determinant", test_matrix_determinant2)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_prop_Suite, "test 3 of omp_matrix_determinant", test_matrix_determinant3)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_prop_Suite, "test 1 of omp_matrix_ij_minor", test_matrix_ij_minor1)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_prop_Suite, "test 2 of omp_matrix_ij_minor", test_matrix_ij_minor2)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_prop_Suite, "test 3 of omp_matrix_ij_minor", test_matrix_ij_minor3)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_prop_Suite, "test 1 of omp_matrix_cofactor", test_matrix_cofactor)) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    CU_pSuite mat_scal_Suite = NULL;
    mat_scal_Suite = CU_add_suite("omp_matrix_scalars", init_suite_scal, clean_suite_scal);

    if (NULL == mat_scal_Suite) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    if (NULL == CU_add_test(mat_scal_Suite, "test 1 of omp_matrix_subtract_scalar", test_matrix_subtract_scalar)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_scal_Suite, "test 1 of omp_matrix_multiply_row_scalar", test_matrix_multiply_row_scalar)) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    CU_pSuite mat_tran_Suite = NULL;
    mat_tran_Suite = CU_add_suite("seq_matrix_transform", init_suite_tran, clean_suite_tran);

    if (NULL == mat_tran_Suite) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    if (NULL == CU_add_test(mat_tran_Suite, "test 1 of omp_matrix_subtract_rows", test_matrix_subtract_rows1)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_tran_Suite, "test 2 of omp_matrix_subtract_rows", test_matrix_subtract_rows2)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_tran_Suite, "test 1 of omp_matrix_inverse", test_matrix_inverse1)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_tran_Suite, "test 2 of omp_matrix_inverse", test_matrix_inverse2)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_tran_Suite, "test 3 of omp_matrix_inverse", test_matrix_inverse3)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_tran_Suite, "test 1 of omp_matrix_transpose", test_matrix_transpose1)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_tran_Suite, "test 2 of omp_matrix_transpose", test_matrix_transpose2)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_tran_Suite, "test 1 of omp_matrix_ij_minor_matrix", test_matrix_ij_minor_matrix1)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_tran_Suite, "test 2 of omp_matrix_ij_minor_matrix", test_matrix_ij_minor_matrix2)) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    CU_pSuite mat_util_Suite = NULL;
    mat_util_Suite = CU_add_suite("omp_matrix_utils", init_suite_util, clean_suite_util);

    if (NULL == mat_util_Suite) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    if (NULL == CU_add_test(mat_util_Suite, "test 1 of omp_matrix_identity", test_matrix_identity)) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    if (NULL == CU_add_test(mat_util_Suite, "test 1 of omp_matrix_init_integers", test_matrix_init_integers)) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();
   
    return CU_get_error();
}