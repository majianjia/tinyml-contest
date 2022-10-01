#include "nnom.h"

/* Weights, bias and Q format */
#define TENSOR_CONV1D_KERNEL_0 {-16, -9, -4, 18, 28, -18, -10, -23, -1, -11, 11, 25, 8, -5, -12, -28, 55, -6, -47, 46, -12, 2, 9, 24, 26, -31, 39, 19, -68, 43, 14, 21, 20, 14, 4, 13, 14, 15, 20, 20}

#define TENSOR_CONV1D_KERNEL_0_DEC_BITS {3}

#define TENSOR_CONV1D_BIAS_0 {24, -16, -82, -39, -16, -42, -23, -13}

#define TENSOR_CONV1D_BIAS_0_DEC_BITS {8}

#define CONV1D_BIAS_LSHIFT {2}

#define CONV1D_OUTPUT_RSHIFT {6}

#define TENSOR_CONV1D_1_KERNEL_0 {-14, -15, -18, -9, -5, 1, 4, 10, 4, 25, -2, -11, 43, -33, 67, -57, 44, 7, 17, 10, -2, -9, -15, -12, -8, 0, 3, -19, -11, -14, 7, 6, 17, 8, -2, -21}

#define TENSOR_CONV1D_1_KERNEL_0_DEC_BITS {3}

#define TENSOR_CONV1D_1_BIAS_0 {5, -29, -65, -32}

#define TENSOR_CONV1D_1_BIAS_0_DEC_BITS {8}

#define CONV1D_1_BIAS_LSHIFT {2}

#define CONV1D_1_OUTPUT_RSHIFT {6}

#define TENSOR_CONV1D_2_KERNEL_0 {-41, -18, -25, 23, 30, 62, 61, 45, 5, -23, -57, -13, 14, 127, 42, 10, -47, -25, -12, 11, -35, -25, -40, -13, 23, 18, -43, 5, 66, -62, -62, 46, 53, -27, 100, -41, 77, 83, -32, 42, 25, -16, -38, -60, -47, -32, -2, 10, 35, 43, 56, 41}

#define TENSOR_CONV1D_2_KERNEL_0_DEC_BITS {5}

#define TENSOR_CONV1D_2_BIAS_0 {-62, -65, -54, -17}

#define TENSOR_CONV1D_2_BIAS_0_DEC_BITS {9}

#define CONV1D_2_BIAS_LSHIFT {3}

#define CONV1D_2_OUTPUT_RSHIFT {8}

#define TENSOR_CONV1D_3_KERNEL_0 {9, -16, 14, 25, -28, 37, -35, -7, -31, 25, -43, -25, -38, -41, 25, -16, 15, -15, -8, 10, 5, 48, -29, -24, -29, 18, 6, 3, -20, -20, -6, -7, -20, -11, -6, 37, -15, 35, -32, -44, -9, -3, -10, 2, -28, -28, -18, -41, 9, 0, -6, 27, -30, 42, -10, -33, -46, 26, -38, -5, -49, -32, -19, -34, -17, -12, 19, 38, -11, 33, 14, -40, 4, 45, 3, -5, -18, 7, -18, -19, -10, -20, 3, 17, -14, 2, 5, -34, -40, 44, -16, -39, -29, -2, 46, -35, 53, 25, 36, -55, 25, -63, 52, -11, 30, -34, -21, 7, 61, -69, -4, 42, 9, -48, -12, -42, -34, -16, 45, -12, 14, -23, 34, 34, -36, 41, 0, 20, 19, -32, 59, -47, 24, -30, 48, -10, 13, -40, 18, 72, 65, 7, -11, -9, 29, 48, -11, -37, 35, -15, 19, 16, 58, -49, 69, 18, 33, 41, -12, 75, 14, 26, 14, -33, 25, -21, 19, 2, 43, -14, 8, 11, 49, -25, 13, 49, -61, 26, 14, -21, -65, -50, -16, -17, 8, -45, 24, -52, 11, -21, -40, -21}

#define TENSOR_CONV1D_3_KERNEL_0_DEC_BITS {9}

#define TENSOR_CONV1D_3_BIAS_0 {98, 62, -10, -13}

#define TENSOR_CONV1D_3_BIAS_0_DEC_BITS {8}

#define CONV1D_3_BIAS_LSHIFT {5}

#define CONV1D_3_OUTPUT_RSHIFT {9}

#define TENSOR_GRU_GRU_CELL_KERNEL_0 {65, 5, 17, 0, 14, 10, 40, -8, 34, 22, 41, 10, 27, -102, 35, -32, 0, -22, -39, -47, 51, -45, 76, 67, 25, 22, 15, 14, -4, -17, -32, 28, 18, 3, 10, -46, -34, 5, -29, 15, -16, 7, 69, 71, 13, 63, 23, -60, 107, 4, 92, -33, 65, -12, 72, -39, 9, 22, 33, 77, -48, 123, -31, 99, 79, 37, -10, -21, 60, -14, -41, 20, 65, -28, 18, -47, 73, 109, 10, 127, 39, 75, 16, -23, 55, 39, 47, 76, 36, 52, 13, 80, 42, -20, -37, -20, -10, -43, -15, 28, -8, 17, 29, 37, 34, 44, 44, -25, -52, -84, -56, -75, -34, -20, -37, -36, 21, -46, 20, 19, -1, -20, 13, 12, -15, -28, -45, 44, 1, 44, -22, -11, -10, 5, -10, 5, 93, -16, 8, 8, 30, 11, 56, -65}

#define TENSOR_GRU_GRU_CELL_KERNEL_0_DEC_BITS {7}

#define TENSOR_GRU_GRU_CELL_RECURRENT_KERNEL_0 {3, -5, -2, -13, 30, -15, 7, 3, -25, 12, -25, 3, -28, 17, -22, -14, -10, -9, 10, 5, -3, -25, -13, 7, 8, 9, 20, -3, 5, 8, 5, -13, 53, -3, 19, 17, 48, -1, 3, -3, 10, 18, -10, -2, -15, -9, -4, -10, -41, -7, -4, 1, -5, -21, -16, -8, 74, -23, 8, 19, -9, -16, -18, 24, -8, -6, -22, -20, -29, -19, 3, -24, -35, -20, -27, -16, -3, 8, -4, 15, -2, 2, -6, -2, -26, -17, 18, -1, 23, 15, -39, 10, 16, 16, -36, 20, -8, -29, 2, -8, 25, -34, 20, -5, 0, 43, -3, 18, -32, 16, -38, 19, 10, 7, -37, -10, 0, -11, 19, 34, -28, 3, -25, -21, -43, -2, 40, -18, -16, -27, -11, 11, 55, -30, 15, -10, 38, -3, 19, 12, -17, 11, -19, -20, 2, 26, -13, -5, 0, 24, -11, 7, -6, -30, -1, 6, 10, -42, 7, -38, 13, 2, 21, -11, -26, 6, -8, -10, 47, 8, 11, 17, 34, -36, -14, -4, 8, 9, 7, 2, -6, 22, 8, -12, -4, -15, 8, 2, -12, 11, 16, -1, -6, -38, 14, 37, -21, 12, -9, -1, -8, 41, -22, 31, 1, -8, -16, -36, -8, -7, 9, 56, 2, 23, -7, 1, -4, 31, -9, 25, -10, -28, 4, 10, 15, 0, -2, 12, 9, 33, 8, 1, -35, -42, -3, 37, -2, 4, 5, -20, 15, 11, 39, -3, 4, -12, 3, -1, 25, -7, -4, -10, 17, 21, -1, 14, -20, 11, 29, -1, -26, -56, 21, 19, 38, -18, 35, -21, 29, 19, 15, -26, 29, 31, 42, -15, -6, -4, 38, 33, -46, -11, 44, 2, -29, -40, 28, 17, 23, 14, 30, 14, -39, -9, -30, 1, -17, -2, -9, 1, 28, -10, 20, -1, 7, -8, 2, 3, 1, 26, -7, -31, 19, 28, 14, -5, 5, -23, -21, -11, -21, -4, 10, 39, -30, 6, 0, -5, 18, 4, -9, 23, -22, -8, 7, -23, 7, -10, 29, 3, -2, -10, -15, 8, -25, 31, -50, 7, -9, 31, 21, 4, 20, -21, -7, 22, 8, -32, -8, 26, -10, 16, 13, -8, 29, 33, -15, -2, 23, -38, -2, -5, -27, -2, 5, 18, -11, -10, -13, 32, 22, 13, 9, 34, -21, 9, 4, -13, 2, -35, 2, -14, -8, 13, -31, 17, -28, 7, 0, 30, -2, -7, 51, 13, -12, 24, -44, 4, 8, 14, 16, -8, -21, 18, -11, -26, 10, -37, 24, 25, -3, -30, 10, -10, -14, 9, 27, 19, 28, -36, -37, 18}

#define TENSOR_GRU_GRU_CELL_RECURRENT_KERNEL_0_DEC_BITS {6}

#define TENSOR_GRU_GRU_CELL_BIAS_0 {28, -29, 17, -46, -31, -43, 27, -19, -32, -52, 77, -29, 66, 41, 44, 61, 82, 26, 44, 43, 88, 39, 3, 92, 3, 3, 6, 2, 8, 22, 11, 10, 11, 12, 5, 7, 28, -29, 17, -46, -31, -43, 27, -19, -32, -52, 77, -29, 66, 41, 44, 61, 82, 26, 44, 43, 88, 39, 3, 92, 4, 4, 4, -2, 11, 23, 5, 12, 9, 10, 8, 2}

#define TENSOR_GRU_GRU_CELL_BIAS_0_DEC_BITS {8}

#define GRU_BIAS_LSHIFT {3}

#define GRU_OUTPUT_RSHIFT {4}

#define TENSOR_DENSE_KERNEL_0 {3, 10, -15, -5, -21, -20, -29, -10, 12, -21, -6, -7, 13, -2, 4, 5, -16, -24, -10, 6, 7, -4, 11, -8, 18, 3, -19, 3, -5, -13, -6, -7, -6, -6, 5, 7, -9, -1, -5, -4, -1, -9, -8, -14, -1, -11, 2, 0, 7, 5, -2, -13, -2, -9, -23, -11, 5, -15, 8, -4, 7, 1, -21, -5, 0, -2, -16, 4, -1, -14, 7, -9, 10, 11, -28, 3, 4, 5, -14, 0, -5, 0, -7, -3, 15, -3, 6, 1, 0, -12, -2, -7, -1, -7, -13, 1, 9, 1, -10, -14, 4, -11, 2, -2, -3, -14, -6, 3, 19, 3, -10, -11, -5, -4, -14, -4, 5, -13, 13, -9, 18, -3, -9, -9, -9, -9, -6, 7, -1, -16, 2, -8, 20, -13, -2, 0, -1, 0, -5, -1, -14, -7, 13, -2, 12, 25, -2, -9, -16, -2, 1, 15, 13, -11, -4, 3, 16, 13, -10, -4, 5, 5, 5, -3, 0, -1, 16, -17, 9, 6, -11, -11, -4, 5, 0, -5, -2, -19, 10, -1, 3, -4, -23, -1, 2, -3, -6, 10, -1, -1, 3, -17, 28, 15, -13, 8, -10, -15, 3, -2, -1, -10, -4, 2, 7, 6, -15, 3, -11, 4, -3, -4, 3, -18, -1, -1, 15, 13, -17, -24, -1, -11, -3, 0, -3, -11, 5, -15, 11, 9, -19, -4, 1, 4, -13, 12, 1, -13, 17, -8, 22, 7, -5, -1, -11, -3, -1, 7, 6, 0, 13, -9, 24, 16, -24, 0, 7, 0, 8, 9, -5, -8, 11, -8, 16, -2, -21, -7, -9, -4, -2, 6, -9, -9, 14, 6, 31, 9, -13, -18, -4, -6, 0, 11, 6, -18, 26, -13, 23, 3, -18, -3, -8, -17, -11, 13, -8, -9, 18, -6, 15, 12, -14, 4, -1, -8, -8, 1, -1, -14, 12, 0, 13, 13, -9, -6, 5, -10, 12, -2, -9, 0, -2, -7, 16, 15, -12, -15, -1, 2, -4, -1, 8, -10, 17, -8, 17, 12, -9, -16, 6, -9, -6, 9, 13, -10, 23, -14, 2, 13, -15, -12, -9, -8, -13, 0, 3, -4, 15, -4, 22, 8, -15, -1, -5, -4, 12, 4, -11, 6, 10, -3, 5, 4, -18, -9, -4, -22, 0, 6, -7, 3, 10, -6, 29, 2, -18, -15, -5, 0, -6, 0, -3, -16, 15, 2, 0, -8, -9, -10, -10, -15, -4, 1, 3, -14, 5, -13, 20, 17, -23, -1, -3, -2, 3, 0, -11, -4, 15, -15, 31, 5, -2, 6, -5, -5, -4, 0, -2, 4, 15, 5, 24, 13, -17, 0, -12, -5, -6, 2, 8, -12, 8, -9, 19, -2, -10, -23, -8, 2, -10, 2, 3, -13, 4, -11, 26, 14, -13, -11, 0, -6, -3, 0, -16, -7, 8, -10, 22, 7, -14, -9, -4, -10, 6, 12, -7, 1, 23, 2, 19, 1, -6, -6, -4, -13, 1, 3, 7, 5, 12, -1, 25, 7, -15, -5, -10, -5, 3, 10, 0, -12, 9, -11, 16, 11, -15, -9, -7, -2, -7, -6, 3, 0, 12, -9, 10, 15, -12, -5, -4, -9, -9, 20, 2, -7, 11, -10, 25, 6, -9, 2, -7, -9, 8, -3, -8, -9, 4, -1, 19, -4, -4, -6, 2, -18, 0, -1, -5, -16, 9, 3, 26, 8, -2, -17, 3, -7, -3, -3, 9, -16, 7, -11, 25, -1, -6, -12, 2, 1, 1, 7, 1, -5, 1, -12, 20, 7, -16, -1, -8, 1, 2, -3, -3, -9, 21, -14, 27, 15, -10, -4, -3, -15, -1, -2, -3, -3, 8, -6, 34, 4, -15, -8, 1, -14, 1, -1, 6, -5, 4, -1, 8, 8, -9, -17, 5, -11, -8, -4, 2, -7, 11, -10, 20, 8, -28, -1, 4, 0, -20, 0, 2, -15, 2, -22, 12, 21, -7, -9, 4, -6, -3, 2, -8, 4, 18, -11, 26, 12, -7, 3, 2, -11, 8, 4, 0, -1, 10, -8, 18, 8, -8, -10, -12, -10, -8, 1, -3, -9, 10, -12, 25, 6, 0, -14, -10, -2, -2, -8, 4, -16, 18, -7, 24, 9, -16, -10, 1, -9, -11, 4, -5, -4, 2, -18, 26, 14, -12, -8, -9, -14, 14, 1, -6, -6, 14, -14, 25, 8, -24, -12, -8, -9, 9, 6, 5, 1, 13, -15, 30, 9, -23, -13, -4, -6, -8, -2, -15, -7, 8, -5, 19, 14, -13, -13, 9, -1, -5, -6, -1, -8, 3, -16, 18, 14, -9, -4, -6, -3, -4, 3, -1, -5, 15, -13, 25, 9, -5, 4, 2, -11, 2, -3, -2, 1, 21, 0, 34, 17, -6, -18, 2, -26, -5, 6, -10, -2, 24, -7, 32, 3, -9, -22, -8, -9, 4, 6, 6, -13, 14, -8, 31, 12, -15, -9, 2, 8, -11, 0, -10, -8, 8, -16, 28, 8, -9, -4, -6, 8, 13, 0, -8, -19, 15, 3, 17, 28, -13, -4, 2, -18, -1, -6, -16, 11, 12, -14, 38, 6, -2, -12, -2, -6, 2, -1, -7, -13, 21, -17, 40, 5, -13, -14, 4, -7, 2, 0, 1, -12, 25, -5, 32, 5, -3, -11, -11, -6, -9, 3, -18, -15, 20, -18, 26, 21, -20, 6, 3, -5, -2, 1, -17, -8, 32, -1, 42, 7, -17, -5, -4, -20, 9, -3, -17, -12, 32, -9, 28, 7, -21, -17, 2, -19, 8, 13, -9, -6, 19, -19, 49, -2, -35, -16, -9, -14, -9, 15, -14, 4, 54, -23, 48, 15, -46, -4, 3, -21, 15, 4, -38, 20, 46, -19, 76, 32, -47, -9, 2, -30, 6, 14, -13, 10, 57, -23}

#define TENSOR_DENSE_KERNEL_0_DEC_BITS {8}

#define TENSOR_DENSE_BIAS_0 {-65}

#define TENSOR_DENSE_BIAS_0_DEC_BITS {10}

#define DENSE_BIAS_LSHIFT {5}

#define DENSE_OUTPUT_RSHIFT {12}


/* output q format for each layer */
#define INPUT_1_OUTPUT_DEC 7
#define INPUT_1_OUTPUT_OFFSET 0
#define CONV1D_OUTPUT_DEC 4
#define CONV1D_OUTPUT_OFFSET 0
#define CONV1D_1_OUTPUT_DEC 4
#define CONV1D_1_OUTPUT_OFFSET 0
#define CONV1D_2_OUTPUT_DEC 4
#define CONV1D_2_OUTPUT_OFFSET 0
#define BATCH_NORMALIZATION_OUTPUT_DEC 4
#define BATCH_NORMALIZATION_OUTPUT_OFFSET 0
#define BATCH_NORMALIZATION_1_OUTPUT_DEC 4
#define BATCH_NORMALIZATION_1_OUTPUT_OFFSET 0
#define BATCH_NORMALIZATION_2_OUTPUT_DEC 4
#define BATCH_NORMALIZATION_2_OUTPUT_OFFSET 0
#define RE_LU_OUTPUT_DEC 4
#define RE_LU_OUTPUT_OFFSET 0
#define RE_LU_1_OUTPUT_DEC 4
#define RE_LU_1_OUTPUT_OFFSET 0
#define RE_LU_2_OUTPUT_DEC 4
#define RE_LU_2_OUTPUT_OFFSET 0
#define DROPOUT_OUTPUT_DEC 4
#define DROPOUT_OUTPUT_OFFSET 0
#define DROPOUT_1_OUTPUT_DEC 4
#define DROPOUT_1_OUTPUT_OFFSET 0
#define DROPOUT_2_OUTPUT_DEC 4
#define DROPOUT_2_OUTPUT_OFFSET 0
#define MAX_POOLING1D_OUTPUT_DEC 4
#define MAX_POOLING1D_OUTPUT_OFFSET 0
#define MAX_POOLING1D_1_OUTPUT_DEC 4
#define MAX_POOLING1D_1_OUTPUT_OFFSET 0
#define MAX_POOLING1D_2_OUTPUT_DEC 4
#define MAX_POOLING1D_2_OUTPUT_OFFSET 0
#define CONCATENATE_OUTPUT_DEC 4
#define CONCATENATE_OUTPUT_OFFSET 0
#define CONV1D_3_OUTPUT_DEC 4
#define CONV1D_3_OUTPUT_OFFSET 0
#define BATCH_NORMALIZATION_3_OUTPUT_DEC 4
#define BATCH_NORMALIZATION_3_OUTPUT_OFFSET 0
#define RE_LU_3_OUTPUT_DEC 4
#define RE_LU_3_OUTPUT_OFFSET 0
#define DROPOUT_3_OUTPUT_DEC 4
#define DROPOUT_3_OUTPUT_OFFSET 0
#define MAX_POOLING1D_3_OUTPUT_DEC 4
#define MAX_POOLING1D_3_OUTPUT_OFFSET 0
#define GRU_OUTPUT_DEC 7
#define GRU_OUTPUT_OFFSET 0
#define FLATTEN_OUTPUT_DEC 7
#define FLATTEN_OUTPUT_OFFSET 0
#define DROPOUT_4_OUTPUT_DEC 7
#define DROPOUT_4_OUTPUT_OFFSET 0
#define DENSE_OUTPUT_DEC 3
#define DENSE_OUTPUT_OFFSET 0
#define ACTIVATION_OUTPUT_DEC 3
#define ACTIVATION_OUTPUT_OFFSET 0

/* bias shift and output shift for none-weighted layer */

/* tensors and configurations for each layer */
static int8_t nnom_input_data[1250] = {0};

const nnom_shape_data_t tensor_input_1_0_dim[] = {1250, 1};
const nnom_qformat_param_t tensor_input_1_0_dec[] = {7};
const nnom_qformat_param_t tensor_input_1_0_offset[] = {0};
const nnom_tensor_t tensor_input_1_0 = {
    .p_data = (void*)nnom_input_data,
    .dim = (nnom_shape_data_t*)tensor_input_1_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_input_1_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_input_1_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};

const nnom_io_config_t input_1_config = {
    .super = {.name = "input_1"},
    .tensor = (nnom_tensor_t*)&tensor_input_1_0
};
const int8_t tensor_conv1d_kernel_0_data[] = TENSOR_CONV1D_KERNEL_0;

const nnom_shape_data_t tensor_conv1d_kernel_0_dim[] = {5, 1, 8};
const nnom_qformat_param_t tensor_conv1d_kernel_0_dec[] = TENSOR_CONV1D_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_kernel_0 = {
    .p_data = (void*)tensor_conv1d_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_bias_0_data[] = TENSOR_CONV1D_BIAS_0;

const nnom_shape_data_t tensor_conv1d_bias_0_dim[] = {8};
const nnom_qformat_param_t tensor_conv1d_bias_0_dec[] = TENSOR_CONV1D_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_bias_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_bias_0 = {
    .p_data = (void*)tensor_conv1d_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_output_shift[] = CONV1D_OUTPUT_RSHIFT;
const nnom_qformat_param_t conv1d_bias_shift[] = CONV1D_BIAS_LSHIFT;
const nnom_conv2d_config_t conv1d_config = {
    .super = {.name = "conv1d"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_conv1d_bias_0,
    .output_shift = (nnom_qformat_param_t *)&conv1d_output_shift, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_bias_shift, 
    .filter_size = 8,
    .kernel_size = {5},
    .stride_size = {2},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};
const int8_t tensor_conv1d_1_kernel_0_data[] = TENSOR_CONV1D_1_KERNEL_0;

const nnom_shape_data_t tensor_conv1d_1_kernel_0_dim[] = {9, 1, 4};
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_dec[] = TENSOR_CONV1D_1_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_1_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_1_kernel_0 = {
    .p_data = (void*)tensor_conv1d_1_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_1_bias_0_data[] = TENSOR_CONV1D_1_BIAS_0;

const nnom_shape_data_t tensor_conv1d_1_bias_0_dim[] = {4};
const nnom_qformat_param_t tensor_conv1d_1_bias_0_dec[] = TENSOR_CONV1D_1_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_1_bias_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_1_bias_0 = {
    .p_data = (void*)tensor_conv1d_1_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_1_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_1_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_1_output_shift[] = CONV1D_1_OUTPUT_RSHIFT;
const nnom_qformat_param_t conv1d_1_bias_shift[] = CONV1D_1_BIAS_LSHIFT;
const nnom_conv2d_config_t conv1d_1_config = {
    .super = {.name = "conv1d_1"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_1_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_conv1d_1_bias_0,
    .output_shift = (nnom_qformat_param_t *)&conv1d_1_output_shift, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_1_bias_shift, 
    .filter_size = 4,
    .kernel_size = {9},
    .stride_size = {2},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};
const int8_t tensor_conv1d_2_kernel_0_data[] = TENSOR_CONV1D_2_KERNEL_0;

const nnom_shape_data_t tensor_conv1d_2_kernel_0_dim[] = {13, 1, 4};
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_dec[] = TENSOR_CONV1D_2_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_2_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_2_kernel_0 = {
    .p_data = (void*)tensor_conv1d_2_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_2_bias_0_data[] = TENSOR_CONV1D_2_BIAS_0;

const nnom_shape_data_t tensor_conv1d_2_bias_0_dim[] = {4};
const nnom_qformat_param_t tensor_conv1d_2_bias_0_dec[] = TENSOR_CONV1D_2_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_2_bias_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_2_bias_0 = {
    .p_data = (void*)tensor_conv1d_2_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_2_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_2_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_2_output_shift[] = CONV1D_2_OUTPUT_RSHIFT;
const nnom_qformat_param_t conv1d_2_bias_shift[] = CONV1D_2_BIAS_LSHIFT;
const nnom_conv2d_config_t conv1d_2_config = {
    .super = {.name = "conv1d_2"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_2_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_conv1d_2_bias_0,
    .output_shift = (nnom_qformat_param_t *)&conv1d_2_output_shift, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_2_bias_shift, 
    .filter_size = 4,
    .kernel_size = {13},
    .stride_size = {2},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_config = {
    .super = {.name = "max_pooling1d"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {4},
    .stride_size = {4},
    .num_dim = 1
};

const nnom_pool_config_t max_pooling1d_1_config = {
    .super = {.name = "max_pooling1d_1"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {4},
    .stride_size = {4},
    .num_dim = 1
};

const nnom_pool_config_t max_pooling1d_2_config = {
    .super = {.name = "max_pooling1d_2"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {4},
    .stride_size = {4},
    .num_dim = 1
};

const nnom_concat_config_t concatenate_config = {
    .super = {.name = "concatenate"},
    .axis = -1
};
const int8_t tensor_conv1d_3_kernel_0_data[] = TENSOR_CONV1D_3_KERNEL_0;

const nnom_shape_data_t tensor_conv1d_3_kernel_0_dim[] = {3, 16, 4};
const nnom_qformat_param_t tensor_conv1d_3_kernel_0_dec[] = TENSOR_CONV1D_3_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_3_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_3_kernel_0 = {
    .p_data = (void*)tensor_conv1d_3_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_3_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_3_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_3_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 3,
    .bitwidth = 8
};
const int8_t tensor_conv1d_3_bias_0_data[] = TENSOR_CONV1D_3_BIAS_0;

const nnom_shape_data_t tensor_conv1d_3_bias_0_dim[] = {4};
const nnom_qformat_param_t tensor_conv1d_3_bias_0_dec[] = TENSOR_CONV1D_3_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_conv1d_3_bias_0_offset[] = {0};
const nnom_tensor_t tensor_conv1d_3_bias_0 = {
    .p_data = (void*)tensor_conv1d_3_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_conv1d_3_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_conv1d_3_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_conv1d_3_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t conv1d_3_output_shift[] = CONV1D_3_OUTPUT_RSHIFT;
const nnom_qformat_param_t conv1d_3_bias_shift[] = CONV1D_3_BIAS_LSHIFT;
const nnom_conv2d_config_t conv1d_3_config = {
    .super = {.name = "conv1d_3"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_conv1d_3_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_conv1d_3_bias_0,
    .output_shift = (nnom_qformat_param_t *)&conv1d_3_output_shift, 
    .bias_shift = (nnom_qformat_param_t *)&conv1d_3_bias_shift, 
    .filter_size = 4,
    .kernel_size = {3},
    .stride_size = {1},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};

const nnom_pool_config_t max_pooling1d_3_config = {
    .super = {.name = "max_pooling1d_3"},
    .padding_type = PADDING_VALID,
    .output_shift = 0,
    .kernel_size = {2},
    .stride_size = {2},
    .num_dim = 1
};
const int8_t tensor_gru_gru_cell_kernel_0_data[] = TENSOR_GRU_GRU_CELL_KERNEL_0;

const nnom_shape_data_t tensor_gru_gru_cell_kernel_0_dim[] = {4, 36};
const nnom_qformat_param_t tensor_gru_gru_cell_kernel_0_dec[] = TENSOR_GRU_GRU_CELL_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_gru_gru_cell_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_gru_gru_cell_kernel_0 = {
    .p_data = (void*)tensor_gru_gru_cell_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_gru_gru_cell_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_gru_gru_cell_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_gru_gru_cell_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};
const int8_t tensor_gru_gru_cell_recurrent_kernel_0_data[] = TENSOR_GRU_GRU_CELL_RECURRENT_KERNEL_0;

const nnom_shape_data_t tensor_gru_gru_cell_recurrent_kernel_0_dim[] = {12, 36};
const nnom_qformat_param_t tensor_gru_gru_cell_recurrent_kernel_0_dec[] = TENSOR_GRU_GRU_CELL_RECURRENT_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_gru_gru_cell_recurrent_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_gru_gru_cell_recurrent_kernel_0 = {
    .p_data = (void*)tensor_gru_gru_cell_recurrent_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_gru_gru_cell_recurrent_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_gru_gru_cell_recurrent_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_gru_gru_cell_recurrent_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};
const int8_t tensor_gru_gru_cell_bias_0_data[] = TENSOR_GRU_GRU_CELL_BIAS_0;

const nnom_shape_data_t tensor_gru_gru_cell_bias_0_dim[] = {2, 36};
const nnom_qformat_param_t tensor_gru_gru_cell_bias_0_dec[] = TENSOR_GRU_GRU_CELL_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_gru_gru_cell_bias_0_offset[] = {0};
const nnom_tensor_t tensor_gru_gru_cell_bias_0 = {
    .p_data = (void*)tensor_gru_gru_cell_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_gru_gru_cell_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_gru_gru_cell_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_gru_gru_cell_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};

const nnom_gru_cell_config_t gru_gru_cell_config = {
    .super = {.name = "gru"},
    .weights = (nnom_tensor_t*)&tensor_gru_gru_cell_kernel_0,
    .recurrent_weights = (nnom_tensor_t*)&tensor_gru_gru_cell_recurrent_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_gru_gru_cell_bias_0,
    .q_dec_z = 3,
    .q_dec_h = 7,
    .units = 12
};

const nnom_rnn_config_t gru_config = {
    .super = {.name = "gru"},
    .return_sequence = true,
    .stateful = false,
    .go_backwards = false
};

const nnom_flatten_config_t flatten_config = {
    .super = {.name = "flatten"}
};
const int8_t tensor_dense_kernel_0_data[] = TENSOR_DENSE_KERNEL_0;

const nnom_shape_data_t tensor_dense_kernel_0_dim[] = {936, 1};
const nnom_qformat_param_t tensor_dense_kernel_0_dec[] = TENSOR_DENSE_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_dense_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_dense_kernel_0 = {
    .p_data = (void*)tensor_dense_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_dense_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};
const int8_t tensor_dense_bias_0_data[] = TENSOR_DENSE_BIAS_0;

const nnom_shape_data_t tensor_dense_bias_0_dim[] = {1};
const nnom_qformat_param_t tensor_dense_bias_0_dec[] = TENSOR_DENSE_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_dense_bias_0_offset[] = {0};
const nnom_tensor_t tensor_dense_bias_0 = {
    .p_data = (void*)tensor_dense_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_dense_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_dense_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_dense_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_qformat_param_t dense_output_shift[] = DENSE_OUTPUT_RSHIFT;
const nnom_qformat_param_t dense_bias_shift[] = DENSE_BIAS_LSHIFT;
const nnom_dense_config_t dense_config = {
    .super = {.name = "dense"},
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .weight = (nnom_tensor_t*)&tensor_dense_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_dense_bias_0,
    .output_shift = (nnom_qformat_param_t *)&dense_output_shift,
    .bias_shift = (nnom_qformat_param_t *)&dense_bias_shift
};
static int8_t nnom_output_data[1] = {0};

const nnom_shape_data_t tensor_output0_dim[] = {1};
const nnom_qformat_param_t tensor_output0_dec[] = {ACTIVATION_OUTPUT_DEC};
const nnom_qformat_param_t tensor_output0_offset[] = {0};
const nnom_tensor_t tensor_output0 = {
    .p_data = (void*)nnom_output_data,
    .dim = (nnom_shape_data_t*)tensor_output0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_output0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_output0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 1,
    .bitwidth = 8
};

const nnom_io_config_t output0_config = {
    .super = {.name = "output0"},
    .tensor = (nnom_tensor_t*)&tensor_output0
};
/* model version */
#define NNOM_MODEL_VERSION (10000*0 + 100*4 + 3)

/* nnom model */
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[19];

	check_model_version(NNOM_MODEL_VERSION);
	new_model(&model);

	layer[0] = input_s(&input_1_config);
	layer[1] = model.hook(conv2d_s(&conv1d_config), layer[0]);
	layer[2] = model.hook(conv2d_s(&conv1d_1_config), layer[0]);
	layer[3] = model.hook(conv2d_s(&conv1d_2_config), layer[0]);
	layer[4] = model.active(act_relu(), layer[1]);
	layer[5] = model.active(act_relu(), layer[2]);
	layer[6] = model.active(act_relu(), layer[3]);
	layer[7] = model.hook(maxpool_s(&max_pooling1d_config), layer[4]);
	layer[8] = model.hook(maxpool_s(&max_pooling1d_1_config), layer[5]);
	layer[9] = model.hook(maxpool_s(&max_pooling1d_2_config), layer[6]);
	layer[10] = model.mergex(concat_s(&concatenate_config), 3 ,layer[7] ,layer[8] ,layer[9]);
	layer[11] = model.hook(conv2d_s(&conv1d_3_config), layer[10]);
	layer[12] = model.active(act_relu(), layer[11]);
	layer[13] = model.hook(maxpool_s(&max_pooling1d_3_config), layer[12]);
	layer[14] = model.hook(rnn_s(gru_cell_s(&gru_gru_cell_config), &gru_config), layer[13]);
	layer[15] = model.hook(flatten_s(&flatten_config), layer[14]);
	layer[16] = model.hook(dense_s(&dense_config), layer[15]);
	layer[17] = model.active(act_sigmoid(DENSE_OUTPUT_DEC), layer[16]);
	layer[18] = model.hook(output_s(&output0_config), layer[17]);
	model_compile(&model, layer[0], layer[18]);
	return &model;
}
