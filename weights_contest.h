#include "nnom.h"

/* Weights, bias and Q format */
#define TENSOR_CONV1D_KERNEL_0 {39, -69, 59, -17, 3, 27, 15, 0, -14, -15, 21, -18, -21, 51, -26, 29, -41, -9, 66, -47, 14, 16, 17, 24, 18, 21, -13, -13, 45, -24, 6, 15, 17, 19, 16, 27, 12, -2, -6, -9}

#define TENSOR_CONV1D_KERNEL_0_DEC_BITS {3}

#define TENSOR_CONV1D_BIAS_0 {-58, -33, -27, -41, 0, -108, 23, -92}

#define TENSOR_CONV1D_BIAS_0_DEC_BITS {8}

#define CONV1D_BIAS_LSHIFT {2}

#define CONV1D_OUTPUT_RSHIFT {6}

#define TENSOR_CONV1D_1_KERNEL_0 {49, 38, 20, -2, -32, -56, -71, -45, -11, 43, 14, -53, -68, -42, -34, 13, 49, 71, 3, 40, 39, -33, -55, -78, -56, 8, -49, -61, -4, 22, 66, 65, 21, 1, -40, -33}

#define TENSOR_CONV1D_1_KERNEL_0_DEC_BITS {5}

#define TENSOR_CONV1D_1_BIAS_0 {-5, 7, -109, -48}

#define TENSOR_CONV1D_1_BIAS_0_DEC_BITS {9}

#define CONV1D_1_BIAS_LSHIFT {3}

#define CONV1D_1_OUTPUT_RSHIFT {8}

#define TENSOR_CONV1D_2_KERNEL_0 {16, 39, -64, 90, -59, 83, 14, -56, 54, 62, -9, -39, 91, -11, -12, -17, -9, -10, 3, 3, 17, 21, 42, 15, 5, -33, 28, 23, 9, 3, -17, -19, -32, -17, -20, -3, 5, 10, 23, 37, 19, -5, -17, -33, -40, -28, -16, -12, -14, -7, 4, 19}

#define TENSOR_CONV1D_2_KERNEL_0_DEC_BITS {4}

#define TENSOR_CONV1D_2_BIAS_0 {-67, -6, 12, -20}

#define TENSOR_CONV1D_2_BIAS_0_DEC_BITS {8}

#define CONV1D_2_BIAS_LSHIFT {3}

#define CONV1D_2_OUTPUT_RSHIFT {7}

#define TENSOR_CONV1D_3_KERNEL_0 {2, 0, 25, 18, -27, 7, -17, -6, -42, -15, 19, 1, 0, -29, -38, -29, 4, 15, -3, 1, -7, 28, -18, 33, -27, -11, 20, 13, -2, 1, -26, -32, 24, -8, 6, 6, -20, 16, -25, 10, -16, -16, -6, -3, 17, -38, -22, -35, 25, -3, 24, 43, -43, 49, -19, 11, -31, -2, 17, -7, 40, -37, -62, -45, 28, 4, 45, 39, -27, 45, -34, 60, -31, 3, 7, 3, 30, -12, -34, -25, 45, -22, 56, 22, -24, 42, -33, 7, -23, -21, 0, -3, 22, -43, -23, -30, -25, -9, -42, -19, 18, 13, 30, 76, 11, 28, 75, 43, -36, 23, -9, 84, -17, 3, -53, -20, 63, -24, 41, -31, 33, 56, 62, -22, -32, 52, 38, -25, -40, 62, -29, -22, -29, -53, -15, 55, -25, -39, -15, 64, -39, 1, -21, -4, -21, 46, -61, -39, -13, 10, 40, 43, 45, 56, 0, 32, -52, 37, 57, 56, -22, 13, -39, -56, 58, 5, -16, -29, 55, 48, -36, 37, -52, 57, 51, -27, -52, 61, -14, -19, -48, -9, -1, 41, -11, -3, -18, 55, -41, -49, -15, -2}

#define TENSOR_CONV1D_3_KERNEL_0_DEC_BITS {9}

#define TENSOR_CONV1D_3_BIAS_0 {77, 55, -20, -24}

#define TENSOR_CONV1D_3_BIAS_0_DEC_BITS {8}

#define CONV1D_3_BIAS_LSHIFT {5}

#define CONV1D_3_OUTPUT_RSHIFT {8}

#define TENSOR_GRU_GRU_CELL_KERNEL_0 {39, -20, 7, 42, 96, -29, -61, 2, 33, -41, 44, -40, -13, 14, -23, -45, 37, -24, 48, -19, 11, -5, 2, 37, -114, 29, -118, 49, 34, 6, 53, 36, 78, 11, 10, 33, 115, 69, 85, 8, 41, 24, -72, -61, -6, 22, 0, 9, -10, 29, 60, 76, 30, -48, 4, -16, 22, -72, 73, -104, 23, 80, 20, 59, -15, -60, 36, 4, 29, -48, -60, -41, 8, 27, 15, 52, 50, -1, 58, -4, 39, 13, 17, -1, -39, 2, -10, -11, -90, 61, -72, 31, 21, -1, 21, 60}

#define TENSOR_GRU_GRU_CELL_KERNEL_0_DEC_BITS {7}

#define TENSOR_GRU_GRU_CELL_RECURRENT_KERNEL_0 {30, 16, 3, 6, 14, 2, 11, 1, -28, 4, 3, -17, -3, -9, -22, -16, -16, -17, -8, -5, -10, -28, 29, -14, 13, 5, -6, -9, 8, -20, -28, -7, 21, 50, -8, 36, -22, -16, 25, 32, 15, 15, 17, -14, 28, 24, -17, 33, 7, -6, -5, 10, -18, -47, -6, -7, -15, -1, -1, 17, -20, 12, 9, -14, 12, 17, -40, -1, -12, 37, -22, -27, -9, 27, 13, -15, 9, -27, -38, -20, 11, 7, 21, 10, 24, 2, -8, 15, -14, 8, 18, -16, 3, 0, -10, 70, 37, 28, -10, -2, 27, -10, -33, 53, -3, -20, 12, 15, 1, 19, -26, -11, 7, 17, 5, -24, 0, -34, -2, 39, -6, -27, 17, 42, -11, 27, 10, 9, 12, -8, -23, 32, -12, -15, 29, -19, 8, 12, -15, 16, 23, -28, 21, -1, -2, -11, 17, -18, -2, 19, -1, -29, -26, -2, 10, -24, 3, -9, -11, -58, 12, -6, 12, 11, 4, 16, -6, -15, -7, 0, -16, 2, 10, -29, -8, 6, 8, -23, -16, 25, -29, -17, 22, 23, -7, 15, -41, 15, -19, -12, 11, 11}

#define TENSOR_GRU_GRU_CELL_RECURRENT_KERNEL_0_DEC_BITS {6}

#define TENSOR_GRU_GRU_CELL_BIAS_0 {16, 9, 36, -36, -7, 8, 48, -61, 26, 67, 58, 82, 63, 23, 16, 16, -6, 0, -1, 38, -3, -2, 6, -7, 16, 9, 36, -36, -7, 8, 48, -61, 26, 67, 58, 82, 63, 23, 16, 16, 2, 0, -9, 27, -6, -3, 7, -1}

#define TENSOR_GRU_GRU_CELL_BIAS_0_DEC_BITS {8}

#define GRU_BIAS_LSHIFT {4}

#define GRU_OUTPUT_RSHIFT {5}

#define TENSOR_GRU_1_GRU_CELL_1_KERNEL_0 {62, 51, -42, 51, -29, 47, 6, 79, -25, 29, -34, 57, 43, -10, -8, 5, 17, -47, 18, 48, -56, -32, -15, -37, -97, 47, -84, 62, -30, -118, 5, -105, 23, 42, 66, -3, -27, 55, 9, 10, -7, 38, 48, 41, -37, -45, -17, -15, 9, -49, -47, -31, 40, -38, -18, -1, 44, 8, -6, 55, -25, -106, -34, 7, 100, 41, 7, -100, 65, -24, 6, 52, -41, -9, -29, -30, 26, 29, 21, -34, -29, 0, 49, 1, -33, 28, 72, 34, 41, -13, 37, 36, -2, 69, 84, 85, 53, 1, -96, -111, 40, 10, -53, -54, -58, -95, -98, -93, -34, 87, -54, -14, 16, 78, 27, -50, -2, -59, 61, 67, -80, -79, 79, 27, 39, -15, 53, -9, 3, -30, -19, 23, 30, -20, -29, -30, 6, 59, 42, 33, 3, 10, -22, -26, -51, 20, 0, -5, 15, 36, 2, -61, -45, -36, -9, -38, 14, -9, 43, -39, -11, -33, 24, 24, 56, -13, -24, 49, 4, 29, -2, -29, -40, 53, 27, -45, -50, -23, -7, 19, 59, -72, 12, 14, 55, 52, -11, 30, -7, 14, 19, 60}

#define TENSOR_GRU_1_GRU_CELL_1_KERNEL_0_DEC_BITS {7}

#define TENSOR_GRU_1_GRU_CELL_1_RECURRENT_KERNEL_0 {-20, -6, -22, -5, -1, -6, -12, -23, 25, -17, -5, -14, 21, 9, 24, 26, 2, -27, 34, -11, 2, -13, 11, 10, -7, 1, -14, 21, -27, 2, 12, 2, 32, 39, 12, 27, -33, 5, -24, 26, -30, -45, -26, -39, 24, -33, 25, -3, -6, 30, -16, -18, -7, -22, -30, 13, -2, -12, 15, 43, -14, 5, -7, -11, 11, 5, 28, 24, 23, 21, -1, -23, -24, 7, -24, -15, -23, 1, -26, -15, 14, -30, -28, -30, -47, 4, -66, -10, 5, 38, -18, 3, -4, -6, -36, 14, 34, 18, 44, 38, -19, -11, -2, -20, 47, 13, 31, -17, -17, -7, -34, 11, -40, -26, -21, -33, -1, 11, -11, 17, 29, 40, 6, -6, -24, -9, 7, 4, 19, 24, -10, 13, -33, -38, -52, -26, 40, 9, 35, 22, 15, 26, 38, -20, -8, -37, -10, -19, 18, 14, 7, 3, 44, -14, -35, -49, -28, -28, -8, 6, 18, 0, -10, -7, 27, 41, -28, 9, -26, -14, -18, 4, 40, -27, 6, -32, 4, 35, 33, 7, -5, 0, 20, 6, 12, 14, 31, 38, 28, 36, -6, -6}

#define TENSOR_GRU_1_GRU_CELL_1_RECURRENT_KERNEL_0_DEC_BITS {6}

#define TENSOR_GRU_1_GRU_CELL_1_BIAS_0 {-28, 49, -79, -49, 41, 54, -4, -10, 104, 55, 66, 59, 55, 10, 111, 85, 17, 7, -8, 6, 4, 0, 11, -5, -28, 49, -79, -49, 41, 54, -4, -10, 104, 55, 66, 59, 55, 10, 111, 85, 17, 5, -9, 4, 6, 1, 11, -2}

#define TENSOR_GRU_1_GRU_CELL_1_BIAS_0_DEC_BITS {8}

#define GRU_1_BIAS_LSHIFT {6}

#define GRU_1_OUTPUT_RSHIFT {7}

#define TENSOR_DENSE_KERNEL_0 {-12, -4, 2, -20, 16, -4, -15, 35, -27, -18, 13, -7, 4, 3, -11, 13, -12, -15, -5, -12, -2, -10, -5, -1, -9, -25, 6, -4, -4, 9, -14, -1, -8, 0, 10, 4, 14, 16, -5, 20, -2, -27, 2, -8, 29, 3, -1, 26, -10, -4, 3, -15, -11, 2, -7, 4, -31, -20, 12, 4, -17, -6, -5, 1, -12, -25, 20, 15, 9, 4, -8, 2, -4, -12, 10, 1, 14, 10, -5, 5, -10, -4, 29, -1, 3, 27, -3, 9, -20, -9, 14, 5, 4, 19, -2, -5, -3, -14, 24, 16, 4, 20, 7, -3, -12, -15, 8, 7, 6, 19, 0, 8, -3, -15, 6, 4, 18, 16, 11, 30, 1, -3, -8, -2, 7, 8, 12, 23, -8, -22, 13, 15, 13, 7, -1, 10, -5, -28, 1, -1, -1, 25, 5, 8, 5, -24, 17, 4, 16, 3, 11, 19, -1, -25, -2, -13, 15, 16, 5, 24, -8, -9, 21, 0, 22, 27, 17, 4, -4, -27, 15, 5, 22, 27, -2, 6, -10, -18, 10, -4, 29, 5, 4, 4, 1, -14, -7, -6, 22, 2, 9, 11, -3, -22, 10, -19, 4, 14, 9, 30, -7, -27, -1, 11, 1, 26, 8, -4, -17, -9, 11, 8, 13, 22, 10, 4, 1, -13, 5, 0, 11, 19, 9, 8, 4, -24, 5, 1, 12, 9, -3, 15, 8, -29, 20, 1, 8, 10, -7, 19, -18, -26, -17, 9, 17, 32, -1, 6, 1, -30, 17, -6, 16, 6, 1, 3, -10, -33, 4, -13, 18, 20, 6, 6, 4, -9, -3, -19, 16, 15, 4, 31, -2, -15, 4, 17, 0, 20, -4, 17, -6, -30, 9, -4, 12, 9, 2, 2, -6, -30, 6, -3, 8, 11, 10, 10, -6, -23, 4, -11, 17, 21, -7, 10, -3, -16, -18, -7, 14, 15, 5, 19, -2, -30, -1, 5, -2, -1, 7, 9, -8, -27, 27, 12, 15, 6, 0, 16, 1, -14, 11, -10, 9, 14, 3, -12, -9, -22, 13, 1, 20, 19, 8, 28, -13, -15, -4, 2, 11, 11, -3, 9, -12, -15, 12, 16, 15, 7, -3, 3, 3, -17, -7, 16, 20, 18, 15, -4, 1, -16, 14, 13, 14, 13, 1, 11, 3, -24, 23, 2, 17, 10, -4, 27, -1, -27, 21, -11, 20, 24, 5, 17, -5, -3, -2, 8, 5, 19, -5, -6, -1, -14, 0, -10, 23, 26, -3, 6, 6, -6, 13, -7, 8, 15, 10, 10, -8, -12, 12, -14, 17, 25, -6, 23, -3, -17, 3, 8, 1, 35, 13, 5, -6, -29, 21, 10, 22, 7, 8, 15, 7, -30, 7, 13, 1, 17, 6, 13, -2, -16, 0, -7, 16, 26, -2, 22, -8, -10, 11, -14, 21, 16, 0, 25, 3, -20, 6, -18, 17, 13, -1, -2, -2, -28, -2, 5, 6, 29, 3, 6, -16, -14, 1, -18, 14, 29, 10, 18, 5, -31, -3, 4, 6, 27, 6, 18, -8, -25, 3, -15, 16, 16, -5, 20, 1, -2, 9, -3, 3, 7, 4, 8, -9, -17, 5, 0, 6, 22, 8, -5, -1, -8, 9, 2, 26, 19, -1, 16, 15, -20, -2, -9, 10, 27, -14, 22, -13, -25, 9, -3, 22, 24, -1, 13, -11, -34, -3, 5, 14, 18, 4, -1, -5, -34, 19, -5, 14, 7, 6, 20, -1, -10, 2, -1, 22, 22, 3, 17, 3, -20, 4, -11, 15, 19, -2, 20, -10, -42, 3, -15, 24, 13, 16, 8, -11, -32, 25, -15, 18, 22, 7, 9, -2, -41, 13, -2, 32, 38, 16, 16, -1, -50, 12, -13, 26, 44, 13, 29, 8, -49, 14, -6, 41, 49, 30, 31, 19, -72, 23, -40, 57, 54, 14, 44}

#define TENSOR_DENSE_KERNEL_0_DEC_BITS {8}

#define TENSOR_DENSE_BIAS_0 {-87}

#define TENSOR_DENSE_BIAS_0_DEC_BITS {11}

#define DENSE_BIAS_LSHIFT {4}

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
#define CONV1D_3_OUTPUT_DEC 5
#define CONV1D_3_OUTPUT_OFFSET 0
#define BATCH_NORMALIZATION_3_OUTPUT_DEC 5
#define BATCH_NORMALIZATION_3_OUTPUT_OFFSET 0
#define RE_LU_3_OUTPUT_DEC 5
#define RE_LU_3_OUTPUT_OFFSET 0
#define DROPOUT_3_OUTPUT_DEC 5
#define DROPOUT_3_OUTPUT_OFFSET 0
#define MAX_POOLING1D_3_OUTPUT_DEC 5
#define MAX_POOLING1D_3_OUTPUT_OFFSET 0
#define GRU_OUTPUT_DEC 7
#define GRU_OUTPUT_OFFSET 0
#define GRU_1_OUTPUT_DEC 7
#define GRU_1_OUTPUT_OFFSET 0
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

const nnom_shape_data_t tensor_gru_gru_cell_kernel_0_dim[] = {4, 24};
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

const nnom_shape_data_t tensor_gru_gru_cell_recurrent_kernel_0_dim[] = {8, 24};
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

const nnom_shape_data_t tensor_gru_gru_cell_bias_0_dim[] = {2, 24};
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
    .units = 8
};

const nnom_rnn_config_t gru_config = {
    .super = {.name = "gru"},
    .return_sequence = true,
    .stateful = false,
    .go_backwards = false
};
const int8_t tensor_gru_1_gru_cell_1_kernel_0_data[] = TENSOR_GRU_1_GRU_CELL_1_KERNEL_0;

const nnom_shape_data_t tensor_gru_1_gru_cell_1_kernel_0_dim[] = {8, 24};
const nnom_qformat_param_t tensor_gru_1_gru_cell_1_kernel_0_dec[] = TENSOR_GRU_1_GRU_CELL_1_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_gru_1_gru_cell_1_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_gru_1_gru_cell_1_kernel_0 = {
    .p_data = (void*)tensor_gru_1_gru_cell_1_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_gru_1_gru_cell_1_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_gru_1_gru_cell_1_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_gru_1_gru_cell_1_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};
const int8_t tensor_gru_1_gru_cell_1_recurrent_kernel_0_data[] = TENSOR_GRU_1_GRU_CELL_1_RECURRENT_KERNEL_0;

const nnom_shape_data_t tensor_gru_1_gru_cell_1_recurrent_kernel_0_dim[] = {8, 24};
const nnom_qformat_param_t tensor_gru_1_gru_cell_1_recurrent_kernel_0_dec[] = TENSOR_GRU_1_GRU_CELL_1_RECURRENT_KERNEL_0_DEC_BITS;
const nnom_qformat_param_t tensor_gru_1_gru_cell_1_recurrent_kernel_0_offset[] = {0};
const nnom_tensor_t tensor_gru_1_gru_cell_1_recurrent_kernel_0 = {
    .p_data = (void*)tensor_gru_1_gru_cell_1_recurrent_kernel_0_data,
    .dim = (nnom_shape_data_t*)tensor_gru_1_gru_cell_1_recurrent_kernel_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_gru_1_gru_cell_1_recurrent_kernel_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_gru_1_gru_cell_1_recurrent_kernel_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};
const int8_t tensor_gru_1_gru_cell_1_bias_0_data[] = TENSOR_GRU_1_GRU_CELL_1_BIAS_0;

const nnom_shape_data_t tensor_gru_1_gru_cell_1_bias_0_dim[] = {2, 24};
const nnom_qformat_param_t tensor_gru_1_gru_cell_1_bias_0_dec[] = TENSOR_GRU_1_GRU_CELL_1_BIAS_0_DEC_BITS;
const nnom_qformat_param_t tensor_gru_1_gru_cell_1_bias_0_offset[] = {0};
const nnom_tensor_t tensor_gru_1_gru_cell_1_bias_0 = {
    .p_data = (void*)tensor_gru_1_gru_cell_1_bias_0_data,
    .dim = (nnom_shape_data_t*)tensor_gru_1_gru_cell_1_bias_0_dim,
    .q_dec = (nnom_qformat_param_t*)tensor_gru_1_gru_cell_1_bias_0_dec,
    .q_offset = (nnom_qformat_param_t*)tensor_gru_1_gru_cell_1_bias_0_offset,
    .qtype = NNOM_QTYPE_PER_TENSOR,
    .num_dim = 2,
    .bitwidth = 8
};

const nnom_gru_cell_config_t gru_1_gru_cell_config = {
    .super = {.name = "gru_1"},
    .weights = (nnom_tensor_t*)&tensor_gru_1_gru_cell_1_kernel_0,
    .recurrent_weights = (nnom_tensor_t*)&tensor_gru_1_gru_cell_1_recurrent_kernel_0,
    .bias = (nnom_tensor_t*)&tensor_gru_1_gru_cell_1_bias_0,
    .q_dec_z = 5,
    .q_dec_h = 7,
    .units = 8
};

const nnom_rnn_config_t gru_1_config = {
    .super = {.name = "gru_1"},
    .return_sequence = true,
    .stateful = false,
    .go_backwards = false
};

const nnom_flatten_config_t flatten_config = {
    .super = {.name = "flatten"}
};
const int8_t tensor_dense_kernel_0_data[] = TENSOR_DENSE_KERNEL_0;

const nnom_shape_data_t tensor_dense_kernel_0_dim[] = {624, 1};
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
	nnom_layer_t* layer[20];

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
	layer[15] = model.hook(rnn_s(gru_cell_s(&gru_1_gru_cell_config), &gru_1_config), layer[14]);
	layer[16] = model.hook(flatten_s(&flatten_config), layer[15]);
	layer[17] = model.hook(dense_s(&dense_config), layer[16]);
	layer[18] = model.active(act_sigmoid(DENSE_OUTPUT_DEC), layer[17]);
	layer[19] = model.hook(output_s(&output0_config), layer[18]);
	model_compile(&model, layer[0], layer[19]);
	return &model;
}
