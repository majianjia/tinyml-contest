#include "nnom.h"

/* Weights, bias and Q format */
#define TENSOR_CONV1D_KERNEL_0 {5, -10, -21, -26, -22, 21, 16, 16, 3, -9, -14, 61, -61, 5, 35, -1, 4, -16, -16, -30, 45, -69, 49, 13, -7, -9, 4, 17, 25, 21, 3, 13, 18, 22, 17, -5, -13, -22, 51, -60}

#define TENSOR_CONV1D_KERNEL_0_DEC_BITS {3}

#define TENSOR_CONV1D_BIAS_0 {-24, -9, 5, -62, -105, -27, 11, -12}

#define TENSOR_CONV1D_BIAS_0_DEC_BITS {8}

#define CONV1D_BIAS_LSHIFT {2}

#define CONV1D_OUTPUT_RSHIFT {6}

#define TENSOR_CONV1D_1_KERNEL_0 {47, 31, 0, -43, -53, -69, -41, -48, -37, 57, 51, 57, 33, 13, -24, -28, -55, -38, -43, -33, -13, 40, 66, 84, 42, 7, -65, 54, 54, 66, 30, 10, -40, -47, -47, 6}

#define TENSOR_CONV1D_1_KERNEL_0_DEC_BITS {5}

#define TENSOR_CONV1D_1_BIAS_0 {-10, -69, -47, -64}

#define TENSOR_CONV1D_1_BIAS_0_DEC_BITS {10}

#define CONV1D_1_BIAS_LSHIFT {2}

#define CONV1D_1_OUTPUT_RSHIFT {8}

#define TENSOR_CONV1D_2_KERNEL_0 {25, 16, -8, -24, -26, -22, -13, -5, 4, 4, 15, 17, 22, -10, -12, -3, 6, 19, 21, 16, 18, 7, -17, -30, -24, -8, -41, 46, -29, 72, -90, 102, -50, 7, 14, -48, 65, -72, 21, 28, -21, 25, 33, -32, 45, 82, -110, 57, 50, 16, -20, 63}

#define TENSOR_CONV1D_2_KERNEL_0_DEC_BITS {4}

#define TENSOR_CONV1D_2_BIAS_0 {5, -23, -64, -119}

#define TENSOR_CONV1D_2_BIAS_0_DEC_BITS {8}

#define CONV1D_2_BIAS_LSHIFT {3}

#define CONV1D_2_OUTPUT_RSHIFT {7}

#define TENSOR_CONV1D_3_KERNEL_0 {-40, -22, -38, -51, -41, -38, -23, -43, -11, -29, 20, 59, -17, 31, -14, -34, 35, -37, -21, 31, -38, 16, -3, -45, 53, 40, 12, -27, -8, -30, -41, -19, 46, 59, 8, 49, 7, 33, 53, -34, 6, 32, 35, 60, 6, 67, -20, -17, 37, 27, -21, 31, -25, 22, 6, -23, 74, 30, 13, 30, 37, 30, 6, -35, 33, 18, -22, 31, -12, 16, 35, -8, 4, 23, 20, 41, 1, 25, -25, -21, -41, -37, -49, -61, -34, -23, 15, -24, -25, 45, 15, 61, 19, -25, -21, -59, -22, -6, 18, -2, -2, -8, -11, 11, -32, -21, -21, -4, -41, -14, -3, 26, 5, -13, 10, -11, 15, -6, -21, 21, -23, -25, -9, 2, -16, -11, 2, 13, -13, -14, 6, -11, 9, -17, -12, -7, -29, -27, -22, -27, -32, -39, 18, -7, -33, -29, 39, 29, 27, -26, -12, 41, -35, -32, -14, -7, -54, -9, 14, 35, -10, -21, 16, 8, 43, -13, 8, 32, -4, -36, -4, -11, -6, -31, 23, 40, -8, -29, 30, -24, 39, -15, -33, 8, -35, -24, -13, -34, -38, -11, 42, 6}

#define TENSOR_CONV1D_3_KERNEL_0_DEC_BITS {9}

#define TENSOR_CONV1D_3_BIAS_0 {-16, -19, 110, 49}

#define TENSOR_CONV1D_3_BIAS_0_DEC_BITS {8}

#define CONV1D_3_BIAS_LSHIFT {5}

#define CONV1D_3_OUTPUT_RSHIFT {8}

#define TENSOR_GRU_GRU_CELL_KERNEL_0 {52, 14, -6, -8, 29, 14, -6, -46, -13, 97, -52, -18, -9, 27, 3, -40, 75, 22, -74, 35, 18, -24, -25, 34, -24, -1, 6, 18, -12, -20, 25, -49, -39, 23, -38, -40, 28, -84, -12, -67, 13, -15, -16, -5, 41, 38, -5, 8, 76, -57, -58, -14, -12, -15, -24, -63, 40, 41, 51, 81, 41, 53, 93, 36, -27, 77, -126, -2, 47, -19, 10, -14, 22, 76, 74, 72, -5, 73, 26, 77, 16, 64, 4, 108, 42, 61, -62, 16, 61, 10, -33, -61, 71, 69, 76, 34, -15, 43, 25, 81, -29, -46, -58, -6, 12, 11, 39, -38, -3, 37, 22, 24, 2, -17, -75, -45, -53, -79, -5, -5, 14, 0, 1, -19, 33, -3, 19, 22, -16, 1, -21, -22, -29, 63, 11, -1, 42, 23, -35, -3, -16, -34, 33, -29}

#define TENSOR_GRU_GRU_CELL_KERNEL_0_DEC_BITS {7}

#define TENSOR_GRU_GRU_CELL_RECURRENT_KERNEL_0 {5, -2, -9, 16, -12, -51, 2, -2, 18, -8, 11, 3, 14, -13, -3, -12, 3, -19, -21, 1, -24, 8, -33, 37, 33, -8, 3, -20, -4, -16, -3, -30, -24, 9, 6, 3, -5, 56, 7, -3, -3, -4, -4, 28, -2, -18, 0, 43, -1, -2, -1, -1, -38, -21, 9, 3, 3, 11, 8, 10, -15, -13, -10, 11, -7, -5, -24, -5, -11, 8, -11, 19, 4, 6, 27, -16, -20, 9, 7, -1, -1, -7, 1, 4, 23, -20, -17, -2, 37, -27, 14, 22, 3, 14, 18, 5, 41, 14, -9, 71, -5, 14, -2, -8, 23, -11, 9, -71, 3, 6, -5, 4, -23, -16, -29, -2, 9, -12, -13, 0, 2, -3, -18, -4, 1, -7, 8, 37, -49, -27, -20, 2, -3, 2, -6, 8, 16, 21, -41, -29, 26, 46, -11, 8, -7, -9, -10, -6, 1, -16, -31, 3, 6, 5, 17, 16, 5, 13, 25, 16, 48, 21, 21, 45, 26, 5, 8, 22, 9, 9, 26, 2, 15, -5, 4, 17, 28, 23, 43, 48, 12, 72, 20, 30, -9, 8, 6, 16, 16, -10, -9, -12, -12, -4, -11, 9, 3, -29, 8, -8, 38, -6, 15, 19, -5, 7, 18, 15, 21, 3, -4, -12, 8, 14, -7, 31, 21, 14, 20, 31, 22, 18, -21, 5, 20, 24, 19, 18, 27, 38, 10, 19, 19, 23, -18, -12, -22, -13, 17, 30, 9, 20, 10, 0, 11, 38, 13, 20, -5, -2, -25, -1, -15, 7, -2, -15, 0, 20, 5, -20, 7, -6, -9, -2, 20, -7, -7, -6, 30, 26, -10, -4, -34, -28, -24, -18, -10, -10, 12, -7, -27, -14, 3, -6, 9, 16, -8, -34, 13, -6, 7, -4, 9, -7, -12, -29, -29, -21, -21, -17, 8, 2, 0, -3, -48, -18, -28, 4, 22, 4, -6, 34, 3, -9, -17, -31, 22, 11, 21, 13, 16, 4, -6, 13, -13, -8, -3, -1, -5, -3, 5, 5, 7, -5, -40, -3, -19, -5, -17, -12, -26, -3, -2, -14, -3, 18, 13, -15, 4, 6, 13, 5, 38, 33, 7, 24, -13, 15, 28, -1, 28, 29, 16, 2, 20, 7, 1, -3, -19, 10, -3, -19, -16, -13, -21, 7, -8, 15, -39, 3, 6, 8, 5, -34, 4, 29, -20, -11, -6, -18, 22, -34, 66, 19, 11, -5, -3, 30, 1, -10, 42, -3, -7, -29, 4, 41, 23, 6, -12, 2, 34, 6, 25, -5, -7, 18, 16, -1, 26, 17, -16, 12, -16, 42, 14, 4, -23, -41, 10, -8, -9, 29}

#define TENSOR_GRU_GRU_CELL_RECURRENT_KERNEL_0_DEC_BITS {6}

#define TENSOR_GRU_GRU_CELL_BIAS_0 {22, -19, -40, -33, -22, 48, 55, -65, 16, 43, -8, -25, 7, -8, 41, -9, 52, 31, 23, 21, 51, 32, 42, 70, -5, 5, -1, -5, -2, 4, -2, -9, 6, -3, -7, 4, 22, -19, -40, -33, -22, 48, 55, -65, 16, 43, -8, -25, 7, -8, 41, -9, 52, 31, 23, 21, 51, 32, 42, 70, -5, 6, -2, -5, -2, 4, -2, -11, 6, -2, -8, 3}

#define TENSOR_GRU_GRU_CELL_BIAS_0_DEC_BITS {7}

#define GRU_BIAS_LSHIFT {5}

#define GRU_OUTPUT_RSHIFT {5}

#define TENSOR_DENSE_KERNEL_0 {-1, -23, 4, -30, -21, 3, 5, -12, -30, -34, 6, 7, 3, -15, -3, 17, -14, -23, -23, 17, -1, 5, 20, -4, -24, -12, -15, -5, 19, 13, -21, 11, -25, -12, -1, 12, -6, -16, 16, -22, -11, -12, -35, 13, -26, -17, 20, 21, 13, 8, -29, -25, -36, -21, -42, -1, -1, 4, -1, 9, 26, 21, -3, -21, -38, -5, -51, 21, -16, -19, 7, 6, -1, -9, -8, 13, -17, -12, 8, -4, -12, 1, 4, 7, 9, -7, 1, 6, -22, -12, -52, 15, -19, 0, 1, 13, 5, 2, -27, -29, -26, 16, -27, 7, -17, 0, 5, 19, 12, -14, -5, -21, -36, 3, -42, 3, -3, 17, 17, 17, 17, 4, -26, -30, -39, 20, -27, 21, 13, 12, 53, 1, 7, -15, 10, 6, 0, -12, -28, 7, 5, 10, -11, 15, 41, 5, -21, -28, -23, 4, -38, 17, -10, 6, 4, 19, 18, -6, -11, -39, -30, 6, -46, 0, -4, 7, 19, 16, 30, 0, -5, -24, -39, 20, -77, 8, 5, 12, 25, 16, 22, 10, -32, 6, -48, -5, -42, 23, 7, 9, 18, -3, -20, -32, 6, -40, -21, 7, -28, 18, 5, 7, 11, 23, 32, -23, 8, -36, -15, 20, -5, 3, 0, -3, -5, 20, 18, 7, -17, -43, -28, -14, -36, 8, -2, 44, -8, 22, 41, -28, -42, -20, -42, 3, -35, 18, 14, -24, -6, 6, -6, -8, -9, -15, -22, 4, -44, 13, 15, 10, 8, 11, 9, -14, -9, -12, -31, -7, -46, 10, 2, 26, 14, 24, 9, 0, 25, -20, -29, 6, -73, -5, 16, 19, -1, 7, 30, 2, -11, -34, -14, 0, -58, -2, -15, 3, 5, 11, 40, -21, -28, -42, -30, 4, -26, 22, -7, 12, 20, 2, 6, -13, 10, -11, -22, -21, -25, 1, -7, 8, -20, 25, 22, -32, -3, -1, -14, -4, -36, 18, -5, 8, -26, 22, 26, -6, -4, -54, -28, 2, -49, -22, -2, 26, 3, 7, 18, -23, -4, -23, -46, -16, -47, 21, 6, 9, 14, 20, 24, -13, -25, -24, -28, -15, -33, 16, 3, 3, 4, -3, 13, -2, 30, -10, -31, -10, -23, 9, -1, 17, -2, 20, 33, -14, -12, -5, -8, 21, -24, 1, -12, -8, 17, 20, -2, -16, -8, -31, -46, -28, -32, -9, -8, 21, 13, 8, 29, 6, -30, -27, -22, -11, -40, 25, 1, 14, 2, 9, 16, -8, -4, -5, -33, -24, -44, 21, 13, 16, -4, -1, 18, -15, 14, -14, -35, 19, -47, 10, 1, 23, -6, 29, 32, -19, -25, -32, -24, 29, -48, -10, -3, 17, 5, 9, 43, -1, 14, -27, -25, 25, -38, -8, 4, 10, 19, 26, 13, 4, -6, -47, -16, -18, -44, 31, 9, -17, 42, 6, 38, -10, 13, 5, -48, -20, -34, 5, 13, 15, -12, 19, 21, -7, -1, -10, -37, -10, -60, 10, -12, 9, 5, 20, 47, -13, -17, -51, -4, -27, -62, 1, -1, 19, 13, 32, 33, -1, -9, -54, -26, -2, -31, -9, 1, 7, -16, 7, 24, 22, -26, -36, -31, 4, -23, 29, 14, -15, 24, 5, 25, -26, 10, 21, -36, -21, -22, -6, -2, 13, 1, 27, 11, -30, 0, -28, -10, 3, -58, -27, -8, 20, 18, 18, 47, 5, 10, -8, -33, 17, -51, 10, 10, 24, 9, 26, 42, -3, -15, -27, -39, -19, -38, -15, -7, 10, 9, 21, 22, 4, 15, -10, -32, 2, -18, -2, 7, 13, 20, 0, 26, -10, -9, -11, -12, -5, -42, 14, 14, 24, 6, 12, 24, -2, -8, -42, -24, 19, -64, 6, 10, 26, -16, 5, -3, -22, -13, -51, -35, -6, -41, -10, -7, 43, 19, 15, 55, 6, -13, -34, -47, -27, -19, 31, 12, 2, 27, 19, -4, -19, 11, -20, -12, -2, -21, 17, -8, 14, 16, 5, 44, 5, 15, -7, -10, 17, -40, 8, -1, 16, 5, 9, 21, 12, 28, -47, -47, -4, -55, 14, 1, 7, -3, 3, 52, -23, 6, -7, -41, -17, -44, 8, 7, 8, -1, 20, 22, -21, -20, -37, -44, 14, -67, 22, 2, 19, -7, -23, 5, -19, 8, -8, -10, -14, -34, 5, 18, 16, 24, 15, 49, -41, 5, -29, -22, -9, -60, -19, -8, 1, -17, 3, 18, -1, -1, -22, -43, -21, -67, -1, 19, 12, -27, 14, 37, -27, -24, -42, -45, -27, -54, -2, 5, 16, -7, -4, 32, -21, 4, -23, -47, -26, -48, 18, 23, 13, 48, -12, 27, -27, -8, -7, -32, 14, -44, 17, 1, 9, -37, 11, 34, -15, 2, -49, -39, -11, -34, 16, 2, 20, 15, 24, 53, -7, -16, -65, -38, -6, -70, 13, 3, 19, -10, 21, 43, 13, 7, -35, -36, -29, -50, 11, -14, 4, 8, -5, 37, 5, 4, -18, -31, -28, -47, 0, 2, 21, -2, 6, 34, -17, -10, -34, -29, -8, -39, -9, -1, 21, -33, 2, 52, -7, -7, -30, -28, -1, -68, 3, 0, 33, 18, -8, 49, -7, -9, -12, -51, -14, -109, -2, 5, 18, 3, 21, 23, 11, -27, -39, -41, -18, -72, 26, 23, 34, 7, -5, 38, -20, 21, -19, -49, -39, -49, 10, -13, 36, -39, 8, 51, -1, 18, -47, -36, -32, -66, 17, -8, 37, -10, 25, 52, 16, 0, -61, -51, -34, -87, 12, 11, 46, -26, 24, 87, 18, -27, -29, -73, -74, -74, 2, 8, 66, -12, 13, 72, -8, -4, -38, -67, -87, -95, 48, 40, 92, -52, 5, 61, 4, -39, -61, -64, -89, -96, 12, 30, 96, -88, 26}

#define TENSOR_DENSE_KERNEL_0_DEC_BITS {9}

#define TENSOR_DENSE_BIAS_0 {-68}

#define TENSOR_DENSE_BIAS_0_DEC_BITS {11}

#define DENSE_BIAS_LSHIFT {5}

#define DENSE_OUTPUT_RSHIFT {13}


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
