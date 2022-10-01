#include "nnom.h"

/* Weights, bias and Q format */
#define TENSOR_CONV1D_KERNEL_0 {26, -54, 31, 15, -25, -38, 35, 13, -51, 23, 18, 21, 12, -7, -15, 10, -35, 55, -42, 20, 18, 11, 22, 13, 19, -28, -7, -12, -12, 9, -26, -1, 25, 17, -39, -1, -22, -2, -4, 7, -29, 8, 5, 27, -26, 24, 13, 11, -12, -26, -25, -21, -9, 4, 17, -12, 7, 13, 24, 20, 33, -54, 38, -11, 6, -7, -25, -21, 7, 13, -30, 66, -30, -22, 39, -6, -6, 36, -51, 26}

#define TENSOR_CONV1D_KERNEL_0_DEC_BITS {3}

#define TENSOR_CONV1D_BIAS_0 {-15, -24, 3, -19, 0, 0, -34, -16, -19, 1, -11, -5, -67, -34, -1, -17}

#define TENSOR_CONV1D_BIAS_0_DEC_BITS {7}

#define CONV1D_BIAS_LSHIFT {3}

#define CONV1D_OUTPUT_RSHIFT {6}

#define TENSOR_CONV1D_1_KERNEL_0 {42, -7, -3, -44, -16, -28, 7, 17, 26, 54, -62, -7, 113, -55, -7, 41, -26, 64, 73, -53, -57, 85, 14, -19, -59, 76, -35, -36, -34, -8, -9, 7, -1, 25, 13, 30, 45, 25, 3, -22, -19, -22, -10, 19, 16, -75, -36, 44, -41, -48, -1, -60, 37, -80, 26, 28, 8, 8, -19, -29, -21, -27, -26, -10, 0, 33, 28, 35, 5, 13, -25, -37}

#define TENSOR_CONV1D_1_KERNEL_0_DEC_BITS {4}

#define TENSOR_CONV1D_1_BIAS_0 {26, -84, -5, 5, -15, -43, 20, 16}

#define TENSOR_CONV1D_1_BIAS_0_DEC_BITS {8}

#define CONV1D_1_BIAS_LSHIFT {3}

#define CONV1D_1_OUTPUT_RSHIFT {7}

#define TENSOR_CONV1D_2_KERNEL_0 {10, 42, -48, 43, -39, -28, 10, 16, 32, -62, 31, -8, 29, -38, 12, 35, -34, -3, 50, -53, 8, 42, 41, -100, 73, 34, -11, -15, -26, -15, -14, -2, 21, 25, 20, 16, 8, -9, -23, -25, -17, 12, -11, -6, 1, 12, 2, 13, 1, -26, -41, -34, -11, 2, 24, 19, 28, 10, 14, 1, -5, -16, -20, -31, -18, -37, -22, -3, -6, -12, -17, 11, 1, 16, 28, 35, 21, 3, 21, 7, -4, -16, -24, -19, -15, -18, -8, 2, 22, 26, 19, -11, 14, 2, -11, -21, -19, -16, -13, 11, 10, 17, 20, 26}

#define TENSOR_CONV1D_2_KERNEL_0_DEC_BITS {4}

#define TENSOR_CONV1D_2_BIAS_0 {16, -100, -63, -82, -46, -71, -34, -12}

#define TENSOR_CONV1D_2_BIAS_0_DEC_BITS {10}

#define CONV1D_2_BIAS_LSHIFT {1}

#define CONV1D_2_OUTPUT_RSHIFT {7}

#define TENSOR_CONV1D_3_KERNEL_0 {-13, -22, 43, -60, -31, 34, -43, -18, -59, 74, 48, 16, -51, 5, -22, -56, 20, -66, -50, -55, 62, -24, -16, 40, -55, -14, 48, 14, 21, 12, -53, -6, 25, -58, 36, -32, 35, 42, -44, 31, 3, 7, 10, -23, -48, -42, -26, -30, -35, -26, -60, 53, 3, -13, 15, 26, -19, -9, 0, -4, 62, 43, 30, -5, -45, 19, -28, -2, -27, 28, 36, 38, -49, 67, 16, -32, -26, 61, -20, -72, 36, -18, 27, -39, 50, -15, -10, 40, -34, -23, 9, 14, 28, 71, 80, -38, 0, 12, 10, -13, 29, -40, -12, -36, 21, 50, -9, 38, -37, 33, -23, 5, 21, 9, 5, 29, -3, -4, 27, 42, -23, 6, 8, 17, -18, -34, -18, 37, -43, -42, -10, -21, 30, 40, -13, 12, -35, 11, 57, -13, -20, 4, -27, -44, 25, -22, -31, 52, 36, -18, 9, 9, -17, -1, -7, -14, 17, 6, 29, 7, -31, -46, 47, -10, -31, 7, 7, -22, 28, 14, 19, 21, 27, 39, -11, -15, 47, -29, -26, 49, 5, -23, -24, -6, -33, -41, 57, 5, 29, 14, 41, -27, -1, 15, -27, -5, -24, 5, 5, 20, 10, -1, -26, -15, 41, -20, 9, 20, 3, -8, 25, 0, -18, 16, -25, 7, 6, 18, 4, -14, -5, 6, -12, -33, 22, 12, 11, 20, -29, 11, 7, -1, 0, 14, -4, -17, 38, 25, 5, -7, -3, -3, -3, -15, -16, -11, -12, -23, 7, 29, -20, -13, -17, -19, 2, -12, -13, 8, 9, 33, -31, 18, -7, 8, 12, -13, -25, -18, 1, 16, 8, 27, -32, 4, 8, -17, -16, 16, -35, -24, -21, -6, 2, 3, -37, -1, 4, -28, 11, -7, -2, -46, -13, -14, 0, 0, -9, 38, 12, 48, -14, 11, -28, 4, 37, -19, 25, -26, 8, -41, 64, 36, 34, -40, 31, 0, 50, 5, 58, 42, -29, -38, 18, -8, -11, 7, -4, 36, -5, 0, -24, 19, 6, 16, -12, -8, 12, -37, 4, 28, 26, 10, 39, 20, -17, -11, 44, 1, 14, 30, 23, -21, -8, -14, 29, -26, -26, 25, -62, 33, -48, 25, -2, -47, -3, 24, -57, -61, 21, -52, -26, -44, -10, -6, 12, -13, 10, -25, -11, -40, -31, 31, 0, -15, 20, 26, -4, -17, -3, -32, 14, -7, 27, 15, 16, -4, 24, 22, 12, 3, 16, 5, 9, -14, -13, 5, -30, -27, -16, 23, -34, -18, -42, -3, -9, -28, 39, -2, -25, 29, -25, -18, -15, -2, 27, 10, -20, 11, 10, -24, 0, 38, 12, 6, -4, 19, -6, 9, -23, -18, -33, -11, -19, 5, -31, 2, -16, -8, 19, 8, -16, 25, -24, -16, 32, 23, -14, -14, 0, -16, 25, 27, 45, 20, -19, 21, 21, -18, 0, 11, -26, 8, 10, 18, 4, -10, -4, -34, -36, -35, 37, 42, -24, 5, -40, 30, 24, -41, -25, -23, -1, -9, 15, -1, 26, -7, 13, -11, 53, -47, 23, -24, -7, -26, -34, -7, -9, -38, -15, 8, -18, 22, 8, 20, -41, 52, -2, 37, 50, 34, 22, 0, 4, -36, -12, -11, -1, 57, -32, 36, 10, -9, 19, 24, 4, 20, 39, 25, -38, -30, -35, -40, -35, -5, -10, 59, 9, 11, -37, 16, 22, -30, 0, -3, 2, -10, -2, -2, 63, 6, 13, 37, 28, -32, -14, -16, -28, -22, 26, -24, -45, -31, 5, 20, -26, -26, 1, 5, 14, -23, -18, 37, -11, 55, 29, 31, 53, 51, -45, 3, -7, 5, 27, -27, -52, 61, -9, -37, 42, 28, -6, 0, 10, -12, 69, -15, 16, 38, -22, 5, 16, 13, 58, 39, -21, 1, 43, -15, 2, 11, -27, 20, -57, -27, 21, -18, 12, -11, -5, -18, 43, 16, 37, -18, 20, -38, -29, -30, 21, -24, -48, -45, -19, -13, 28, 38, -27, 32, 8, -48, 23, -45, -1, 22, 5, -22, -5, -14, -12, 39, 30, 2, -51, 12, -56, 1, -35, 30, -22, -55, -46, 22, -1, 6, 3, 1, -16, 7, 13, -3, -20, -14, -23, -3, 7, -10, -12, 6, -1, -10, -9, 3, 1, -20, -6, -9, -7, -7, -9, -31, 1, -16, -19, 1, -6, 18, 9, 17, -15, 6, 1, 7, 5, -7, 0, 8, 3, -14, 6, 19, -24, -7, -11, 0, -2, -11, -20, -6, -5, -13, -27, -9, -8, -29, -7, -20, 2, 3, -9, 7, -16, -14, 6, 0, 1, 5, -9, 9, -8, -17, -7, -10, -9, 12, 5, 4, 15, 6, -4, -15, 0, 13, -10, -23, -6, -22, -4, -7}

#define TENSOR_CONV1D_3_KERNEL_0_DEC_BITS {9}

#define TENSOR_CONV1D_3_BIAS_0 {-29, -94, 77, -33, 76, 63, -17, 122}

#define TENSOR_CONV1D_3_BIAS_0_DEC_BITS {8}

#define CONV1D_3_BIAS_LSHIFT {5}

#define CONV1D_3_OUTPUT_RSHIFT {9}

#define TENSOR_GRU_GRU_CELL_KERNEL_0 {38, 31, -43, 23, -39, 3, 4, -10, 8, -1, -30, 25, -11, 25, 34, -29, -5, -62, -12, -23, -4, 38, -67, 11, 1, 31, 14, -53, 1, -47, 30, -3, -41, 16, 1, 26, 31, 3, -1, -33, 33, -32, -29, -11, 39, 18, -8, -34, 51, 1, -9, 31, 33, 36, 33, 20, -36, -1, 11, 4, 14, -31, -18, -7, -22, -7, 34, -37, 3, -39, -29, -30, 2, 21, -67, -3, 8, 38, 9, 27, -20, 34, -10, 45, 35, 33, 33, 9, -59, -45, 28, 35, -18, 4, -27, -18, -52, 11, -63, 19, -11, 36, -71, 30, -5, 13, 6, 35, 40, 24, 18, 17, 43, 36, 11, -21, 33, -22, 27, 31, 25, -18, -32, 34, 10, 10, 27, 27, 30, -61, 24, -49, -18, 86, -30, 56, 54, 59, 53, -13, 57, -22, -28, 26, 19, 43, 44, 55, 74, -19, 73, 27, 28, -43, -5, 2, -28, 18, 12, -4, -16, 26, 12, -40, -61, 32, -29, 34, 9, 27, 71, 17, 50, 49, -44, 62, 53, 61, -4, 60, 49, 76, 34, 55, 60, -28, 18, 36, -56, 70, 2, 51, 5, 58, 22, 28, 14, 68, -7, 59, 3, -22, -20, 10, 27, 1, 10, 37, 11, -6, 43, 30, 78, 16, 10, 1, 43, 25, -4, 41, 60, 26, 43, 56, 10, -3, -15, 5, 30, 35, 5, 54, 55, 19, 9, -15, -46, 17, 11, 29, 55, -42, -7, -46, -28, 57, -35, -6, -5, -19, -12, 37, -10, 4, -15, 50, 13, -8, -28, -41, 34, 38, 14, 0, 5, -30, -22, -50, 1, -42, -10, -16, 25, -14, -11, 3, 2, 0, -11, -30, -3, -11, 17, 7, -9, 20, -19, -29, -22, 19, -68, -23, 34, 14, 38, -24, 13, -32, 6, -29, -10, 34, 44, -22, -17, 8, 36, 5, -7, -26, -27, 31, 3, -39, -12, 16, 10, -9, -11, 38, -52, -21, -21, -19, -4, 42, -52, 1, 41, 39, -66, -61, -5, 21, -25, 35, 27, 31, -11, 7, 33, -35, -20, -12, -53, -20, -4, -2, -21, 10, 26, -12, -50, -14, -31, 50, 1, 17, -54, 29, 13, -3, -13, 50, -3, 18, -72, 18, 3, -4, -19, -1, -20, 32, -6, -7, 4, -4, 35, -30, -16, 20, -18, 19}

#define TENSOR_GRU_GRU_CELL_KERNEL_0_DEC_BITS {7}

#define TENSOR_GRU_GRU_CELL_RECURRENT_KERNEL_0 {-37, -39, -19, 10, -5, -8, -13, 29, 39, 4, -21, -5, 9, -38, 1, -12, -31, -14, 7, 2, -15, 20, -28, -6, 31, -7, -40, -9, 19, -4, -37, -5, -33, -6, -47, 19, -38, -32, -43, -3, -30, -21, 40, -13, -25, 22, 15, -6, -14, -3, -26, 15, -9, -31, -23, -16, -10, 56, 3, -28, -9, 26, 44, 36, 2, 3, 18, -26, -57, -51, -17, -25, -21, -20, -5, 28, 39, -56, 38, -58, -8, -2, 11, 12, -23, 10, 3, -52, -11, -23, -16, -21, 1, 12, 7, -12, 23, 42, 37, 19, 24, -2, -11, 7, 19, 45, -39, -21, -18, 16, -18, -23, -5, 4, 15, -3, -33, -23, -16, -3, -20, 17, -4, -43, 7, -18, 15, 34, -3, -23, -36, 8, -5, 13, -17, -27, 3, 32, -29, -28, 15, 10, -59, 1, 2, 24, -7, 7, 5, 42, -5, 0, -42, 35, 24, 10, -19, 33, -10, -64, 27, 5, -9, -35, -4, -22, 33, -6, 24, -22, -34, -29, -5, -6, 11, 17, -4, -9, 42, -15, 7, -45, -50, -52, 6, 14, 17, -3, 4, 9, 45, 40, -9, 6, 12, 6, -8, -31, 16, -26, 15, 17, -4, -13, -14, -16, -23, 42, -8, 25, -12, 10, 2, -56, 14, -34, 7, -8, -3, -16, -14, 77, -14, -61, 20, -35, 24, 23, 19, -36, -6, 4, -8, 4, -34, -27, -8, -53, 10, 46, 36, -38, -28, 6, -2, -28, 27, 34, 8, 15, 2, 0, 42, -27, 18, -18, 59, 43, 47, 22, 28, -40, 43, 31, -24, 58, -52, -12, 9, -1, 45, 3, 2, -5, 21, 37, -7, -29, 38, 7, -11, -40, 25, 56, -42, -17, 32, 16, 60, -6, 6, 41, 8, -41, 30, -3, 7, 58, -35, -30, 30, 6, -64, 22, 9, 34, 44, 29, 27, -1, 27, 20, -32, 8, 7, -30, -43, -9, -4, 14, 31, 51, -50, 22, 52, -18, 19, -7, -51, 29, -2, -5, 22, 1, 9, -4, -1, -26, 15, 22, 11, -21, 22, 32, 1, -58, 29, -3, -46, 12, 75, -7, -13, 19, -34, 35, 14, -47, 41, -45, -16, 9, 26, -47, 36, -28, -42, 1, 1, 24, -22, 35, -12, -4, 24, -4, -24, -10, -5, -14, 16, -68, -16, 20, 30, 11, 34, -30, 12, -36, 17, 34, 11, 22, -17, 25, 51, 47, -22, -6, -28, -11, 50, -8, 9, 26, 39, -19, -3, 23, 56, -7, -64, -19, 20, 16, 19, -72, 13, -48, 0, -3, -20, -11, 38, -39, -47, -3, 11, 31, 33, 3, -11, -22, 5, -17, -3, -22, 20, -2, -3, 15, 41, 20, -67, -17, 30, -8, 48, -19, 8, -56, -29, 46, -18, 55, 8, -46, -42, -16, 11, 63, 46, 9, 41, -19, 9, -11, 4, -5, -19, 16, -19, 6, 81, -3, -12, -37, -23, 45, 0, -49, 30, -23, 0, 43, -42, 46, 48, -58, -52, -1, -24, 11, 7, -30, 43, -10, 23, 0, -34, 33, 21, 50, -8, -16, 24, 17, -2, -33, 1, -18, 12, 23, -32, 41, 15, -20, 74, 15, -86, -52, -17, -45, 55, 49, 2, 27, 0, 52, -32, 34, -1, -22, 61, 34, 13, -47, -2, 13, -35, -11, 15, 71, 1, 65, 36, 53, 37, -5, 17, 15, -39, 37, 15, -33, 31, 22, -20, -39, -33, 36, -36, -39, 42, 37, 16, 31, -11, 26, 11, -12, 24, -18, 20, 20, -8, 57, -8, 46, 12, 61, -11, 21, -10, 9, -11, -43, -4, -41, 19, -99, 10, 17, -56, 3, -32, 22, -26, 48, -4, -25, -12, -20, 11, -49, 4, 41, 1, 36, 1, -8, -28, 33, -41, 58, -31, 25, 6, -5, 3, 35, 19, -42, 9, 33, -23, -27, -35, 26, 5, 16, 5, 17, -32, -16, -15, 15, -5, 32, -5, -21, 29, -44, -9, -34, 14, -10, -4, -21, 9, -26, -18, 61, -28, 18, 48, 20, -23, -24, 85, -60, 15, -31, -40, 32, -6, -29, -26, -15, -49, -16, 27, 32, -1, 13, 52, -52, 48, -42, -15, 4, -1, 48, 7, -31, -5, 8, 28, -1, -50, -42, 38, -21, -17, 15, 43, 34, 11, -17, -14, -10, -17, -6, -8, 51, 10, 14, 1, 13, -45, -56, 12, -14, -51, -28, 25, -58, 22, -18, 17, 12, 26, 15, 21, -2, -13, -42, -19, 3, -23, 62, 39, 49, -38, -11, 35, 18, 32, 19, 9, 15, -7, 22, 33, -31, -35, -10, -29, -16, 10, 4, 14, -31, -56, 8, -44, -28, -11, 14, 31, -14, -40, -14, -12, 21, -5, 41}

#define TENSOR_GRU_GRU_CELL_RECURRENT_KERNEL_0_DEC_BITS {7}

#define TENSOR_GRU_GRU_CELL_BIAS_0 {57, -52, 58, -22, -58, -17, 14, -40, -12, -4, -12, -1, -13, -39, -25, 67, 30, 44, 20, 80, 42, 33, -6, 106, 27, 26, 107, 66, 36, 7, 20, 27, 4, 14, 3, 17, -10, 12, 1, 21, -2, -16, 0, 15, -4, 10, -15, -18, 57, -52, 58, -22, -58, -17, 14, -40, -12, -4, -12, -1, -13, -39, -25, 67, 30, 44, 20, 80, 42, 33, -6, 106, 27, 26, 107, 66, 36, 7, 20, 27, 5, 11, -2, 22, -13, 7, 0, 21, -6, -17, -4, 22, -6, 10, -12, -18}

#define TENSOR_GRU_GRU_CELL_BIAS_0_DEC_BITS {8}

#define GRU_BIAS_LSHIFT {3}

#define GRU_OUTPUT_RSHIFT {4}

#define TENSOR_GRU_1_GRU_CELL_1_KERNEL_0 {-15, -44, -40, -1, -78, -31, -67, 55, 18, -8, 42, 38, 3, 7, -45, -52, -19, -48, 13, 36, 69, 29, -34, 18, 10, 10, -57, 29, 22, -73, -69, 56, -46, -16, -62, -25, 42, 3, 19, 75, 13, -50, 33, 75, -84, 90, 109, -48, -69, -59, 47, -9, -39, -19, 22, 31, -71, -73, -6, 10, -13, -9, 41, -36, -12, 25, 48, 29, 25, 50, -59, 9, 44, 49, 47, 43, 51, 5, 4, -36, 40, 0, -28, 52, -3, -37, 44, 9, -54, 25, 11, 41, -21, 64, 10, 21}

#define TENSOR_GRU_1_GRU_CELL_1_KERNEL_0_DEC_BITS {7}

#define TENSOR_GRU_1_GRU_CELL_1_RECURRENT_KERNEL_0 {11, 54, -43, 53, -60, -38, -47, -7, -73, 60, -62, -37}

#define TENSOR_GRU_1_GRU_CELL_1_RECURRENT_KERNEL_0_DEC_BITS {7}

#define TENSOR_GRU_1_GRU_CELL_1_BIAS_0 {66, 60, -25, 28, 6, -4, 66, 60, -25, 28, 10, -2}

#define TENSOR_GRU_1_GRU_CELL_1_BIAS_0_DEC_BITS {8}

#define GRU_1_BIAS_LSHIFT {6}

#define GRU_1_OUTPUT_RSHIFT {7}

#define TENSOR_DENSE_KERNEL_0 {-43, -5, -13, -25, 17, -10, -1, 27, 1, 19, 22, 45, -12, -8, -42, 22, 2, -10, -43, -17, 1, 60, -39, 29, 0, 6, 7, 29, -27, 36, -54, -11, -19, 17, 7, 20, -26, 49, -38, 19, -35, -12, 0, 15, -28, 23, -46, 36, -11, 12, 7, 53, -26, 26, -18, 56, -16, 3, -46, -4, -3, -6, -5, 47, -17, 32, -24, 49, -36, 42, 13, 39, -39, -11, -11, 37, -34, 33, -48, 36, -9, 7, -16, -3, -40, -3, -40, 26, -34, 5, 13, 44, -41, 40, -37, 34, -25, 32, -19, -2, -18, -9, -57, 52, -24, 6, -15, 2, -55, -5, -3, 33, -37, 46, -37, 12, -9, 54, -47, 7, -23, 0, -35, 15, -53, 60, -30, -1, 2, 46, -19, 2, -55, 41, -50, 12, -57, 5, -6, 6, -25, 57, -48, 46, -21, 50, -45, 49, -34, 49, -44, 68, -30, 32, -71, 44}

#define TENSOR_DENSE_KERNEL_0_DEC_BITS {8}

#define TENSOR_DENSE_BIAS_0 {-85}

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

const nnom_shape_data_t tensor_conv1d_kernel_0_dim[] = {5, 1, 16};
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

const nnom_shape_data_t tensor_conv1d_bias_0_dim[] = {16};
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
    .filter_size = 16,
    .kernel_size = {5},
    .stride_size = {2},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};
const int8_t tensor_conv1d_1_kernel_0_data[] = TENSOR_CONV1D_1_KERNEL_0;

const nnom_shape_data_t tensor_conv1d_1_kernel_0_dim[] = {9, 1, 8};
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

const nnom_shape_data_t tensor_conv1d_1_bias_0_dim[] = {8};
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
    .filter_size = 8,
    .kernel_size = {9},
    .stride_size = {2},
    .padding_size = {0, 0},
    .dilation_size = {1},
    .padding_type = PADDING_SAME
};
const int8_t tensor_conv1d_2_kernel_0_data[] = TENSOR_CONV1D_2_KERNEL_0;

const nnom_shape_data_t tensor_conv1d_2_kernel_0_dim[] = {13, 1, 8};
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

const nnom_shape_data_t tensor_conv1d_2_bias_0_dim[] = {8};
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
    .filter_size = 8,
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

const nnom_shape_data_t tensor_conv1d_3_kernel_0_dim[] = {3, 32, 8};
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

const nnom_shape_data_t tensor_conv1d_3_bias_0_dim[] = {8};
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
    .filter_size = 8,
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

const nnom_shape_data_t tensor_gru_gru_cell_kernel_0_dim[] = {8, 48};
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

const nnom_shape_data_t tensor_gru_gru_cell_recurrent_kernel_0_dim[] = {16, 48};
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

const nnom_shape_data_t tensor_gru_gru_cell_bias_0_dim[] = {2, 48};
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
    .units = 16
};

const nnom_rnn_config_t gru_config = {
    .super = {.name = "gru"},
    .return_sequence = true,
    .stateful = false,
    .go_backwards = false
};
const int8_t tensor_gru_1_gru_cell_1_kernel_0_data[] = TENSOR_GRU_1_GRU_CELL_1_KERNEL_0;

const nnom_shape_data_t tensor_gru_1_gru_cell_1_kernel_0_dim[] = {16, 6};
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

const nnom_shape_data_t tensor_gru_1_gru_cell_1_recurrent_kernel_0_dim[] = {2, 6};
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

const nnom_shape_data_t tensor_gru_1_gru_cell_1_bias_0_dim[] = {2, 6};
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
    .units = 2
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

const nnom_shape_data_t tensor_dense_kernel_0_dim[] = {156, 1};
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
