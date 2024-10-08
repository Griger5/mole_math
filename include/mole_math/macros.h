#define DEFAULT_CHAR 'a'
#define GET_SECOND_ARG(a1, a2, ...) a2
#define GET_FLAG(...) GET_SECOND_ARG(0, ##__VA_ARGS__, DEFAULT_CHAR)

// cuda macros

#define GLOBAL_IDX_X() threadIdx.x + blockIdx.x * blockDim.x
#define GLOBAL_STRIDE_X() gridDim.x * blockDim.x

#define GLOBAL_IDX_Y() threadIdx.y + blockIdx.y * blockDim.y