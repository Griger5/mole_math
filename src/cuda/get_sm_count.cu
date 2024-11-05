#include <stdio.h>

int main(void) {
    int deviceId;
    int num_of_SM;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, deviceId);

    printf("%d", num_of_SM);

    return 0;
}