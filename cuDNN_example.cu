#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>

#include <cuda.h>
#include <cudnn.h>
#include <time.h>

#ifdef _WIN32
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;
using int64_t = long long;
using uint64_t = unsigned long long;
#else
#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long
#endif


#define CUDA_CALL(f) { \
    cudaError_t err = (f); \
    if (err != cudaSuccess) { \
      std::cout \
          << "    Error occurred: " << err << std::endl; \
      std::exit(1); \
    } \
  }
  
#define CUDNN_CALL(f) { \
cudnnStatus_t err = (f); \
if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
} \
}

void DisplayHeader()
{
    const int kb = 1024;
    const int mb = kb * kb;
    std::cout << "NBody.GPU" << std::endl << "=========" << std::endl << std::endl;

    std::cout << "CUDA version:   v" << CUDART_VERSION << std::endl;    
    
    int devCount;
    cudaGetDeviceCount(&devCount);
    std::cout << "CUDA Devices: " << std::endl << std::endl;

    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        std::cout << i << ": " << props.name << ": " << props.major << "." << props.minor << std::endl;
        std::cout << "  Global memory:   " << props.totalGlobalMem / mb << " MB" << std::endl;
        std::cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << " KB" << std::endl;
        std::cout << "  Constant memory: " << props.totalConstMem / kb << " KB" << std::endl;
        std::cout << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;

        std::cout << "  Warp size:         " << props.warpSize << std::endl;
        std::cout << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
        std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
        std::cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << std::endl;
        std::cout << std::endl;
    }
}


void print(const float *data, int n, int c, int h, int w) {
  std::vector<float> buffer(1 << 20);
  CUDA_CALL(cudaMemcpy(
        buffer.data(), data,
        n * c * h * w * sizeof(float),
        cudaMemcpyDeviceToHost));
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(7) << std::setprecision(5) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
      break;
    }
    break;
  }
  std::cout << std::endl;
}

__global__ void dev_const(float *px, float k) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = k;
}

__global__ void dev_iota(float *px) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = tid*0.001;
}

int main(){

  DisplayHeader();
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  // input
  const int in_n = 32;
  const int in_c = 32;
  const int in_h = 16;
  const int in_w = 16;
  std::cout << "in_n: " << in_n << std::endl;
  std::cout << "in_c: " << in_c << std::endl;
  std::cout << "in_h: " << in_h << std::endl;
  std::cout << "in_w: " << in_w << std::endl;
  std::cout << std::endl;

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));

  float *in_data;
  CUDA_CALL(cudaMalloc(
        &in_data, in_n * in_c * in_h * in_w * sizeof(float)));

  // filter
  const int filt_k = 64;
  const int filt_c = 32;
  const int filt_h = 3;
  const int filt_w = 3;
  std::cout << "filt_k: " << filt_k << std::endl;
  std::cout << "filt_c: " << filt_c << std::endl;
  std::cout << "filt_h: " << filt_h << std::endl;
  std::cout << "filt_w: " << filt_w << std::endl;
  std::cout << std::endl;

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filt_k, filt_c, filt_h, filt_w));

  float *filt_data;
  CUDA_CALL(cudaMalloc(
      &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));

  // convolution
  const int pad_h = 1;
  const int pad_w = 1;
  const int str_h = 1;
  const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;
  std::cout << "pad_h: " << pad_h << std::endl;
  std::cout << "pad_w: " << pad_w << std::endl;
  std::cout << "str_h: " << str_h << std::endl;
  std::cout << "str_w: " << str_w << std::endl;
  std::cout << "dil_h: " << dil_h << std::endl;
  std::cout << "dil_w: " << dil_w << std::endl;
  std::cout << std::endl;

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  // CUDNN_CONVOLUTION
  // In this mode, a convolution operation will be done when applying the filter to the images.
  
  // CUDNN_CROSS_CORRELATION
  // In this mode, a cross-correlation operation will be done when applying the filter to the images.

  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;
  
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));
  CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
  // CUDNN_DEFAULT_MATH
  // Tensor Core operations are not used on pre-NVIDIA A100 GPU devices. On A100 GPU architecture devices, Tensor Core TF32 operation is permitted.

  // CUDNN_TENSOR_OP_MATH
  // The use of Tensor Core operations is permitted but will not actively perform datatype down conversion on tensors in order to utilize Tensor Cores.

  // CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
  // The use of Tensor Core operations is permitted and will actively perform datatype down conversion on tensors in order to utilize Tensor Cores.

  // CUDNN_FMA_MATH
  // Restricted to only kernels that use FMA instructions.

  std::cout << "out_n: " << out_n << std::endl;
  std::cout << "out_c: " << out_c << std::endl;
  std::cout << "out_h: " << out_h << std::endl;
  std::cout << "out_w: " << out_w << std::endl;
  std::cout << std::endl;

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

  float *out_data_ref;
  CUDA_CALL(cudaMalloc(
        &out_data_ref, out_n * out_c * out_h * out_w * sizeof(float)));
  
  float *out_data;
  CUDA_CALL(cudaMalloc(
        &out_data, out_n * out_c * out_h * out_w * sizeof(float)));

  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
  // This algorithm expresses the convolution as a matrix product without actually explicitly forming the matrix that holds the input tensor data.
  
  // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
  // This algorithm expresses convolution as a matrix product without actually explicitly forming the matrix that holds the input tensor data, but still needs some memory workspace to precompute some indices in order to facilitate the implicit construction of the matrix that holds the input tensor data.
  
  // CUDNN_CONVOLUTION_FWD_ALGO_GEMM
  // This algorithm expresses the convolution as an explicit matrix product. A significant memory workspace is needed to store the matrix that holds the input tensor data.

  std::cout << "Convolution algorithm: " << algo << std::endl;
  std::cout << std::endl;

  // workspace
  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  float *ws_data;
  CUDA_CALL(cudaMalloc(&ws_data, ws_size));

  std::cout << "Workspace size: " << ws_size << std::endl;
  std::cout << std::endl;

  float alpha = 1.f;
  float beta = 0.f;

  dim3 DimGrid(1, 1, 512);
  dim3 DimBlock(16, 8, 1);

  // perform
  dev_const<<<in_n * in_c, in_w * in_h>>>(in_data, 1.f);
  dev_const<<<filt_k * filt_c, filt_w * filt_h>>>(filt_data, 0.f);

  float GPUtime;
  for(int i=0; i<10; i++){
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 
    cudaEventRecord(start, 0); 
    CUDNN_CALL(cudnnConvolutionForward(
        cudnn,
        &alpha, in_desc, in_data, filt_desc, filt_data,
        conv_desc, algo, ws_data, ws_size,
        &beta, out_desc, out_data_ref));
    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&GPUtime, start, stop); 
  }
  // print(out_data_ref, out_n, out_c, out_h, out_w);
  printf("Compute time on GPU with CuDNN: %3.6f ms \n", GPUtime);

  // finalizing
  CUDA_CALL(cudaFree(ws_data));
  CUDA_CALL(cudaFree(out_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDA_CALL(cudaFree(filt_data));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDA_CALL(cudaFree(in_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CALL(cudnnDestroy(cudnn));

  return 0;
}

// compile and run
// nvcc -lcudnn -O3 --gpu-architecture=sm_86 -lineinfo cuDNN_example.cu && ./a.out

// profile tensor core usage
// sudo /usr/local/cuda/bin/nv-nsight-cu-cli --csv --log-file file.csv --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active  ./work/test/a.out
