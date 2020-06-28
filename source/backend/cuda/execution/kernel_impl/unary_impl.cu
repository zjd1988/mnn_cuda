/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuda_runtime.h>
#include "backend/cuda/execution/kernel_impl/unary_impl.cuh"
template <typename T>
__global__ void ExponentialKernel(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = exp(input[i]);
  }
  return;
}
template <>
__global__ void ExponentialKernel(half *input, half *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hexp(input[i]);
  }
  return;
}
template <typename T>
__global__ void LogarithmKernel(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = logf(input[i]);
  }
  return;
}
template <typename T>
__global__ void NegativeKernel(T *input, T *output, size_t count) {
  T neg_one = -1;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = neg_one * input[i];
  }
  return;
}
template <typename T>
__global__ void ReciprocalKernel(T *input, T *output, size_t count) {
  T one = 1.0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = one / input[i];
  }
  return;
}
template <typename T>
__global__ void SquareKernel(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = input[i] * input[i];
  }
  return;
}
template <typename T>
__global__ void SqrtKernel(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = sqrt(input[i]);
  }
  return;
}
template <>
__global__ void SqrtKernel(half *input, half *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hsqrt(input[i]);
  }
  return;
}
template <typename T>
__global__ void RsqrtKernel(T *input, T *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = rsqrt(input[i]);
  }
  return;
}
template <>
__global__ void RsqrtKernel(half *input, half *output, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = hrsqrt(input[i]);
  }
  return;
}
template <typename T>
__global__ void ZeroslikeKernel(T *output, size_t count) {
  T zero = 0.0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (count); i += blockDim.x * gridDim.x) {
    output[i] = zero;
  }
  return;
}

void Exponential(void *input, void *output, size_t count, MNN::CUDARuntime* runtime, MNNCUDADataType_t data_type)
{
  cudaStream_t cuda_stream = runtime->stream();
  int block_num = runtime->blocks_num(count);
  int threads_num = runtime->threads_num();
  if(data_type == CUDA_FLOAT32)
    ExponentialKernel<<<block_num, threads_num, 0, cuda_stream>>>((float*)input, (float*)output, count);
  else
    ExponentialKernel<<<block_num, threads_num, 0, cuda_stream>>>((half*)input, (half*)output, count);
}
void Logarithm(void *input, void *output, size_t count, MNN::CUDARuntime* runtime, MNNCUDADataType_t data_type)
{
  cudaStream_t cuda_stream = runtime->stream();
  int block_num = runtime->blocks_num(count);
  int threads_num = runtime->threads_num();
  if(data_type == CUDA_FLOAT32)
    LogarithmKernel<<<block_num, threads_num, 0, cuda_stream>>>((float*)input, (float*)output, count);
  else
    LogarithmKernel<<<block_num, threads_num, 0, cuda_stream>>>((half*)input, (half*)output, count);  
}
void Negative(void *input, void *output, size_t count, MNN::CUDARuntime* runtime, MNNCUDADataType_t data_type)
{
  cudaStream_t cuda_stream = runtime->stream();
  int block_num = runtime->blocks_num(count);
  int threads_num = runtime->threads_num();
  if(data_type == CUDA_FLOAT32)
    NegativeKernel<<<block_num, threads_num, 0, cuda_stream>>>((float*)input, (float*)output, count);
  else
    NegativeKernel<<<block_num, threads_num, 0, cuda_stream>>>((half*)input, (half*)output, count);    
}
void Reciprocal(void *input, void *output, size_t count, MNN::CUDARuntime* runtime, MNNCUDADataType_t data_type)
{
  cudaStream_t cuda_stream = runtime->stream();
  int block_num = runtime->blocks_num(count);
  int threads_num = runtime->threads_num();
  if(data_type == CUDA_FLOAT32)
    ReciprocalKernel<<<block_num, threads_num, 0, cuda_stream>>>((float*)input, (float*)output, count);
  else
    ReciprocalKernel<<<block_num, threads_num, 0, cuda_stream>>>((half*)input, (half*)output, count);      
}
void Square(void *input, void *output, size_t count, MNN::CUDARuntime* runtime, MNNCUDADataType_t data_type)
{
  cudaStream_t cuda_stream = runtime->stream();
  int block_num = runtime->blocks_num(count);
  int threads_num = runtime->threads_num();
  if(data_type == CUDA_FLOAT32)
    SquareKernel<<<block_num, threads_num, 0, cuda_stream>>>((float*)input, (float*)output, count);
  else
    SquareKernel<<<block_num, threads_num, 0, cuda_stream>>>((half*)input, (half*)output, count);   
}
void Sqrt(void *input, void *output, size_t count, MNN::CUDARuntime* runtime, MNNCUDADataType_t data_type)
{
  cudaStream_t cuda_stream = runtime->stream();
  int block_num = runtime->blocks_num(count);
  int threads_num = runtime->threads_num();
  if(data_type == CUDA_FLOAT32)
    SqrtKernel<<<block_num, threads_num, 0, cuda_stream>>>((float*)input, (float*)output, count);
  else
    SqrtKernel<<<block_num, threads_num, 0, cuda_stream>>>((half*)input, (half*)output, count);    
}
void Rsqrt(void *input, void *output, size_t count, MNN::CUDARuntime* runtime, MNNCUDADataType_t data_type)
{
  cudaStream_t cuda_stream = runtime->stream();
  int block_num = runtime->blocks_num(count);
  int threads_num = runtime->threads_num();
  if(data_type == CUDA_FLOAT32)
    RsqrtKernel<<<block_num, threads_num, 0, cuda_stream>>>((float*)input, (float*)output, count);
  else
    RsqrtKernel<<<block_num, threads_num, 0, cuda_stream>>>((half*)input, (half*)output, count);
}
void Zeroslike(void *output, size_t count, MNN::CUDARuntime* runtime, MNNCUDADataType_t data_type)
{
  cudaStream_t cuda_stream = runtime->stream();
  int block_num = runtime->blocks_num(count);
  int threads_num = runtime->threads_num();
  if(data_type == CUDA_FLOAT32)
    ZeroslikeKernel<<<block_num, threads_num, 0, cuda_stream>>>((float*)output, count);
  else
    ZeroslikeKernel<<<block_num, threads_num, 0, cuda_stream>>>((half*)output, count);
}


void callUnary(void *input, void *output, size_t count, MNN::CUDARuntime* runtime, MNNCUDADataType_t data_type,
   CudaUnaryOpType op_type)
{
  switch(op_type) {
    case CudaUnaryOpOperation_ABS:
      break;
    case CudaUnaryOpOperation_NEG:
      break;
    case CudaUnaryOpOperation_FLOOR:
      break;
    case CudaUnaryOpOperation_CEIL:
      break;    
    case CudaUnaryOpOperation_SQUARE:
      break;    
    case CudaUnaryOpOperation_SQRT:
      Sqrt(input, output, count, runtime, data_type);
      break;
    case CudaUnaryOpOperation_RSQRT:
      break;
    case CudaUnaryOpOperation_EXP:
      break;
    case CudaUnaryOpOperation_LOG:
      break;
    case CudaUnaryOpOperation_SIN:
      break;
    case CudaUnaryOpOperation_COS:
    case CudaUnaryOpOperation_TAN:
    case CudaUnaryOpOperation_ASIN:
    case CudaUnaryOpOperation_ACOS:
    case CudaUnaryOpOperation_ATAN:
    case CudaUnaryOpOperation_RECIPROCAL:
    case CudaUnaryOpOperation_LOG1P:
    case CudaUnaryOpOperation_BNLL:
    case CudaUnaryOpOperation_ACOSH:
    case CudaUnaryOpOperation_SINH:
    case CudaUnaryOpOperation_ASINH:
    case CudaUnaryOpOperation_ATANH:
    case CudaUnaryOpOperation_SIGN:
    case CudaUnaryOpOperation_ROUND:
    case CudaUnaryOpOperation_COSH:
    case CudaUnaryOpOperation_ERF:
    case CudaUnaryOpOperation_ERFC:
    case CudaUnaryOpOperation_ERFINV:
    case CudaUnaryOpOperation_EXPM1:
    case CudaUnaryOpOperation_SIGMOID:
    case CudaUnaryOpOperation_TANH:
    default:
      break;
  }
  return;
}