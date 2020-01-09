::#!/usr/bin/env bash
cd src/cuda

::# compile dcn
"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0/bin/nvcc.exe" -c -o dcn_v2_im2col_cuda.cu.o dcn_v2_im2col_cuda.cu -x cu -Xcompiler -fPIC
"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0/bin/nvcc.exe" -c -o dcn_v2_im2col_cuda_double.cu.o dcn_v2_im2col_cuda_double.cu -x cu -Xcompiler -fPIC

::# compile dcn-roi-pooling
"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0/bin/nvcc.exe" -c -o dcn_v2_psroi_pooling_cuda.cu.o dcn_v2_psroi_pooling_cuda.cu -x cu -Xcompiler -fPIC
"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0/bin/nvcc.exe" -c -o dcn_v2_psroi_pooling_cuda_double.cu.o dcn_v2_psroi_pooling_cuda_double.cu -x cu -Xcompiler -fPIC

cd -
python build.py
python build_double.py
