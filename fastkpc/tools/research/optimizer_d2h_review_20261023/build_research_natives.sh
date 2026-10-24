#!/bin/sh
# Build research natives in WORK (outside fastkpc/build): relink of the unchanged
# production objects (provenance), and control/patched natives that differ from it
# only in mgcv_multi_penalty_gcv_small64.o compiled from WORK/{control,patched}/src/cuda.
set -e
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
WORK=$1
BUILD="$ROOT/build"
NVCC=/usr/local/cuda/bin/nvcc
CXX=$(R CMD config CXX17)
R_CPPFLAGS=$(R CMD config --cppflags)
RCPP_FLAGS=$(Rscript -e 'cat(Rcpp:::CxxFlags())')
RCPP_ARMADILLO_INCLUDE=$(Rscript -e 'cat(system.file("include", package="RcppArmadillo"))')
RCPP_EIGEN_INCLUDE=$(Rscript -e 'cat(system.file("include", package="RcppEigen"))')
RSPECTRA_INCLUDE=$(Rscript -e 'cat(system.file("include", package="RSpectra"))')
BLAS_LIBS=$(R CMD config BLAS_LIBS)
LAPACK_LIBS=$(R CMD config LAPACK_LIBS)
FLIBS=$(R CMD config FLIBS)
COMMON_INC="$R_CPPFLAGS $RCPP_FLAGS -I$RCPP_ARMADILLO_INCLUDE -I$RCPP_EIGEN_INCLUDE -I$RSPECTRA_INCLUDE -I/usr/local/cuda/include -I$ROOT/src -I$ROOT/src/cuda"
OBJS_BEFORE="dcov_exact_cpu full_cuda_ci_contract full_cuda_ci_semantic_abi full_cuda_ci_native_setup full_cuda_ci_one_call hsic_cpu legacy_dcov_gamma_cpp mgcv_multi_penalty_cpp mgcv_multi_penalty_gcv_capacity r_api_multi_penalty_capacity ci_method fastspline_basis fastspline_solver residual_backend residual_backend_registry residual_cache orientation_matrix orientation_rules regrvonps_native regrvonps_device wanpdag_engine skeleton_engine skeleton_task_scheduler skeleton_engine_cuda cuda_status r_api_cuda dcov_batch_cuda full_cuda_ci_vertical full_cuda_ci_structured_eig full_cuda_ci_method_batch legacy_dcov_spectra_matvec_cuda hsic_batch_cuda fastspline_batched_solver fastspline_residual_cuda mgcv_fixed_sp_stable mgcv_fixed_sp_runtime mgcv_single_penalty_gcv mgcv_multi_penalty_gcv"
OBJS_AFTER="mgcv_multi_penalty_gcv_ext192 mgcv_multi_penalty_gcv_ext384 mgcv_multi_penalty_gcv_ext559 mgcv_extract_fixed_sp_cuda"
link() {  # $1 = small64 object, $2 = output
  set --  "$1" "$2" 
  before=""; for o in $OBJS_BEFORE; do before="$before $BUILD/$o.o"; done
  after=""; for o in $OBJS_AFTER; do after="$after $BUILD/$o.o"; done
  "$CXX" -shared -Wl,-x -o "$2" $before "$1" $after \
    $LAPACK_LIBS $BLAS_LIBS $FLIBS -lcrypto \
    -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver \
    -L"$(R RHOME)/lib" -lR
}
link "$BUILD/mgcv_multi_penalty_gcv_small64.o" "$WORK/relink/fastkpc_cuda.so"
for v in control patched; do
  "$NVCC" -O3 -arch=sm_89 --fmad=false -diag-suppress 2464 \
    -Xcompiler -fPIC -std=c++17 $COMMON_INC \
    -c "$WORK/$v/src/cuda/mgcv_multi_penalty_gcv_small64.cu" \
    -o "$WORK/$v/mgcv_multi_penalty_gcv_small64.o" &
done
wait
for v in control patched; do link "$WORK/$v/mgcv_multi_penalty_gcv_small64.o" "$WORK/$v/fastkpc_cuda.so"; done
sha256sum "$BUILD/fastkpc_cuda.so" "$WORK"/*/fastkpc_cuda.so "$BUILD/mgcv_multi_penalty_gcv_small64.o" "$WORK"/*/mgcv_multi_penalty_gcv_small64.o
echo BUILD_DONE
