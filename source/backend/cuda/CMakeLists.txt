set(CUDA_MIN_VERSION "9.0")
if(MNN_CUDA)
    find_package(CUDA ${CUDA_MIN_VERSION})
endif()

if(CUDA_FOUND)
    include(${CMAKE_CURRENT_SOURCE_DIR}/SelectCudaComputeArch.cmake)
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${CUDA_ARCHS})
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH_FLAGS}")
    # Fix for some combinations of CUDA and GCC (e.g. under Ubuntu 16.04).
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES -Wno-deprecated-gpu-targets -w ")
    message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
                    " archs: ${CUDA_ARCH_FLAGS_readable})")
else()
    message(FATAL_ERROR "CUDA not found >= ${CUDA_MIN_VERSION} required)")
endif()

file(GLOB_RECURSE MNN_CUDA_SRC ${CMAKE_CURRENT_LIST_DIR}/*)
message(STATUS "message ${CUDA_NVCC_FLAGS} !!!!!!!!!!!")

IF(MNN_SEP_BUILD)
    # add_library(MNN_Cuda SHARED ${MNN_CUDA_SRC} )
    cuda_add_library(MNN_Cuda SHARED ${MNN_CUDA_SRC} )
    target_link_libraries(MNN_Cuda MNN ${CUDA_LIBRARIES})
ELSE()
    add_library(MNN_Cuda OBJECT ${MNN_CUDA_SRC} )
    set(MNN_CUDA_LIBS ${CUDA_LIBRARIES} PARENT_SCOPE)
ENDIF()

target_include_directories(MNN_Cuda PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include/
)

