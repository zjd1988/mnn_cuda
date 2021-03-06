//
//  MNNDefine.h
//  MNN
//
//  Created by MNN on 2018/08/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNNDefine_h
#define MNNDefine_h

#include <assert.h>
#include <stdio.h>

#if defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_OS_IPHONE
#define MNN_BUILD_FOR_IOS
#endif
#endif

#ifdef MNN_USE_LOGCAT
#include <android/log.h>
#define MNN_ERROR(format, ...) __android_log_print(ANDROID_LOG_ERROR, "MNNJNI", format, ##__VA_ARGS__)
#define MNN_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MNNJNI", format, ##__VA_ARGS__)
#else
#define MNN_PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define MNN_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#endif

#ifdef DEBUG
#define MNN_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            MNN_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
            assert(res);                                         \
        }                                                        \
    }

#define MNN_THROW(msg)                                           \
    {                                                            \
        MNN_PRINT(msg);                                          \
        MNN_ASSERT(false);                                       \
    }    
#else
#define MNN_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            MNN_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
        }                                                        \
    }

#define MNN_THROW(msg)                                           \
    {                                                            \
        MNN_PRINT(msg);                                          \
        MNN_ASSERT(false);                                       \
    }
#endif

#define FUNC_PRINT(x) MNN_PRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define FUNC_PRINT_ALL(x, type) MNN_PRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#if defined(_MSC_VER)
#ifdef BUILDING_DLL
#define MNN_PUBLIC __declspec(dllexport)
#else
#define MNN_PUBLIC __declspec(dllimport)
#endif
#else
#define MNN_PUBLIC __attribute__((visibility("default")))
#endif

#endif /* MNNDefine_h */
