//
//  BackendRegister.cpp
//  MNN
//
//  Created by MNN on 2019/05/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <mutex>

namespace MNN {
extern void registerCPUBackendCreator();

#ifdef ENABLE_ARMV82
#if defined(__aarch64__) && defined(__APPLE__)
extern void registerArm82BackendCreator();
#endif
#endif


#ifdef MNN_CODEGEN_REGISTER
extern void registerMetalBackendCreator();
#endif
void registerBackend() {
    static std::once_flag s_flag;
    std::call_once(s_flag, [&]() {
        registerCPUBackendCreator();

#ifdef ENABLE_ARMV82        
#if defined(__aarch64__) && defined(__APPLE__)
        registerArm82BackendCreator();
#endif
#endif

#ifdef MNN_CODEGEN_REGISTER
        registerMetalBackendCreator();
#endif
    });
}
} // namespace MNN
