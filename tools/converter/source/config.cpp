//
//  config.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "config.hpp"
#include <MNN/VCS.h>
#define VERSION_STR(version) #version
const std::string ProjectConfig::version = VERSION_STR(MNN_VERSION);
ProjectConfig *ProjectConfig::m_pConfig  = nullptr;
std::mutex ProjectConfig::m_mutex;

ProjectConfig *ProjectConfig::obtainSingletonInstance() {
    if (m_pConfig == nullptr) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_pConfig == nullptr) {
            m_pConfig = new ProjectConfig();
        }
    }
    return m_pConfig;
}
