//
//  Module.cpp
//  MNN
//
//  Created by MNN on 2019/11/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Module.hpp"
#include <MNN/expr/ExprCreator.hpp>
#include "FixModule.hpp"
using namespace MNN::Express;
namespace MNN {
namespace Train {

Express::VARP Module::forward(Express::VARP input) {
    return this->onForward({input})[0];
}
std::vector<Express::VARP> Module::parameters() const {
    std::vector<Express::VARP> result;
    _collectParameters(result);
    return result;
}
bool Module::loadParameters(const std::vector<Express::VARP>& parameters) {
    std::vector<Express::VARP> result;
    _collectParameters(result);
    if (parameters.empty() || parameters.size() != result.size()) {
        MNN_ERROR("Error parameters, empty or parameter size not match \n");
        return false;
    }
    for (int i=0; i<parameters.size(); ++i) {
        auto dstInfo = result[i]->getInfo();
        auto srcInfo = parameters[i]->getInfo();
        if (dstInfo->dim.size() != srcInfo->dim.size() || dstInfo->order != srcInfo->order) {
            MNN_ERROR("Error parameters %d, dim size or order not match \n", i);
            return false;
        }
        if (dstInfo->size != srcInfo->size || dstInfo->type != srcInfo->type) {
            MNN_ERROR("Error parameters %d, size or type not match \n", i);
            return false;
        }
        Variable::replace(result[i], parameters[i]);
    }
    return true;
}
void Module::setIsTraining(const bool isTraining) {
    mIsTraining = isTraining;
    for (auto c : mChildren) {
        c->setIsTraining(isTraining);
    }
}

bool Module::getIsTraining() {
    return mIsTraining;
}

void Module::registerModel(const std::vector<std::shared_ptr<Module>>& children) {
    mChildren.insert(mChildren.begin(), children.begin(), children.end());
}
void Module::addParameter(VARP parameter) {
    mParameters.emplace_back(parameter);
}
void Module::_collectParameters(std::vector<Express::VARP>& result) const {
    for (auto p : mParameters) {
        result.push_back(p);
    }
    for (auto c : mChildren) {
        c->_collectParameters(result);
    }
}
std::shared_ptr<Module> Module::transform(const std::vector<Express::VARP>& inputs,
                                          const std::vector<Express::VARP>& outputs) {
    std::vector<std::pair<VARP, Express::Dimensionformat>> inputsPair;
    for (auto i : inputs) {
        auto info = i->getInfo();
        if (nullptr == info) {
            MNN_ERROR("Error to load inputs info for module\n");
            return nullptr;
        }
        inputsPair.emplace_back(std::make_pair(i, info->order));
    }

    // Load Parameters
    auto order = Variable::getExecuteOrder(outputs);
    std::vector<VARP> parameters;
    for (auto v : order) {
        if (v->get() != nullptr) {
            continue;
        }
        auto type = v->inputType();
        if (VARP::TRAINABLE == type) {
            parameters.emplace_back(Variable::create(v, 0));
        }
    }

    // FIXME: Find better way to tread NC4HW4 outputs
    std::vector<VARP> newOutputs = outputs;
    for (auto& v : newOutputs) {
        if (v->getInfo() != nullptr) {
            if (v->getInfo()->order == NC4HW4) {
                v = _Convert(v, NCHW);
            }
        }
    }
    std::shared_ptr<Module> m(new FixModule(newOutputs, parameters, inputsPair));
    return m;
}
void Module::clearCache() {
    for (auto c : mChildren) {
        c->clearCache();
    }
    this->onClearCache();
}

} // namespace Train
} // namespace MNN
