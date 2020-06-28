//
//  Lenet.hpp
//  MNN
//
//  Created by MNN on 2020/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef LenetModels_hpp
#define LenetModels_hpp

#include "Module.hpp"
#include "NN.hpp"

namespace MNN {
namespace Train {
namespace Model {

class MNN_PUBLIC Lenet : public Module {
public:
    Lenet();

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP>& inputs) override;

    std::shared_ptr<Module> conv1;
    std::shared_ptr<Module> conv2;
    std::shared_ptr<Module> ip1;
    std::shared_ptr<Module> ip2;
    std::shared_ptr<Module> dropout;
};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // LenetModels_hpp
