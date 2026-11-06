#pragma once

#include <vector>
#include <functional>
#include <string>
#include <memory>

class Tensor;

namespace AIProject::Autograd
{
    struct AutogradNode
    {
        // Own parent snapshots so temporary Tensor values survive backward.
        std::vector<std::shared_ptr<Tensor>> parents;

        std::function<void(Tensor&)> backward;

        std::string operationName;
    };
}