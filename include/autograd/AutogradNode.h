#pragma once

#include <vector>
#include <functional>
#include <string>

class Tensor;

namespace Tinynn::Autograd
{
    struct AutogradNode
    {
        std::vector<Tensor*> parents;

        std::function<void(Tensor&)> backward;

        std::string operationName;
    };
}