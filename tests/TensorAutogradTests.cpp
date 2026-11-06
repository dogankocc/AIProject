#include "core/Tensor.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace
{
    int failures = 0;

    void ExpectTensor(
        const std::string& testName,
        const Tensor& actual,
        const Tensor::Shape& expectedShape,
        const std::vector<float>& expectedValues,
        float tolerance = 1e-5f)
    {
        if (actual.GetShape() != expectedShape || actual.Size() != expectedValues.size())
        {
            std::cerr << "FAIL " << testName << ": shape/size mismatch\n";
            ++failures;
            return;
        }

        for (size_t i = 0; i < expectedValues.size(); ++i)
        {
            if (std::fabs(actual[i] - expectedValues[i]) > tolerance)
            {
                std::cerr << "FAIL " << testName << ": value " << i
                    << " expected " << expectedValues[i]
                    << ", got " << actual[i] << '\n';
                ++failures;
                return;
            }
        }

        std::cout << "PASS " << testName << '\n';
    }

    void Run(const std::string& testName, const std::function<void()>& test)
    {
        try
        {
            test();
        }
        catch (const std::exception& error)
        {
            std::cerr << "FAIL " << testName << ": unexpected exception: "
                << error.what() << '\n';
            ++failures;
        }
    }

    Tensor MakeTensor(
        const Tensor::Shape& shape,
        const std::vector<float>& values)
    {
        return Tensor(shape, values);
    }
}

int main()
{
    Run("Temporary Pow result survives through SumTensor backward", []
    {
        Tensor input = MakeTensor({ 4 }, { 1, 2, 3, 4 });
        input.SetRequiresGrad(true);

        Tensor loss = input.Pow(2).SumTensor();
        loss.Backward();

        ExpectTensor("Temporary Pow result survives through SumTensor backward",
            input.Grad(), { 4 }, { 2, 4, 6, 8 });
    });

    Run("Sum(dim) propagates gradients", []
    {
        Tensor input = MakeTensor({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
        input.SetRequiresGrad(true);

        Tensor weights = MakeTensor({ 2 }, { 2, 4 });
        Tensor loss = (input.Sum(1) * weights).SumTensor();
        loss.Backward();

        ExpectTensor("Sum(dim) propagates gradients", input.Grad(),
            { 2, 3 }, { 2, 2, 2, 4, 4, 4 });
    });

    Run("Sum(dim, keepDim) propagates gradients", []
    {
        Tensor input = MakeTensor({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
        input.SetRequiresGrad(true);

        Tensor weights = MakeTensor({ 2, 1 }, { 3, 5 });
        Tensor loss = (input.Sum(1, true) * weights).SumTensor();
        loss.Backward();

        ExpectTensor("Sum(dim, keepDim) propagates gradients", input.Grad(),
            { 2, 3 }, { 3, 3, 3, 5, 5, 5 });
    });

    Run("Normalize(dim) gradient matches finite differences", []
    {
        const Tensor::Shape shape = { 2, 3 };
        std::vector<float> values = { 1, 2, 4, 2, 5, 9 };
        const std::vector<float> weightValues = { 1, -2, 0.5f, -1, 3, 2 };
        Tensor weights = MakeTensor(shape, weightValues);

        auto computeLoss = [&](const std::vector<float>& inputValues)
        {
            Tensor sample = MakeTensor(shape, inputValues);
            return (sample.Normalize(1) * weights).Sum();
        };

        Tensor input = MakeTensor(shape, values);
        input.SetRequiresGrad(true);
        Tensor loss = (input.Normalize(1) * weights).SumTensor();
        loss.Backward();

        const float epsilon = 1e-3f;
        std::vector<float> numericalGradient(values.size());
        for (size_t i = 0; i < values.size(); ++i)
        {
            std::vector<float> plusValues = values;
            std::vector<float> minusValues = values;
            plusValues[i] += epsilon;
            minusValues[i] -= epsilon;
            numericalGradient[i] =
                (computeLoss(plusValues) - computeLoss(minusValues)) /
                (2.0f * epsilon);
        }

        ExpectTensor("Normalize(dim) gradient matches finite differences",
            input.Grad(), shape, numericalGradient, 2e-3f);
    });

    Run("Slice scatters gradients to selected region", []
    {
        Tensor input = MakeTensor({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
        input.SetRequiresGrad(true);

        Tensor weights = MakeTensor({ 3 }, { 2, 3, 5 });
        Tensor loss = (input.Slice(0, 1) * weights).SumTensor();
        loss.Backward();

        ExpectTensor("Slice scatters gradients to selected region", input.Grad(),
            { 2, 3 }, { 0, 0, 0, 2, 3, 5 });
    });

    Run("Permute maps gradients back to original axes", []
    {
        Tensor input = MakeTensor({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
        input.SetRequiresGrad(true);

        Tensor weights = MakeTensor({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
        Tensor loss = (input.Permute({ 1, 0 }) * weights).SumTensor();
        loss.Backward();

        ExpectTensor("Permute maps gradients back to original axes", input.Grad(),
            { 2, 3 }, { 1, 3, 5, 2, 4, 6 });
    });

    Run("View preserves gradient order", []
    {
        Tensor input = MakeTensor({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
        input.SetRequiresGrad(true);

        Tensor weights = MakeTensor({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
        Tensor loss = (input.View({ 3, 2 }) * weights).SumTensor();
        loss.Backward();

        ExpectTensor("View preserves gradient order", input.Grad(),
            { 2, 3 }, { 1, 2, 3, 4, 5, 6 });
    });

    Run("Flatten preserves gradient order", []
    {
        Tensor input = MakeTensor({ 2, 2 }, { 1, 2, 3, 4 });
        input.SetRequiresGrad(true);

        Tensor weights = MakeTensor({ 4 }, { 2, 3, 5, 7 });
        Tensor loss = (input.Flatten() * weights).SumTensor();
        loss.Backward();

        ExpectTensor("Flatten preserves gradient order", input.Grad(),
            { 2, 2 }, { 2, 3, 5, 7 });
    });

    Run("Squeeze restores removed singleton axes", []
    {
        Tensor input = MakeTensor({ 1, 2, 1 }, { 1, 2 });
        input.SetRequiresGrad(true);

        Tensor weights = MakeTensor({ 2 }, { 7, 11 });
        Tensor loss = (input.Squeeze() * weights).SumTensor();
        loss.Backward();

        ExpectTensor("Squeeze restores removed singleton axes", input.Grad(),
            { 1, 2, 1 }, { 7, 11 });
    });

    Run("Unsqueeze removes added singleton axis", []
    {
        Tensor input = MakeTensor({ 2 }, { 1, 2 });
        input.SetRequiresGrad(true);

        Tensor weights = MakeTensor({ 1, 2 }, { 7, 11 });
        Tensor loss = (input.Unsqueeze(0) * weights).SumTensor();
        loss.Backward();

        ExpectTensor("Unsqueeze removes added singleton axis", input.Grad(),
            { 2 }, { 7, 11 });
    });

    Run("BroadcastTo reduces repeated gradient contributions", []
    {
        Tensor input = MakeTensor({ 1, 2 }, { 1, 2 });
        input.SetRequiresGrad(true);

        Tensor weights = MakeTensor({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
        Tensor loss = (input.BroadcastTo({ 3, 2 }) * weights).SumTensor();
        loss.Backward();

        ExpectTensor("BroadcastTo reduces repeated gradient contributions", input.Grad(),
            { 1, 2 }, { 9, 12 });
    });

    Run("Expand propagates through BroadcastTo", []
    {
        Tensor input = MakeTensor({ 2 }, { 1, 2 });
        input.SetRequiresGrad(true);

        Tensor weights = MakeTensor({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
        Tensor loss = (input.Expand({ 3, 2 }) * weights).SumTensor();
        loss.Backward();

        ExpectTensor("Expand propagates through BroadcastTo", input.Grad(),
            { 2 }, { 9, 12 });
    });

    Run("Transpose propagates through Permute", []
    {
        Tensor input = MakeTensor({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
        input.SetRequiresGrad(true);

        Tensor weights = MakeTensor({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
        Tensor loss = (input.Transpose() * weights).SumTensor();
        loss.Backward();

        ExpectTensor("Transpose propagates through Permute", input.Grad(),
            { 2, 3 }, { 1, 3, 5, 2, 4, 6 });
    });

    Run("Chained views propagate gradients to the leaf", []
    {
        Tensor input = MakeTensor({ 2, 3 }, { 1, 2, 3, 4, 5, 6 });
        input.SetRequiresGrad(true);

        Tensor weights = MakeTensor({ 3, 2 }, { 1, 2, 3, 4, 5, 6 });
        Tensor transformed = input.Permute({ 1, 0 }).Transpose(0, 1).View({ 3, 2 });
        Tensor loss = (transformed * weights).SumTensor();
        loss.Backward();

        ExpectTensor("Chained views propagate gradients to the leaf", input.Grad(),
            { 2, 3 }, { 1, 2, 3, 4, 5, 6 });
    });

    Run("Backward formulas propagate gradients with recording disabled", []
    {
        {
            Tensor input = MakeTensor({ 2 }, { 2, 3 });
            input.SetRequiresGrad(true);
            input.Pow(3.0f).SumTensor().Backward();
            ExpectTensor("Pow backward gradient", input.Grad(), { 2 }, { 12, 27 });
        }

        {
            Tensor input = MakeTensor({ 2 }, { 1, 2 });
            input.SetRequiresGrad(true);
            input.Log().SumTensor().Backward();
            ExpectTensor("Log backward gradient", input.Grad(), { 2 }, { 1, 0.5f });
        }

        {
            Tensor input = MakeTensor({ 2 }, { 1, 4 });
            input.SetRequiresGrad(true);
            input.Sqrt().SumTensor().Backward();
            ExpectTensor("Sqrt backward gradient", input.Grad(), { 2 }, { 0.5f, 0.25f });
        }

        {
            Tensor input = MakeTensor({ 3 }, { -2, 0, 3 });
            input.SetRequiresGrad(true);
            input.Abs().SumTensor().Backward();
            ExpectTensor("Abs backward gradient", input.Grad(), { 3 }, { -1, 0, 1 });
        }

        {
            Tensor input = MakeTensor({ 3 }, { -1, 0, 2 });
            input.SetRequiresGrad(true);
            input.ReLU().SumTensor().Backward();
            ExpectTensor("ReLU backward gradient", input.Grad(), { 3 }, { 0, 0, 1 });
        }

        {
            Tensor input = MakeTensor({ 2 }, { 0, std::log(3.0f) });
            input.SetRequiresGrad(true);
            input.Sigmoid().SumTensor().Backward();
            ExpectTensor("Sigmoid backward gradient", input.Grad(), { 2 }, { 0.25f, 0.1875f });
        }

        {
            Tensor input = MakeTensor({ 2 }, { 0, 1 });
            input.SetRequiresGrad(true);
            input.Tanh().SumTensor().Backward();
            const float tanhOne = std::tanh(1.0f);
            ExpectTensor("Tanh backward gradient", input.Grad(), { 2 },
                { 1, 1.0f - tanhOne * tanhOne });
        }
    });

    Run("Grad recording is restored after Backward", []
    {
        Tensor input = MakeTensor({ 1 }, { 2 });
        input.SetRequiresGrad(true);

        input.Pow(2.0f).SumTensor().Backward();

        if (!Tensor::IsGradEnabled())
        {
            std::cerr << "FAIL Grad recording is restored after Backward: grad mode stayed disabled\n";
            ++failures;
            return;
        }

        Tensor subsequentOperation = input * 3.0f;
        if (!subsequentOperation.RequiresGrad())
        {
            std::cerr << "FAIL Grad recording is restored after Backward: later operation has no graph\n";
            ++failures;
            return;
        }

        std::cout << "PASS Grad recording is restored after Backward\n";
    });

    Run("Disabled grad mode suppresses graph recording", []
    {
        Tensor input = MakeTensor({ 2 }, { 2, 3 });
        input.SetRequiresGrad(true);

        Tensor::SetGradEnabled(false);
        Tensor result = input.Pow(2.0f);
        Tensor::SetGradEnabled(true);

        if (result.RequiresGrad())
        {
            std::cerr << "FAIL Disabled grad mode suppresses graph recording: result requires grad\n";
            ++failures;
            return;
        }

        ExpectTensor("Disabled grad mode preserves numeric result", result,
            { 2 }, { 4, 9 });
    });

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All autograd tests passed.\n";
    return 0;
}
