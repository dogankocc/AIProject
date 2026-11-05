#pragma once

#include "core/Tensor.h"

#include <cmath>
#include <stdexcept>

class Embedding
{
public:
    Embedding(size_t vocabSize, size_t embeddingDim)
        : m_weights(CreateWeights(vocabSize, embeddingDim))
    {
        m_weights.SetRequiresGrad(true);
    }

    // tokenIds can have any shape. The output appends embeddingDim to that shape.
    Tensor Forward(const Tensor& tokenIds) const
    {
        Tensor contiguousTokenIds = tokenIds.IsContiguous()
            ? tokenIds
            : tokenIds.Clone();
        Tensor flatTokenIds = contiguousTokenIds.Flatten();

        for (size_t i = 0; i < flatTokenIds.Size(); ++i)
        {
            const float tokenId = flatTokenIds[i];
            if (!std::isfinite(tokenId) ||
                tokenId < 0.0f ||
                std::floor(tokenId) != tokenId ||
                tokenId >= static_cast<float>(VocabSize()))
            {
                throw std::out_of_range(
                    "Token IDs must be integers within the vocabulary range.");
            }
        }

        Tensor selected = m_weights.IndexSelect(0, flatTokenIds);

        Tensor::Shape outputShape = tokenIds.GetShape();
        outputShape.push_back(EmbeddingDim());
        return selected.View(outputShape);
    }

    Tensor operator()(const Tensor& tokenIds) const
    {
        return Forward(tokenIds);
    }

    Tensor& Weights()
    {
        return m_weights;
    }

    const Tensor& Weights() const
    {
        return m_weights;
    }

    size_t VocabSize() const
    {
        return m_weights.GetShape()[0];
    }

    size_t EmbeddingDim() const
    {
        return m_weights.GetShape()[1];
    }

private:
    static Tensor CreateWeights(size_t vocabSize, size_t embeddingDim)
    {
        if (vocabSize == 0 || embeddingDim == 0)
        {
            throw std::invalid_argument(
                "Embedding dimensions must be greater than zero.");
        }

        return Tensor::Randn({ vocabSize, embeddingDim });
    }

    Tensor m_weights;
};
