#include "core/Tensor.h"

#include <numeric>
#include <stdexcept>
#include <sstream>

Tensor::Tensor()
{
}

Tensor::Tensor(const Shape& shape)
    : m_shape(shape),
    m_strides(ComputeContiguousStrides(shape)),
    m_data(std::make_shared<std::vector<float>>(ComputeElementCount(shape)))
{
}

Tensor::Tensor(const Shape& shape, float initialValue)
    : m_shape(shape),
    m_strides(ComputeContiguousStrides(shape)),
    m_data(std::make_shared<std::vector<float>>(ComputeElementCount(shape), initialValue))
{
}

Tensor::Tensor(
    const Shape& shape,
    const std::vector<float>& data)
    : m_shape(shape),
    m_strides(ComputeContiguousStrides(shape)),
    m_data(std::make_shared<std::vector<float>>(data))
{
    const size_t expectedSize = ComputeElementCount(shape);

    if (expectedSize != data.size())
    {
        throw std::invalid_argument(
            "Data size does not match shape.");
    }
}

Tensor::Tensor(
    const Shape& shape,
    const Strides& strides,
    const DataPtr& data)
    : m_shape(shape),
    m_strides(strides),
    m_data(data)
{
}

float* Tensor::Data()
{
    if (!m_data)
    {
        return nullptr;
    }
    return m_data->data();
}

const float* Tensor::Data() const
{
    if (!m_data)
    {
        return nullptr;
    }
    return m_data->data();
}

const Tensor::Shape& Tensor::GetShape() const
{
    return m_shape;
}

const Tensor::Strides& Tensor::GetStrides() const
{
    return m_strides;
}

size_t Tensor::Rank() const
{
    return m_shape.size();
}

size_t Tensor::Size() const
{
    return ComputeElementCount(m_shape);
}

bool Tensor::IsUnique() const
{
    return m_data ? m_data.use_count() == 1 : true;
}

long Tensor::UseCount() const
{
    return m_data ? m_data.use_count() : 0;
}

size_t Tensor::RawSize() const
{
    if (!m_data)
    {
        return 0;
    }
    return m_data->size();
}

void Tensor::Reshape(const Shape& newShape)
{
    const size_t newSize = ComputeElementCount(newShape);

    if (newSize != Size())
    {
        throw std::invalid_argument(
            "Reshape element count mismatch.");
    }

    m_shape = newShape;
    m_strides = ComputeContiguousStrides(newShape);
}

void Tensor::Fill(float value)
{
    if (!m_data)
    {
        return;
    }

    for (size_t i = 0; i < Size(); ++i)
    {
        Shape indices = OffsetToIndices(i, m_shape);
        At(indices) = value;
    }
}

void Tensor::Zero()
{
    Fill(0.0f);
}

float& Tensor::operator[](size_t index)
{
    Shape indices = OffsetToIndices(index, m_shape);
    return At(indices);
}

const float& Tensor::operator[](size_t index) const
{
    Shape indices = OffsetToIndices(index, m_shape);
    return At(indices);
}

float& Tensor::At(
    const std::vector<size_t>& indices)
{
    return (*m_data)[ComputeOffset(indices)];
}

const float& Tensor::At(
    const std::vector<size_t>& indices) const
{
    return (*m_data)[ComputeOffset(indices)];
}

Tensor Tensor::Clone() const
{
    if (!m_data)
    {
        return Tensor();
    }

    // Mantıksal sırayla kopyala (view'lar için contiguous yap)
    Tensor result(m_shape);

    for (size_t i = 0; i < Size(); ++i)
    {
        Shape indices = OffsetToIndices(i, m_shape);
        result.At(indices) = At(indices);
    }

    return result;
}

Tensor Tensor::Transpose() const
{
    if (m_shape.size() < 2 || !m_data)
    {
        return *this;
    }

    // Yeni shape: son iki ekseni yer değiştir
    Shape newShape = m_shape;
    std::swap(newShape[newShape.size() - 1], newShape[newShape.size() - 2]);

    // Yeni strides: son iki ekseni yer değiştir
    Strides newStrides = m_strides;
    std::swap(newStrides[newStrides.size() - 1], newStrides[newStrides.size() - 2]);

    // Aynı veriyi paylaşan yeni tensor (view)
    return Tensor(newShape, newStrides, m_data);
}

Tensor Tensor::operator+(const Tensor& other) const
{
    if (m_shape != other.m_shape)
    {
        throw std::invalid_argument(
            "Shape mismatch.");
    }

    Tensor result(m_shape);

    for (size_t i = 0; i < Size(); ++i)
    {
        Shape indices = OffsetToIndices(i, m_shape);
        result.At(indices) = At(indices) + other.At(indices);
    }

    return result;
}

Tensor Tensor::operator-(const Tensor& other) const
{
    if (m_shape != other.m_shape)
    {
        throw std::invalid_argument(
            "Shape mismatch.");
    }

    Tensor result(m_shape);

    for (size_t i = 0; i < Size(); ++i)
    {
        Shape indices = OffsetToIndices(i, m_shape);
        result.At(indices) = At(indices) - other.At(indices);
    }

    return result;
}

Tensor Tensor::operator*(const Tensor& other) const
{
    if (m_shape != other.m_shape)
    {
        throw std::invalid_argument(
            "Shape mismatch.");
    }

    Tensor result(m_shape);

    for (size_t i = 0; i < Size(); ++i)
    {
        Shape indices = OffsetToIndices(i, m_shape);
        result.At(indices) = At(indices) * other.At(indices);
    }

    return result;
}

Tensor Tensor::operator*(float scalar) const
{
    Tensor result(m_shape);

    for (size_t i = 0; i < Size(); ++i)
    {
        Shape indices = OffsetToIndices(i, m_shape);
        result.At(indices) = At(indices) * scalar;
    }

    return result;
}

Tensor Tensor::operator/(float scalar) const
{
    Tensor result(m_shape);

    for (size_t i = 0; i < Size(); ++i)
    {
        Shape indices = OffsetToIndices(i, m_shape);
        result.At(indices) = At(indices) / scalar;
    }

    return result;
}

Tensor& Tensor::operator+=(const Tensor& other)
{
    if (m_shape != other.m_shape)
    {
        throw std::invalid_argument(
            "Shape mismatch.");
    }

    for (size_t i = 0; i < Size(); ++i)
    {
        Shape indices = OffsetToIndices(i, m_shape);
        At(indices) += other.At(indices);
    }

    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other)
{
    if (m_shape != other.m_shape)
    {
        throw std::invalid_argument(
            "Shape mismatch.");
    }

    for (size_t i = 0; i < Size(); ++i)
    {
        Shape indices = OffsetToIndices(i, m_shape);
        At(indices) -= other.At(indices);
    }

    return *this;
}

Tensor& Tensor::operator*=(float scalar)
{
    if (!m_data)
    {
        return *this;
    }

    for (size_t i = 0; i < Size(); ++i)
    {
        Shape indices = OffsetToIndices(i, m_shape);
        At(indices) *= scalar;
    }

    return *this;
}

std::string Tensor::ToString() const
{
    std::stringstream ss;

    ss << "Tensor(shape=[";

    for (size_t i = 0; i < m_shape.size(); ++i)
    {
        ss << m_shape[i];

        if (i + 1 < m_shape.size())
        {
            ss << ", ";
        }
    }

    ss << "], strides=[";
    for (size_t i = 0; i < m_strides.size(); ++i)
    {
        ss << m_strides[i];
        if (i + 1 < m_strides.size())
        {
            ss << ", ";
        }
    }

    ss << "], use_count=" << UseCount() << ")";

    return ss.str();
}

size_t Tensor::ComputeOffset(
    const std::vector<size_t>& indices) const
{
    return ComputeOffsetWithStrides(indices, m_shape, m_strides);
}

size_t Tensor::ComputeOffsetWithStrides(
    const std::vector<size_t>& indices,
    const Shape& shape,
    const Strides& strides)
{
    if (indices.size() != shape.size())
    {
        throw std::invalid_argument(
            "Index rank mismatch.");
    }

    size_t offset = 0;

    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (indices[i] >= shape[i])
        {
            throw std::out_of_range(
                "Index out of range.");
        }

        offset += indices[i] * strides[i];
    }

    return offset;
}

size_t Tensor::ComputeElementCount(
    const Shape& shape)
{
    if (shape.empty())
    {
        return 0;
    }

    return std::accumulate(
        shape.begin(),
        shape.end(),
        static_cast<size_t>(1),
        std::multiplies<size_t>());
}

Tensor::Strides Tensor::ComputeContiguousStrides(
    const Shape& shape)
{
    Strides strides(shape.size());

    if (shape.empty())
    {
        return strides;
    }

    size_t stride = 1;

    for (size_t i = shape.size(); i-- > 0;)
    {
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides;
}

Tensor::Shape Tensor::OffsetToIndices(
    size_t offset,
    const Shape& shape)
{
    Shape indices(shape.size());
    Strides strides = ComputeContiguousStrides(shape);

    size_t remaining = offset;

    for (size_t i = 0; i < shape.size(); ++i)
    {
        indices[i] = remaining / strides[i];
        remaining = remaining % strides[i];
    }

    return indices;
}
