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
    m_data(std::make_shared<std::vector<float>>(ComputeElementCount(shape))),
    m_offset(0)
{
}

Tensor::Tensor(const Shape& shape, float initialValue)
    : m_shape(shape),
    m_strides(ComputeContiguousStrides(shape)),
    m_data(std::make_shared<std::vector<float>>(ComputeElementCount(shape), initialValue)),
	m_offset(0)
{
}

Tensor::Tensor(
    const Shape& shape,
    const std::vector<float>& data)
    : m_shape(shape),
    m_strides(ComputeContiguousStrides(shape)),
    m_data(std::make_shared<std::vector<float>>(data)),
	m_offset(0)
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
    const DataPtr& data,
    size_t offset)
    : m_shape(shape),
    m_strides(strides),
    m_data(data),
    m_offset(offset)
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

size_t Tensor::RawSize() const
{
    if (!m_data)
    {
        return 0;
    }
    return m_data->size();
}

bool Tensor::IsUnique() const
{
    return m_data ? m_data.use_count() == 1 : true;
}

long Tensor::UseCount() const
{
    return m_data ? m_data.use_count() : 0;
}

bool Tensor::IsContiguous() const
{
    return m_strides == ComputeContiguousStrides(m_shape);
}

void Tensor::Reshape(const Shape& newShape)
{
    if (!IsContiguous())
    {
        throw std::runtime_error(
            "Reshape is only valid on contiguous tensors. Use Clone() first.");
    }

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

    if (IsContiguous())
    {
        // Hızlı yol: doğrudan ham veri üzerinde
        for (size_t i = 0; i < Size(); ++i)
        {
            (*m_data)[i] = value;
        }
    }
    else
    {
        // i'yi mantıksal indekslere dönüştürmek için
        // Contiguous strideler kullan (i: 0,1,2,... mantıksal sıra)
        Strides contigStrides = ComputeContiguousStrides(m_shape);

        // View: mantıksal erişim
        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices(m_shape.size());
            size_t remaining = i;

            for (size_t dim = 0; dim < m_shape.size(); ++dim)
            {
                indices[dim] = remaining / contigStrides[dim];
                remaining = remaining % contigStrides[dim];
            }

            At(indices) = value;
        }
    }
}

void Tensor::Zero()
{
    Fill(0.0f);
}

// operator[]: HAM bellek indeksi
float& Tensor::operator[](size_t rawIndex)
{
    return (*m_data)[rawIndex];
}

const float& Tensor::operator[](size_t rawIndex) const
{
    return (*m_data)[rawIndex];
}

// At(): MANTIKSAL indeks (her zaman güvenli)
float& Tensor::At(const Shape& indices)
{
    return (*m_data)[ComputeOffset(indices)];
}

const float& Tensor::At(const Shape& indices) const
{
    return (*m_data)[ComputeOffset(indices)];
}

size_t Tensor::ComputeOffset(const Shape& indices) const
{
    if (indices.size() != m_shape.size())
    {
        throw std::invalid_argument("Index rank mismatch.");
    }

    size_t offset = 0;

    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (indices[i] >= m_shape[i])
        {
            throw std::out_of_range("Index out of range.");
        }
        offset += indices[i] * m_strides[i];
    }

    return offset;
}

Tensor Tensor::Clone() const
{
    if (!m_data)
    {
        return Tensor();
    }

    // Sonuç her zaman Contiguous olur
    Tensor result(m_shape);

    if (IsContiguous())
    {
        // Hızlı yol: doğrudan kopya
        *result.m_data = *m_data;
    }
    else
    {
        // View: mantıksal sırayla kopyala
        Strides contigStrides = ComputeContiguousStrides(m_shape);

        for (size_t i = 0; i < Size(); ++i)
        {
            // i → mantıksal indeksler (contiguous mantıkta)
            Shape indices(m_shape.size());
            size_t remaining = i;

            for (size_t dim = 0; dim < m_shape.size(); ++dim)
            {
                indices[dim] = remaining / contigStrides[dim];
                remaining = remaining % contigStrides[dim];
            }

            result[i] = At(indices);
        }
    }

    return result;
}

Tensor Tensor::Transpose(
    size_t dim1,
    size_t dim2) const
{
    if (!m_data)
    {
        return *this;
    }

    if (dim1 >= m_shape.size() ||
        dim2 >= m_shape.size())
    {
        throw std::out_of_range(
            "Transpose dimension out of range.");
    }

    if (dim1 == dim2)
    {
        return *this;
    }

    Shape newShape = m_shape;
    std::swap(newShape[dim1],
        newShape[dim2]);

    Strides newStrides = m_strides;
    std::swap(newStrides[dim1],
        newStrides[dim2]);

    return Tensor(
        newShape,
        newStrides,
        m_data,
        m_offset);
}

Tensor Tensor::Transpose() const
{
    if (m_shape.size() < 2 || !m_data)
    {
        return *this;
    }

    //Transpose kuralı: Son iki ekseni yer değiştir xT[i][j] = x[j][i]
    Shape newShape = m_shape;
    std::swap(newShape[newShape.size() - 1], newShape[newShape.size() - 2]);

    Strides newStrides = m_strides;
    std::swap(newStrides[newStrides.size() - 1], newStrides[newStrides.size() - 2]);

	// Aynı veriyi paylaşan view oluştur, m_offset değişmez çünkü m_data değişmedi
    return Tensor(newShape, newStrides, m_data, m_offset);
}

Tensor Tensor::Slice(size_t dim, size_t index) const
{
    if (!m_data)
    {
        throw std::runtime_error("Tensor has no data.");
    }

    if (dim >= m_shape.size())
    {
        throw std::out_of_range("Invalid dimension.");
    }

    if (index >= m_shape[dim])
    {
        throw std::out_of_range("Index out of range.");
    }

    Shape newShape = m_shape;
    Strides newStrides = m_strides;

    size_t newOffset =
        m_offset +
        index * m_strides[dim];

    newShape.erase(newShape.begin() + dim);
    newStrides.erase(newStrides.begin() + dim);

    return Tensor(
        newShape,
        newStrides,
        m_data,
        newOffset);
}

Tensor Tensor::Permute(
    const std::vector<size_t>& dims) const
{
    if (dims.size() != m_shape.size())
    {
        throw std::invalid_argument(
            "Permutation rank mismatch.");
    }

    Shape newShape(dims.size());
    Strides newStrides(dims.size());

    std::vector<bool> used(dims.size(), false);

    for (size_t i = 0; i < dims.size(); ++i)
    {
        size_t dim = dims[i];

        if (dim >= m_shape.size())
        {
            throw std::out_of_range(
                "Invalid dimension.");
        }

        if (used[dim])
        {
            throw std::invalid_argument(
                "Duplicate dimension.");
        }

        used[dim] = true;

        newShape[i] = m_shape[dim];
        newStrides[i] = m_strides[dim];
    }

    return Tensor(
        newShape,
        newStrides,
        m_data,
        m_offset);
}

// Hızlı yol kontrolü: İki tensor da hem Contiguous hem aynı shape
static bool AreBothContiguousAndSameShape(const Tensor& a, const Tensor& b)
{
    return a.IsContiguous() && b.IsContiguous() && (a.GetShape() == b.GetShape());
}

Tensor Tensor::operator+(const Tensor& other) const
{
    if (m_shape != other.m_shape)
    {
        throw std::invalid_argument("Shape mismatch.");
    }

    Tensor result(m_shape);

    if (AreBothContiguousAndSameShape(*this, other))
    {
        // HIZLI YOL
        for (size_t i = 0; i < Size(); ++i)
        {
            result[i] = (*this)[i] + other[i];
        }
    }
    else
    {
        // GENEL YOL (view'lar için)
        Strides contigStrides = ComputeContiguousStrides(m_shape);

        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices(m_shape.size());
            size_t remaining = i;

            for (size_t dim = 0; dim < m_shape.size(); ++dim)
            {
                indices[dim] = remaining / contigStrides[dim];
                remaining = remaining % contigStrides[dim];
            }

            result.At(indices) = At(indices) + other.At(indices);
        }
    }

    return result;
}

Tensor Tensor::operator-(const Tensor& other) const
{
    if (m_shape != other.m_shape)
    {
        throw std::invalid_argument("Shape mismatch.");
    }

    Tensor result(m_shape);

    if (AreBothContiguousAndSameShape(*this, other))
    {
        for (size_t i = 0; i < Size(); ++i)
        {
            result[i] = (*this)[i] - other[i];
        }
    }
    else
    {
        Strides contigStrides = ComputeContiguousStrides(m_shape);

        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices(m_shape.size());
            size_t remaining = i;

            for (size_t dim = 0; dim < m_shape.size(); ++dim)
            {
                indices[dim] = remaining / contigStrides[dim];
                remaining = remaining % contigStrides[dim];
            }

            result.At(indices) = At(indices) - other.At(indices);
        }
    }

    return result;
}

Tensor Tensor::operator*(const Tensor& other) const
{
    if (m_shape != other.m_shape)
    {
        throw std::invalid_argument("Shape mismatch.");
    }

    Tensor result(m_shape);

    if (AreBothContiguousAndSameShape(*this, other))
    {
        for (size_t i = 0; i < Size(); ++i)
        {
            result[i] = (*this)[i] * other[i];
        }
    }
    else
    {
        Strides contigStrides = ComputeContiguousStrides(m_shape);

        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices(m_shape.size());
            size_t remaining = i;

            for (size_t dim = 0; dim < m_shape.size(); ++dim)
            {
                indices[dim] = remaining / contigStrides[dim];
                remaining = remaining % contigStrides[dim];
            }

            result.At(indices) = At(indices) * other.At(indices);
        }
    }

    return result;
}

Tensor Tensor::operator*(float scalar) const
{
    Tensor result(m_shape);

    if (IsContiguous())
    {
        for (size_t i = 0; i < Size(); ++i)
        {
            result[i] = (*this)[i] * scalar;
        }
    }
    else
    {
        Strides contigStrides = ComputeContiguousStrides(m_shape);

        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices(m_shape.size());
            size_t remaining = i;

            for (size_t dim = 0; dim < m_shape.size(); ++dim)
            {
                indices[dim] = remaining / contigStrides[dim];
                remaining = remaining % contigStrides[dim];
            }

            result.At(indices) = At(indices) * scalar;
        }
    }

    return result;
}

Tensor Tensor::operator/(float scalar) const
{
    Tensor result(m_shape);

    if (IsContiguous())
    {
        for (size_t i = 0; i < Size(); ++i)
        {
            result[i] = (*this)[i] / scalar;
        }
    }
    else
    {
        Strides contigStrides = ComputeContiguousStrides(m_shape);

        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices(m_shape.size());
            size_t remaining = i;

            for (size_t dim = 0; dim < m_shape.size(); ++dim)
            {
                indices[dim] = remaining / contigStrides[dim];
                remaining = remaining % contigStrides[dim];
            }

            result.At(indices) = At(indices) / scalar;
        }
    }

    return result;
}

Tensor& Tensor::operator+=(const Tensor& other)
{
    if (m_shape != other.m_shape)
    {
        throw std::invalid_argument("Shape mismatch.");
    }

    if (AreBothContiguousAndSameShape(*this, other))
    {
        for (size_t i = 0; i < Size(); ++i)
        {
            (*this)[i] += other[i];
        }
    }
    else
    {
        Strides contigStrides = ComputeContiguousStrides(m_shape);

        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices(m_shape.size());
            size_t remaining = i;

            for (size_t dim = 0; dim < m_shape.size(); ++dim)
            {
                indices[dim] = remaining / contigStrides[dim];
                remaining = remaining % contigStrides[dim];
            }

            At(indices) += other.At(indices);
        }
    }

    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other)
{
    if (m_shape != other.m_shape)
    {
        throw std::invalid_argument("Shape mismatch.");
    }

    if (AreBothContiguousAndSameShape(*this, other))
    {
        for (size_t i = 0; i < Size(); ++i)
        {
            (*this)[i] -= other[i];
        }
    }
    else
    {
        Strides contigStrides = ComputeContiguousStrides(m_shape);

        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices(m_shape.size());
            size_t remaining = i;

            for (size_t dim = 0; dim < m_shape.size(); ++dim)
            {
                indices[dim] = remaining / contigStrides[dim];
                remaining = remaining % contigStrides[dim];
            }

            At(indices) -= other.At(indices);
        }
    }

    return *this;
}

Tensor& Tensor::operator*=(float scalar)
{
    if (IsContiguous())
    {
        for (size_t i = 0; i < Size(); ++i)
        {
            (*this)[i] *= scalar;
        }
    }
    else
    {
        Strides contigStrides = ComputeContiguousStrides(m_shape);

        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices(m_shape.size());
            size_t remaining = i;

            for (size_t dim = 0; dim < m_shape.size(); ++dim)
            {
                indices[dim] = remaining / contigStrides[dim];
                remaining = remaining % contigStrides[dim];
            }

            At(indices) *= scalar;
        }
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
        if (i + 1 < m_shape.size()) ss << ", ";
    }

    ss << "], strides=[";
    for (size_t i = 0; i < m_strides.size(); ++i)
    {
        ss << m_strides[i];
        if (i + 1 < m_strides.size()) ss << ", ";
    }

    ss << "], contiguous=" << (IsContiguous() ? "true" : "false");
    ss << ", use_count=" << UseCount() << ")";

    return ss.str();
}

size_t Tensor::ComputeElementCount(const Shape& shape)
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

Tensor::Strides Tensor::ComputeContiguousStrides(const Shape& shape)
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
