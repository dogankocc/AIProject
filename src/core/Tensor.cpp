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
    return m_data->data() + m_offset;
}

const float* Tensor::Data() const
{
    if (!m_data)
    {
        return nullptr;
    }
    return m_data->data() + m_offset;
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

size_t Tensor::StorageSize() const
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
        throw std::runtime_error("Reshape is only valid on contiguous tensors. Use Clone() first.");
    }

    const size_t newSize = ComputeElementCount(newShape);

    if (newSize != Size())
    {
        throw std::invalid_argument("Reshape element count mismatch.");
    }

    m_shape = newShape;
    m_strides = ComputeContiguousStrides(newShape);
}


Tensor::Shape Tensor::LinearToIndices(size_t linearIndex) const
{
    Shape indices(m_shape.size());

    // i'yi mantıksal indekslere dönüştürmek için
    // Contiguous strideler kullan (i: 0,1,2,... mantıksal sıra)
    // View: mantıksal sırayla kopyala
    auto contigStrides = ComputeContiguousStrides(m_shape);

    size_t remaining = linearIndex;

    //Mantıksal görünüm shape üzerinde i. index de hangi eleman var onu buluyoruz 
    //i = "Tensor'un kaçıncı elemanı?" indices = "Bu eleman shape içinde hangi koordinatta?"
    //Örneğin: Shape = {2,3} için Stride = { 3,1 } olur. i=4 için indices = {1,1} olur. Yani 2. satır 2. sütun.
    /* Mantıksal görünüm:(2 satır 3 sütun)
        i=0 -> {0,0}
        i=1 -> {0,1}
        i=2 -> {0,2}
        i=3 -> {1,0}
        i=4 -> {1,1}
        i=5 -> {1,2}

        Linear Index
              ↓
        Multi Dimensional Index
    */
    for (size_t dim = 0; dim < m_shape.size(); ++dim)
    {
        indices[dim] = remaining / contigStrides[dim];//Bu eksende kaç adım ilerlediğimizi buluyor.

        remaining %= contigStrides[dim];//Bu eksenin hesabını yaptıktan sonra alt eksenlere kalan kısmı bırakıyor.
    }

    return indices;
}

Tensor::Shape Tensor::LinearToIndices(size_t linearIndex, const Shape& shape) const
{
    Shape indices(shape.size());

    Strides contigStrides = ComputeContiguousStrides(shape);

    size_t remaining = linearIndex;

    for (size_t dim = 0; dim < shape.size(); ++dim)
    {
        indices[dim] = remaining / contigStrides[dim];

        remaining %= contigStrides[dim];
    }

    return indices;
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
            (*m_data)[m_offset + i] = value;
        }
    }
    else
    {
        for (size_t i = 0; i < Size(); ++i)
        {
			Shape indices = LinearToIndices(i);

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
    if (!IsContiguous())
    {
        throw std::runtime_error(
            "operator[] only valid on contiguous tensors.");
    }
    if (rawIndex >= Size())
    {
        throw std::out_of_range(
            "Raw index out of range.");
    }

    return (*m_data)[m_offset + rawIndex];
}

const float& Tensor::operator[](size_t rawIndex) const
{
    if (!IsContiguous())
    {
        throw std::runtime_error(
            "operator[] only valid on contiguous tensors.");
    }
    if (rawIndex >= Size())
    {
        throw std::out_of_range(
            "Raw index out of range.");
    }

    return (*m_data)[m_offset + rawIndex];
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

    size_t offset = m_offset;

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

Tensor Tensor::Clone(CloneMode mode) const
{
    if (!m_data)
    {
        return Tensor();
    }

    if (mode == CloneMode::Storage)
    {
        return Tensor(
            m_shape,
            m_strides,
            std::make_shared<std::vector<float>>(*m_data),
            m_offset);
    }

    if (mode == CloneMode::Logical &&
        IsContiguous() &&
        m_offset == 0)
    {
        Tensor result(m_shape);

        std::copy(
            m_data->begin(),
            m_data->begin() + Size(),
            result.m_data->begin());

        return result;
    }

    // CloneMode::Logical

    Tensor result(m_shape);

    for (size_t i = 0; i < Size(); ++i)
    {
        result[i] = At(LinearToIndices(i));
    }

    return result;
}

Tensor Tensor::Transpose(size_t dim1, size_t dim2) const
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

Tensor Tensor::Permute(const std::vector<size_t>& dims) const
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

Tensor Tensor::View(const Shape& shape) const
{
    if (!IsContiguous())
    {
        throw std::runtime_error(
            "View is only valid on contiguous tensors.");
    }

    if (ComputeElementCount(shape) != Size())
    {
        throw std::invalid_argument(
            "Element count mismatch.");
    }

    return Tensor(
        shape,
        ComputeContiguousStrides(shape),
        m_data,
        m_offset);
}

Tensor Tensor::Flatten() const
{
    return View({ Size() });
}

Tensor Tensor::Squeeze() const
{
    Shape newShape;
    Strides newStrides;

    for (size_t i = 0; i < m_shape.size(); ++i)
    {
        if (m_shape[i] != 1)
        {
            newShape.push_back(m_shape[i]);
            newStrides.push_back(m_strides[i]);
        }
    }

    return Tensor(
        newShape,
        newStrides,
        m_data,
        m_offset);
}

Tensor Tensor::Unsqueeze(size_t dim) const
{
    if (dim > m_shape.size())
    {
        throw std::out_of_range(
            "Invalid dimension.");
    }

    Shape newShape = m_shape;
    Strides newStrides = m_strides;

    newShape.insert(
        newShape.begin() + dim,
        1);

    size_t newStride =
        (dim < m_strides.size())
        ? m_strides[dim]
        : 1;

    newStrides.insert(
        newStrides.begin() + dim,
        newStride);

    return Tensor(
        newShape,
        newStrides,
        m_data,
        m_offset);
}

Tensor Tensor::BroadcastTo(const Shape& targetShape) const
{
    // Hedef tensorun rank'ı kaynak tensorunkinden küçük olamaz.
    // Örn:{3,4} -> {4}
    if (targetShape.size() < m_shape.size())
    {
        throw std::invalid_argument(
            "Target rank is smaller than source rank.");
    }

    Shape newShape = targetShape;

    // Broadcast sonrası kullanılacak stride'lar
    Strides newStrides(targetShape.size());

    // Hedef shape daha büyükse, kaynak shape sağa hizalanır.
    // Örn:
    // source = {4}
    // target = {3,4}
    //
    // source aslında:
    // {1,4}
    // gibi düşünülür.
    size_t shapeOffset = targetShape.size() - m_shape.size();

    for (size_t i = 0; i < targetShape.size(); ++i)
    {
        // Kaynak shape'de olmayan, sola eklenmiş boyutlar
        // Örn:
        // source = {4}
        // target = {3,4}
        // yeni eklenen ilk boyutun stride'ı 0 olur.
        if (i < shapeOffset)
        {
            // Eklenen boyutlar
            newStrides[i] = 0;
            continue;
        }

        size_t srcDim = i - shapeOffset;

        size_t srcSize = m_shape[srcDim];
        size_t dstSize = targetShape[i];

        // Boyutlar eşitse stride aynen korunur.
        //
        // Örn:
        // source = {3,4}
        // target = {3,4}
        if (srcSize == dstSize)
        {
            newStrides[i] = m_strides[srcDim];
        }
        // Kaynak boyut 1 ise broadcast yapılabilir.
        // Örn:
        // source = {1,4}
        // target = {3,4}
        //
        // İlk boyuttaki stride 0 yapılır.
        // Böylece aynı değer satırlar boyunca tekrar edilmiş gibi görünür.
        else if (srcSize == 1)
        {
            newStrides[i] = 0;
        }
        // Ne eşit ne de kaynak boyut 1 ise
        // broadcasting mümkün değildir.
        //
        // Örn:
        // {2,4} -> {3,4}
        else
        {
            throw std::invalid_argument(
                "Shapes are not broadcast compatible.");
        }
    }
    // Veri kopyalama yok.
    // Aynı data paylaşılır.
    //
    // Değişenler:
    //   shape
    //   stride
    //
    // Değişmeyenler:
    //   data
    //   offset
    return Tensor(
        newShape,
        newStrides,
        m_data,
        m_offset);
}

Tensor Tensor::Expand(const Shape& shape) const
{
    return BroadcastTo(shape);
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

        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices = LinearToIndices(i);
            
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
        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices = LinearToIndices(i);

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
        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices = LinearToIndices(i);

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
        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices = LinearToIndices(i);

            result.At(indices) = At(indices) * scalar;
        }
    }

    return result;
}

Tensor Tensor::operator/(float scalar) const
{
    if (scalar == 0.0f)
    {
        throw std::runtime_error(
            "Division by zero.");
    }
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
        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices = LinearToIndices(i);

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
        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices = LinearToIndices(i);

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
        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices = LinearToIndices(i);

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
        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices = LinearToIndices(i);

            At(indices) *= scalar;
        }
    }

    return *this;
}

Tensor& Tensor::operator/=(float scalar)
{
    if (scalar == 0.0f)
    {
        throw std::runtime_error("Division by zero.");
    }
    if (IsContiguous())
    {
        for (size_t i = 0; i < Size(); ++i)
        {
            (*this)[i] /= scalar;
        }
    }
    else
    {
        for (size_t i = 0; i < Size(); ++i)
        {

            //Mantıksal görünüm shape üzerinde i. index de hangi eleman var onu buluyoruz 
            //i = "Tensor'un kaçıncı elemanı?" indices = "Bu eleman shape içinde hangi koordinatta?"
			//Örneğin: Shape = {2,3} için Stride = { 3,1 } olur. i=4 için indices = {1,1} olur. Yani 2. satır 2. sütun.
            /* Mantıksal görünüm:(2 satır 3 sütun)
                i=0 -> {0,0}
                i=1 -> {0,1}
                i=2 -> {0,2}
                i=3 -> {1,0}
                i=4 -> {1,1}
                i=5 -> {1,2}

                Linear Index
                      ↓
                Multi Dimensional Index
            */
            Shape indices = LinearToIndices(i);

            At(indices) /= scalar;
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

float Tensor::Sum() const
{
    float result = 0.0f;

    if (!m_data)
    {
        return result;
    }

    if (IsContiguous())
    {
        for (size_t i = 0; i < Size(); ++i)
        {
            result += (*this)[i];
        }
    }
    else
    {
        for (size_t i = 0; i < Size(); ++i)
        {
            Shape indices = LinearToIndices(i);

            result += At(indices);
        }
    }

    return result;
}

Tensor Tensor::Sum(size_t dim) const
{
    if (dim >= Rank())
    {
        throw std::out_of_range("Invalid dimension.");
    }

    Shape resultShape = m_shape;

    resultShape.erase(resultShape.begin() + dim);

    Tensor result(resultShape, 0.0f);

    for (size_t i = 0; i < result.Size(); ++i)
    {
        Shape resultIndices = LinearToIndices(i, resultShape);

        float sum = 0.0f;

        for (size_t k = 0; k < m_shape[dim]; ++k)
        {
            Shape srcIndices = resultIndices;

            srcIndices.insert(srcIndices.begin() + dim, k);

            sum += At(srcIndices);
        }

        result.At(resultIndices) = sum;
    }

    return result;
}

float Tensor::Mean() const
{
    if (Size() == 0)
    {
        return 0.0f;
    }

    return Sum() / static_cast<float>(Size());
}

Tensor Tensor::Mean(size_t dim) const
{
    Tensor result = Sum(dim);

    result /= static_cast<float>(m_shape[dim]);

    return result;
}

float Tensor::Max() const
{
    if (!m_data || Size() == 0)
    {
        throw std::runtime_error(
            "Tensor is empty.");
    }

    float result = At(Shape(Rank(), 0));


    for (size_t i = 1; i < Size(); ++i)
    {
		Shape indices = LinearToIndices(i);

        result = std::max(result, At(indices));
    }

    return result;
}

Tensor Tensor::Max(size_t dim) const
{
    if (dim >= Rank())
    {
        throw std::out_of_range("Invalid dimension.");
    }

    Shape resultShape = m_shape;

    resultShape.erase(resultShape.begin() + dim);

    Tensor result(resultShape);

    for (size_t i = 0; i < result.Size(); ++i)
    {
        Shape resultIndices = LinearToIndices(i, resultShape);

        Shape srcIndices = resultIndices;

        srcIndices.insert(srcIndices.begin() + dim, 0);

        float maxValue = At(srcIndices);

        for (size_t k = 1; k < m_shape[dim]; ++k)
        {
            srcIndices[dim] = k;

            maxValue = std::max(maxValue, At(srcIndices));
        }

        result.At(resultIndices) = maxValue;
    }

    return result;
}

float Tensor::Min() const
{
    if (!m_data || Size() == 0)
    {
        throw std::runtime_error("Tensor is empty.");
    }

    float result = At(Shape(Rank(), 0));

    for (size_t i = 1; i < Size(); ++i)
    {
        Shape indices = LinearToIndices(i);

        result = std::min(result, At(indices));
    }

    return result;
}

Tensor Tensor::Min(size_t dim) const
{
    if (dim >= Rank())
    {
        throw std::out_of_range("Invalid dimension.");
    }

    Shape resultShape = m_shape;

    resultShape.erase(resultShape.begin() + dim);

    Tensor result(resultShape);

    for (size_t i = 0; i < result.Size(); ++i)
    {
        Shape resultIndices = LinearToIndices(i, resultShape);

        Shape srcIndices = resultIndices;

        srcIndices.insert(srcIndices.begin() + dim, 0);

        float minValue = At(srcIndices);

        for (size_t k = 1;
            k < m_shape[dim];
            ++k)
        {
            srcIndices[dim] = k;

            minValue = std::min(minValue, At(srcIndices));
        }

        result.At(resultIndices) = minValue;
    }

    return result;
}

size_t Tensor::ArgMax() const
{
    if (!m_data || Size() == 0)
    {
        throw std::runtime_error(
            "Tensor is empty.");
    }

    float maxValue = At(Shape(Rank(), 0));
    size_t maxIndex = 0;

    for (size_t i = 1; i < Size(); ++i)
    {
        Shape indices = LinearToIndices(i);

        float value = At(indices);

        if (value > maxValue)
        {
            maxValue = value;
            maxIndex = i;
        }
    }

    return maxIndex;
}

size_t Tensor::ArgMin() const
{
    if (!m_data || Size() == 0)
    {
        throw std::runtime_error(
            "Tensor is empty.");
    }

    float minValue = At(Shape(Rank(), 0));
    size_t minIndex = 0;

    for (size_t i = 1; i < Size(); ++i)
    {
        Shape indices = LinearToIndices(i);

        float value = At(indices);

        if (value < minValue)
        {
            minValue = value;
            minIndex = i;
        }
    }

    return minIndex;
}

Tensor Tensor::ArgMax(size_t dim) const
{
    if (dim >= Rank())
    {
        throw std::out_of_range("Invalid dimension.");
    }

    Shape resultShape = m_shape;

    resultShape.erase(resultShape.begin() + dim);

    Tensor result(resultShape);


    for (size_t i = 0; i < result.Size(); ++i)
    {
        Shape resultIndices = LinearToIndices(i, resultShape);

        Shape srcIndices = resultIndices;

        srcIndices.insert(srcIndices.begin() + dim, 0);

        float maxValue = At(srcIndices);

        size_t maxIndex = 0;

        for (size_t k = 1; k < m_shape[dim]; ++k)
        {
            srcIndices[dim] = k;

            float value = At(srcIndices);

            if (value > maxValue)
            {
                maxValue = value;
                maxIndex = k;
            }
        }

        result.At(resultIndices) = static_cast<float>(maxIndex);
    }

    return result;
}

Tensor Tensor::ArgMin(size_t dim) const
{
    if (dim >= Rank())
    {
        throw std::out_of_range("Invalid dimension.");
    }

    Shape resultShape = m_shape;

    resultShape.erase(resultShape.begin() + dim);

    Tensor result(resultShape);


    for (size_t i = 0; i < result.Size(); ++i)
    {
        Shape resultIndices = LinearToIndices(i, resultShape);

        Shape srcIndices = resultIndices;

        srcIndices.insert(srcIndices.begin() + dim, 0);

        float minValue = At(srcIndices);

        size_t minIndex = 0;

        for (size_t k = 1; k < m_shape[dim]; ++k)
        {
            srcIndices[dim] = k;

            float value = At(srcIndices);

            if (value < minValue)
            {
                minValue = value;
                minIndex = k;
            }
        }

        result.At(resultIndices) = static_cast<float>(minIndex);
    }

    return result;
}