#pragma once

#include <vector>
#include <string>
#include <memory>

class Tensor
{
public:

    using Shape = std::vector<size_t>;
    using Strides = std::vector<size_t>;
    using DataPtr = std::shared_ptr<std::vector<float>>;

public:

    Tensor();

    explicit Tensor(
        const Shape& shape);

    Tensor(
        const Shape& shape,
        float initialValue);

    Tensor(
        const Shape& shape,
        const std::vector<float>& data);

    // View constructor: aynı veriyi paylaş, farklı shape/strides
    Tensor(
        const Shape& shape,
        const Strides& strides,
        const DataPtr& data);

public:

    float* Data();
    const float* Data() const;

    const Shape& GetShape() const;
    const Strides& GetStrides() const;

    size_t Rank() const;
    size_t Size() const;

    // Verinin benzersiz sahibi miyim?
    bool IsUnique() const;

    // Kaç tensor aynı veriyi paylaşıyor?
    long UseCount() const;

    // Ham bellekteki toplam eleman sayısı (view'larda Size()'tan farklı olabilir)
    size_t RawSize() const;

public:

    void Reshape(
        const Shape& newShape);

    void Fill(float value);
    void Zero();

public:

    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    float& At(const Shape& indices);
    const float& At(const Shape& indices) const;

public:

    // Deep copy: yeni ayrı veri bloğu
    Tensor Clone() const;

    // View: aynı veriyi paylaş, son iki ekseni değiştir
    Tensor Transpose() const;

public:

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;

    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(float scalar);

public:

    std::string ToString() const;

private:

    size_t ComputeOffset(
        const Shape& indices) const;

    static size_t ComputeElementCount(
        const Shape& shape);

    static Strides ComputeContiguousStrides(
        const Shape& shape);

    static size_t ComputeOffsetWithStrides(
        const Shape& indices,
        const Shape& shape,
        const Strides& strides);

    static Shape OffsetToIndices(
        size_t offset,
        const Shape& shape);

private:

    Shape m_shape;
    Strides m_strides;
    DataPtr m_data;
};
