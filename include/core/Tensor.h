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
        const DataPtr& data,
        size_t offset);

public:

    // Bellek doğrudan erişim
    float* Data();
    const float* Data() const;

    // Özellikler
    const Shape& GetShape() const;
    const Strides& GetStrides() const;

    size_t Rank() const;
    size_t Size() const;           // Mantıksal eleman sayısı
    size_t RawSize() const;        // Ham bellekteki toplam eleman

    // Paylaşım durumu
    bool IsUnique() const;
    long UseCount() const;

    // Contiguous kontrolü: m_strides standart mı?
    bool IsContiguous() const;

public:

    // SADECE Contiguous Tensor üzerinde geçerli
    void Reshape(const Shape& newShape);

    // Tüm elemanları doldur
    void Fill(float value);
    void Zero();

public:

    // operator[]: HAM bellek indeksi (contiguous olmayanlarda dikkatli!)
    float& operator[](size_t rawIndex);
    const float& operator[](size_t rawIndex) const;

    // At(): MANTIKSAL indeks (her zaman güvenli)
    float& At(const Shape& indices);
    const float& At(const Shape& indices) const;

public:

    // Deep copy: yeni ayrı veri bloğu, her zaman Contiguous
    Tensor Clone() const;

    // View: aynı veriyi paylaş, son iki ekseni yer değiştir
    // Sonuç: Contiguous OLMAYABILIR
    Tensor Transpose() const;
    
    // İstenilen iki ekseni değiştir
    Tensor Transpose(
        size_t dim1,
        size_t dim2) const;

    Tensor Slice(size_t dim, size_t index) const;
public:

    // Aritmetik işlemler
    // Eğer her iki taraf da Contiguous ise HIZLI yol kullanılır
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;

    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    // In-place işlemler
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(float scalar);

public:

    std::string ToString() const;

private:

    // Bu nesnenin stride'ı ile offset hesapla
    size_t ComputeOffset(const Shape& indices) const;

    // Yardımcı statik fonksiyonlar
    static size_t ComputeElementCount(const Shape& shape);
    static Strides ComputeContiguousStrides(const Shape& shape);

private:

    Shape m_shape;
    Strides m_strides;
    DataPtr m_data;
    size_t m_offset;
};
