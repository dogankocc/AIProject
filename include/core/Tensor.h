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

    // Slice, seçilen bir boyuttaki belirli bir indeksi sabitler, o boyutu tensor'dan kaldırır ve aynı veriyi paylaşan yeni bir view tensor döndürür.
    // Örn. shape={2,3,4}, Slice(1,1) -> shape={2,4} (1. eksenin 1. indeksini sabitle)
	// data ve offset aynı kalır, sadece shape ve strides değişir.
    /**
    auto row = x.Slice(0,1); Bu şu anlama geliyor:
	Yani 0. boyut 1. indeksi seçer, yeni View döner.
    Oluşturulan yeni view tensorun offset'i (m_offset) yeniden hesaplanır.
	Yani m_offset bu yeni Tensor'ün,
    paylaşılan ham veri(m_data) 'nin neresinden başlayacağını belirler.
    Yani m_offset, Slice ile seçilen indeksin ham veri üzerindeki konumunu gösterir.
    Tensorun başladığı fiziksel adresi değiştiren işlemdir.
    */
    Tensor Slice(size_t dim, size_t index) const;

	// Yeni tensorun boyutları hangi sırayla oluşturulacaksa, o sırayla eksenleri,
    // shape ve stride'ları yeniden düzenleyen view oluşturur.
	// Örnek: shape={2,3,4}, Permute({2,0,1}) -> dims[0] = oldDims[2] = 4, dims[1] = oldDims[0] = 2, dims[2] = oldDims[1] = 3, shape = {4,2,3}, strides buna göre yeniden hesaplanır.
    // dims[i], yeni i. eksenin hangi eski eksenden geleceğini belirtir.
    Tensor Permute(
        const std::vector<size_t>& dims) const;

	// View: aynı veriyi paylaş, farklı shape (contiguous olmalı)
    Tensor View(const Shape& shape) const;
    
    // Tüm boyutları tek boyuta indirir.
    Tensor Flatten() const;

	// Boyutu 1 olan eksenleri kaldırır. Örn. shape={2,1,3,1} -> shape={2,3}
	// data ve offset aynı kalır, sadece shape ve strides değişir. Eğer boyutu 1 olan eksen yoksa, aynı tensor döner.
    // Aslında stride değişmiyor. Sadece shape'den sildiğimiz boyutlara karşılık gelen stride'ları da siliyoruz.
    Tensor Squeeze() const;

    // Verilen eksen indeksine uzunluğu 1 olan yeni bir boyut ekleyen view 
    Tensor Unsqueeze(size_t dim) const;
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
	// Tensorun görünüşü Ör.{2,3,4} demek 3boyutlu bir array, 0. boyut 2 elemanlı, 1. boyut 3 elemanlı, 2. boyut 4 elemanlı demektir 
    Shape m_shape;
    // İlgili eksendeki indeks 1 arttığında, fiziksel bellekte(m_data' da) kaç eleman ilerlerim?
    Strides m_strides;
	// Tek boyutlu ham veriyi tutan shared_ptr. Tensor view'ları aynı ham veriyi paylaşır.
    DataPtr m_data;
	// Mantıksal veri(Shape)'den seçilen bir elemanın, gerçek veri(m_data) üzerindeki konumu
    // gerçek veri üzerindeki  View'lar farklı offset ile aynı ham veriyi paylaşabilir.
    size_t m_offset; 
};
