#include "core/Tensor.h"
#include <iostream>

void Print2DTensor(const Tensor& t, const std::string& name)
{
    std::cout << name << ": " << t.ToString() << std::endl;
    const Tensor::Shape& shape = t.GetShape();
    
    if (shape.size() >= 2)
    {
        size_t rows = shape[shape.size() - 2];
        size_t cols = shape[shape.size() - 1];
        
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                Tensor::Shape indices = { i, j };
                std::cout << t.At(indices) << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

void PrintRawData(const Tensor& t, const std::string& name)
{
    std::cout << name << " raw memory: ";
    const float* rawPtr = t.Data();
    for (size_t i = 0; i < t.RawSize(); ++i)
    {
        std::cout << rawPtr[i] << " ";
    }
    std::cout << std::endl << std::endl;
}

int main()
{
    // 2x3 matris oluştur
    Tensor t({ 2, 3 });
    
    // Değerleri doldur:
    // 1 2 3
    // 4 5 6
    t.At({ 0, 0 }) = 1.0f;
    t.At({ 0, 1 }) = 2.0f;
    t.At({ 0, 2 }) = 3.0f;
    t.At({ 1, 0 }) = 4.0f;
    t.At({ 1, 1 }) = 5.0f;
    t.At({ 1, 2 }) = 6.0f;

    Print2DTensor(t, "Original");
    PrintRawData(t, "Original");

    std::cout << "Original use_count: " << t.UseCount() << std::endl << std::endl;

    // Transpose al (VIEW - AYNI VERİYİ PAYLAŞIR)
    Tensor tTransposed = t.Transpose();

    Print2DTensor(tTransposed, "Transposed (View)");
    PrintRawData(tTransposed, "Transposed (View)");

    std::cout << "After Transpose - Original use_count: " << t.UseCount() << std::endl;
    std::cout << "After Transpose - Transposed use_count: " << tTransposed.UseCount() << std::endl << std::endl;

    // ÖNEMLİ TEST: Orjinali değiştir, Transpose da değişiyor mu?
    std::cout << ">>> t.At({0,0}) = 99.0f yapiyoruz <<<" << std::endl << std::endl;
    t.At({ 0, 0 }) = 99.0f;

    Print2DTensor(t, "Original (sonra)");
    Print2DTensor(tTransposed, "Transposed (sonra - view)");

    // Doğruluk kontrolü
    bool ok = true;
    ok = ok && (tTransposed.GetShape() == Tensor::Shape({ 3, 2 }));
    ok = ok && (tTransposed.At({ 0, 0 }) == 99.0f);  // Değişikliği yansıtmalı
    ok = ok && (tTransposed.At({ 0, 1 }) == 4.0f);
    ok = ok && (tTransposed.At({ 1, 0 }) == 2.0f);
    ok = ok && (tTransposed.At({ 1, 1 }) == 5.0f);
    ok = ok && (tTransposed.At({ 2, 0 }) == 3.0f);
    ok = ok && (tTransposed.At({ 2, 1 }) == 6.0f);
    ok = ok && (t.UseCount() == 2);  // İki tensor aynı veriyi paylaşıyor

    std::cout << "Tum testler gecti mi? " << (ok ? "EVET" : "HAYIR") << std::endl;

    return 0;
}
