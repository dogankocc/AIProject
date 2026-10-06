#include "core/Tensor.h"
#include <iostream>
#include <stdexcept>

void Print2DLogical(const Tensor& t, const std::string& name)
{
    std::cout << "=== " << name << " ===" << std::endl;
    std::cout << t.ToString() << std::endl;

    const Tensor::Shape& shape = t.GetShape();

    if (shape.size() >= 2)
    {
        size_t rows = shape[shape.size() - 2];
        size_t cols = shape[shape.size() - 1];

        std::cout << "Mantiksal gorunum (At() ile):" << std::endl;
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

void PrintRawMemory(const Tensor& t, const std::string& name)
{
    std::cout << name << " HAM bellek (operator[] ile): ";
    for (size_t i = 0; i < t.RawSize(); ++i)
    {
        std::cout << t[i] << " ";
    }
    std::cout << std::endl << std::endl;
}

int main()
{
    std::cout << "========== 1. TENSOR OLUSTURMA (CONTIGUOUS) ==========" << std::endl << std::endl;

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

    Print2DLogical(t, "Original (contiguous)");
    PrintRawMemory(t, "Original");
    std::cout << "IsContiguous: " << (t.IsContiguous() ? "EVET" : "HAYIR") << std::endl;
    std::cout << "use_count: " << t.UseCount() << std::endl << std::endl;

    std::cout << "========== 2. TRANSPOSE (VIEW - AYNI VERI, FARKLI YORUM) ==========" << std::endl << std::endl;

    Tensor tTransposed = t.Transpose();

    Print2DLogical(tTransposed, "Transposed (view - NOT contiguous)");
    PrintRawMemory(tTransposed, "Transposed");

    std::cout << "IsContiguous: " << (tTransposed.IsContiguous() ? "EVET" : "HAYIR") << std::endl;
    std::cout << "Original use_count: " << t.UseCount() << std::endl;
    std::cout << "Transposed use_count: " << tTransposed.UseCount() << std::endl << std::endl;

    std::cout << "ONEMLI: Her iki tensor da AYNI ham bellegi gosteriyor!" << std::endl;
    std::cout << "Original[0] = " << t[0] << ", Transposed[0] = " << tTransposed[0] << std::endl << std::endl;

    std::cout << "========== 3. PAYLASIM TESTI ==========" << std::endl << std::endl;

    std::cout << ">>> t.At({0,0}) = 99.0f (orjinali degistir) <<<" << std::endl << std::endl;
    t.At({ 0, 0 }) = 99.0f;

    Print2DLogical(t, "Original (degisimden sonra)");
    Print2DLogical(tTransposed, "Transposed - view (degisimden SONRA - DEGISMIS OLMALI)");

    std::cout << "Dogrulama: tTransposed.At({0,0}) = " << tTransposed.At({0,0}) << " (99 olmali)" << std::endl << std::endl;

    std::cout << "========== 4. RESHAPE KURALI (SADECE CONTIGUOUS) ==========" << std::endl << std::endl;

    // Orjinal contiguous, Reshape çalışır
    std::cout << "Original uzerinde Reshape({3,2}):" << std::endl;
    Tensor t2 = t.Clone();  // Clone her zaman contiguous
    t2.Reshape({ 3, 2 });
    Print2DLogical(t2, "Reshaped (contiguous uzerinde)");

    // Transpose (view) üzerinde Reshape HATA vermeli
    std::cout << "Transposed (view) uzerinde Reshape({2,3}) denemesi:" << std::endl;
    try
    {
        tTransposed.Reshape({ 2, 3 });
        std::cout << "HATA: Exception atilmasi gerekirdi!" << std::endl;
    }
    catch (const std::runtime_error& e)
    {
        std::cout << "DOGRU: Exception yakalandi: " << e.what() << std::endl;
    }
    std::cout << std::endl;

    std::cout << "========== 5. CLONE (DEEP COPY - AYRI VERI BLOGU) ==========" << std::endl << std::endl;

    Tensor tClone = tTransposed.Clone();  // Transpose'u clone'la → contiguous olur

    Print2DLogical(tClone, "Clone of Transposed");
    PrintRawMemory(tClone, "Clone of Transposed");

    std::cout << "IsContiguous: " << (tClone.IsContiguous() ? "EVET" : "HAYIR") << std::endl;
    std::cout << "Clone use_count: " << tClone.UseCount() << " (1 olmali - kendi verisi)" << std::endl << std::endl;

    // Clone üzerinde değişiklik orjinali etkilemez
    std::cout << ">>> tClone.At({0,0}) = -1.0f <<<" << std::endl;
    tClone.At({ 0, 0 }) = -1.0f;

    std::cout << "t.At({0,0}) hala " << t.At({ 0,0 }) << " (99 kalmali, -1 degil)" << std::endl << std::endl;

    std::cout << "========== TUM TESTLER TAMAMLANDI ==========" << std::endl;

    return 0;
}
