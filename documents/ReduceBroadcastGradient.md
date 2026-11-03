
#ReduceBroadcastGradient: 

Bu fonksiyonun amacı şu: Broadcasting sırasında genişletilmiş bir Tensor'un gradient'ini, işlemden önceki orijinal Tensor'un shape'ine geri döndürmek.
1. Önce broadcasting neden gradient topluyor?
Şöyle bir işlem düşün:

Tensor a({3, 1}, 2.0f);
Tensor b({1, 4}, 5.0f);

Tensor c = a * b;

a Tensor'ünün shape'i:

a.shape = (3,1)


b Tensor'ünün shape'i:

b.shape = (1,4)


Broadcasting sonucu:

c.shape = (3,4)


a'nın her satırındaki tek değer 4 sütuna yayılır. b'nin de tek satırdaki 4 değeri 3 satıra yayılır.

Forward işlemi:

Şimdi c'nin elemanlarının gradient'lerinin 1 olduğunu varsayalım:

c.grad = her elemanı 1 olan 3 x 4 lük matris olur ve 

a'nın ilk elemanı 4 ayrı çarpımda kullanılmıştır, dolayısıyla a.grad'in ilk elemanı:

1 + 1 + 1 + 1 = 4 olur

Aynı şey diğer satırlar için de geçerli.

b'nin ilk elemanı ise 3 farklı satırdaki çarpımda kullanılmıştır, dolayısıyla b.grad ın ilk elemanı

1 + 1 + 1 = 3 olur.

Sonuç:

a.grad.shape=(3,1)

a.grad = [4
		  4
		  4]

b.grad.shape=(1,4)

b.grad=[3 3 3 3]

ReduceBroadcastGradient işte bu toplama işlemini otomatik olarak gerçekleştiriyor.

Fonksiyonu satır satır inceleyelim

# Örneğimizde originalShape = {4} olsun:

# A) Sonuç Tensor'ünü oluşturuyoruz.

Tensor result = Tensor::Zeros(originalShape);

Bu satır:
- originalShape boyutunda bir Tensor oluşturuyor.
- Bütün elemanlarını 0 yapıyor.
- Gradient değerlerini toplayacağımız hedef Tensor'ü hazırlıyor.

Neden sıfır?

Çünkü aşağıda aynı indekslere denk gelen gradient'leri += ile biriktireceğiz.

# B) Gradient'in shape'ini alıyoruz

const Shape& gradientShape = gradient.GetShape();

Buradaki gradient, Backward sırasında bize gelen gradient Tensor'üdür.

Broadcasting örneğimizde:
gradientShape = {3, 4};
originalShape = {4};


Şu iki değişkeni birbirinden ayır:
Değişken	Anlamı
gradientShape	Geriye doğru gelen gradient'in boyutu
originalShape	Gradient'i geri döndürmek istediğimiz Tensor'un orijinal boyutu

Fonksiyonun işi, ilkini ikinciye dönüştürmek.

# C) rankDiff ne demek?

size_t rankDiff = gradientShape.size() - originalShape.size();

Buradaki size() Tensor'un eleman sayısı değil, kaç boyutlu olduğunu (rank) ifade ediyor.

Shape		Rank
{3}	    	1
{3, 4}		2
{2, 3, 4}	3

Bizim örneğimizde:

gradientShape = {3, 4};
originalShape = {4};

gradientShape.size() = 2;
originalShape.size() = 1;

rankDiff = 2 - 1 = 1;

Peki neden rankDiff hesaplıyoruz?

Çünkü broadcasting'de boyutları sağdan hizalıyoruz. Bazen Tensor'ün başına yeni boyutlar ekleniyor.

Örneğin:

originalShape = {4};
gradientShape = {3, 4};

Broadcasting:
gradientShape = {3, 4};
originalShape = {1, 4};

broadcast Edilmiş gradientShape Tensor:  (3,4) olur.

Gradient'in birinci boyutu orijinal Tensor'de yok. Bu nedenle orijinal Tensor indeksini oluştururken bir boyut kaydırması yapmamız gerekiyor.

# D) Gradient Tensor'ünün tüm elemanlarını dolaşmak

for (size_t i = 0; i < gradient.Size(); ++i)
{
    Shape gradientIndices = gradient.LinearToIndices(i);
	
gradient.Size()

Örneğimizde: gradientShape = {3, 4}; -> 3 x 4 = 12

Dolayısıyla döngü i = 0 ile i = 11 arasında çalışıyor.

LinearToIndices(i)
Burada asıl önemli nokta şu:

i, Tensor'un bellekteki sıralı eleman numarasıdır.

Ama bizim At() fonksiyonumuz mantıksal indeks istiyor.

i	gradientIndices
0	{0, 0}
1	{0, 1}
2	{0, 2}
3	{0, 3}
4	{1, 0}
5	{1, 1}
6	{1, 2}
7	{1, 3}
8	{2, 0}
9	{2, 1}
10	{2, 2}
11	{2, 3}

Shape gradientIndices = gradient.LinearToIndices(i);

satırı, sıradaki gradient elemanının hangi satırda ve sütunda bulunduğunu öğreniyor.

# originalIndices neden sıfırlarla oluşturuluyor?

Shape originalIndices(originalShape.size(), 0);

Bu satır C++'ta şu anlama geliyor:
- Vektörün uzunluğu originalShape.size() kadar olsun.
- Bütün elemanları 0 olsun.

Örneğimiz:
originalShape = {3, 1};

ozaman;
originalIndices = {0, 0}; 
şeklinde ki bu indeksleri aşağıdaki döngüde doğru konuma taşıyacağız.

# E) En önemli kısım: gradientDim

for (size_t dim = 0; dim < originalShape.size(); ++dim)
{
    size_t gradientDim = dim + rankDiff;
	
	
	Bu satırın görevi:

Orijinal Tensor'deki bir eksenin, gradient Tensor'ünde hangi eksene karşılık geldiğini bulmak.

Örneğimiz:
gradientShape = {3, 4};
originalShape = {1, 4};
rankDiff = 1;

Orijinal Tensor'ün tek ekseni, gradient'in 1. eksenine denk geliyor.

Gradient'in 0. ekseni orijinal Tensor'de bulunmuyor.

Bu yüzden rankDiff ekliyoruz.

Orjinal Tensor'ün ilk ekseni 1 boyutunda,
fakat Broadcasting sonrası bu eksen 3 boyutuna genişlemiş.

Dolayısıyla gradient'in:

gradientIndices = {0, 0};
gradientIndices = {1, 0};
gradientIndices = {2, 0};

indeksleri orijinal Tensor'de aynı elemana denk geliyor:

originalIndices = {0, 0};

Çünkü orjinal Tensor'de o eksende sadece 0 indeksli eleman var.

Dolayısıyla Orijinal boyut 1 değilse gradient indeksini kopyalıyoruz. 
Çünkü gradient indeksinin Orjinal Tensor de bir karşılığı vardır.

Orijinal boyut 1 ise sıfır olarak bırakıyoruz.
Çünkü gradiant'ın Orjinal Tensor'de karşılşığı yoktur.

if (originalShape[dim] != 1)
{
    originalIndices[dim] = gradientIndices[gradientDim];
}

# F. Gradient'leri topluyoruz

result.At(originalIndices) += gradient.At(gradientIndices);

Burada iki farklı Tensor var.

gradient.At(gradientIndices)

Genişletilmiş Tensor'den gradient değerini alıyoruz.

result.At(originalIndices)

Bu gradient'in orijinal Tensor'de hangi elemana karşılık geldiğini buluyoruz.

Bu örnekte originalIndices şu şekilde oluşur:

gradientIndices	 originalIndices	Eklenen gradient
{0,0}			 {0,0}				1
{0,1}			 {0,1}				2
{0,2}			 {0,2}				3
{0,3}			 {0,3}				4
{1,0}			 {0,0}				5
{1,1}			 {0,1}				6
{1,2}			 {0,2}				7
{1,3}			 {0,3}				8
{2,0}			 {0,0}				9
{2,1}			 {0,1}				10
{2,2}			 {0,2}				11
{2,3}			 {0,3}				12

Dikkat et: Gradient'in ilk ekseni 0, 1, 2 olabilir. 
Fakat orijinal Tensor'de ilk eksenin boyutu 1 olduğu için indeks hep 0 olarak kalıyor.

# += İşlemi

lk sütun için:

result.At({0, 0}) += 1;
result.At({0, 0}) += 5;
result.At({0, 0}) += 9; 
Sonuç 15

İkinci sütun için:

result.At({0, 1}) += 2;
result.At({0, 1}) += 6;
result.At({0, 1}) += 10;

Sonuç 18

Üçüncü sütun:
	3 + 7 + 11 = 21
Dördüncü sütun:
	4 + 8 + 12 = 24
	
Sonuç:
	result = [15 18 21 24]