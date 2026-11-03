# Autograd – Element-Wise Çarpma İşlemi

Bu notta Tensor kütüphanemize kazandırdığımız **Autograd (Automatic Differentiation)** özelliğinin, 
element-wise çarpma (`operator*`) işlemi üzerinden nasıl çalıştığını adım adım inceliyoruz.

Autograd, Tensor işlemleri yapıldığında bu işlemlerin birbirleri üzerindeki etkisini takip eder ve geriye doğru türev hesaplamamızı sağlar.


## 1. Önce neyi çözmeye çalışıyoruz?

Elimizde iki Tensor olduğunu düşünelim:

```cpp
Tensor a({}, 3.0f);
Tensor b({}, 4.0f);

Tensor L = a * b;
```

Matematiksel olarak:

L = a x b

Değerleri yerine koyarsak:

L = 3 x 4 = 12


Bilgisayar 12 sonucunu hesaplıyor. Ancak bizim istediğimiz yalnızca sonucu hesaplamak değil:

- `a` değişseydi `L` ne kadar değişirdi?
- `b` değişseydi `L` ne kadar değişirdi?

İşte burada türev devreye giriyor.


## 2. Türev bize ne söylüyor?

Bir fonksiyonun türevi, girdideki küçük bir değişimin çıktıyı ne kadar değiştireceğini anlatır.

Fonksiyonumuz:

L = a x b

`b` sabitken:

dL \ da = b

Yani `L`'nin `a`'ya göre türevi `b`'dir.

Aynı şekilde:

dL \ db = a

Yani `L`'nin `b`'ye göre türevi `a`'dır.

Değerleri yerine koyarsak:

dL \ da = 4

dL \ db = 3

Beklediğimiz sonuç:

```text
a.grad = 4
b.grad = 3
```

Şimdi bunu kodumuzun nasıl yaptığını inceleyelim.

---

## 3. `SetRequiresGrad(true)` ne yapıyor?

```cpp
a.SetRequiresGrad(true);
b.SetRequiresGrad(true);
```

Bu iki satırla Tensor'lara şunu söylüyoruz:

**Bu değişkenlerin hesaplamaya olan katkısını takip et.**

Normalde Tensor yalnızca veri tutar ve matematiksel işlem yapar. `RequiresGrad` aktif olduğunda Tensor artık Autograd sistemine de katılır.

Örneğin:

```cpp
Tensor a({}, 3.0f);
```

Burada `a` sadece 3 değerini tutuyor.

Şunu yaptığımızda:

```cpp
a.SetRequiresGrad(true);
```

Artık bu Tensor'un gradient bilgisi hesaplanmalı.

Önemli ayrım:

- `RequiresGrad`: Bu Tensor için gradient hesaplanmalı mı?
- `Grad`: Hesaplama sonucunda elde edilen gradient değeri nedir?

Bunlar birbirinden farklı şeylerdir.

---

## 4. `a * b` işlemi gerçekleştiğinde ne oluyor?

```cpp
Tensor L = a * b;
```

`operator*` fonksiyonumuz çarpma işlemini gerçekleştirip bir Tensor döndürüyor. Autograd desteğiyle fonksiyon artık iki iş yapıyor:

1. Gerçek çarpma işlemini gerçekleştiriyor.
2. Bu işlemin geriye doğru nasıl türev alınacağını kaydediyor.

Bilgisayar yalnızca 12 sonucunu bilirse, bu sonuca hangi işlemlerle ulaşıldığını ve gradient'in nereye aktarılması gerektiğini bilemez.
Bu nedenle işlem geçmişini bir **hesaplama grafiği (Computational Graph)** şeklinde saklıyoruz.

---

## 5. Hesaplama grafiği (Computational Graph)

Örneğimizde grafik şu ilişkiyi temsil ediyor:

```text
       a = 3       b = 4
          \       /
           \     /
            a * b
              |
              v
            L = 12
```

İki Tensor, çarpım işlemi üzerinden `L` sonucunu oluşturuyor.

Grafiği oluşturmak için `AutogradNode` kullanıyoruz.

Önce:

```cpp
result.m_requiresGrad = true;
result.m_isLeaf = false;
```

Burada `result`, çarpma işleminden dönen Tensor'dur.

- `m_requiresGrad = true`: Bu sonuç için de gradient takip edilecek.
- `m_isLeaf = false`: Bu Tensor doğrudan oluşturulmuş bir başlangıç değişkeni değil, başka Tensor'lardan üretilmiş bir sonuç.

Ardından:

```cpp
result.m_node =
    std::make_shared<Tinynn::Autograd::AutogradNode>();
```

Çarpma işlemine ait bir Autograd düğümü oluşturuyoruz. Düğüm, işlemin geriye doğru nasıl hesaplanacağını saklayacak.

Sonra:

```cpp
result.m_node->operationName = "Mul";
```

Bu işlem düğümünün adını `Mul` olarak belirliyoruz.

Ve en önemli kısım:

```cpp
result.m_node->parents.push_back(lhs);
result.m_node->parents.push_back(rhs);
```

Burada `L` sonucunun hangi Tensor'lardan üretildiğini kaydediyoruz:

- `L`'nin sol girdisi `a`.
- `L`'nin sağ girdisi `b`.

Böylece `Backward` çalıştığında hangi Tensor'lara geri dönmesi gerektiğini biliyor.

---

## 6. Asıl önemli kısım: `backward` fonksiyonu

```cpp
result.m_node->backward = [lhs, rhs](Tensor& output)
{
    // ...
};
```

Burada bir lambda fonksiyonu tanımlıyoruz.

Bu fonksiyonun görevi **çarpma işleminin sonucuna ulaşan gradient'i, girdilere doğru dağıtmak.**

Bu fonksiyon çarpma işlemini tekrar yapmıyor; çarpma işleminin türev kurallarını uyguluyor.

Kodun ilk bölümüne bakalım:

```cpp
if (lhs->RequiresGrad())
```

Bu kontrol, sol taraftaki Tensor için gradient hesaplamamız gerekip gerekmediğini kontrol ediyor.

Gerekiyorsa:

```cpp
if (!lhs->m_grad)
{
    lhs->m_grad =
        std::make_shared<Tensor>(
            Tensor::Zeros(lhs->GetShape()));
}
```

Gradient için bir Tensor oluşturuyoruz.

Neden sıfırlarla başlatıyoruz? Çünkü gradient değerlerini biriktireceğiz. Henüz hiçbir katkı gelmediği için başlangıç değerimiz sıfır olmalı.

Ardından:

```cpp
*(lhs->m_grad) += 
	output.Grad() * (*rhs);
```

İşte matematiğin koda dönüştüğü yer burası.

### `output.Grad()` ne demek?

Bu, çarpım sonucunun gradient değeridir:

dL \ doutput

Bizim örneğimizde `output`, doğrudan `L` olduğu için:

dL \ dL = 1

Dolayısıyla:

```cpp
output.Grad() = 1
```

### `(*rhs)` ne demek?

Bu da sağ taraftaki Tensor'un gerçek değeridir:

```cpp
(*rhs) = 4
```

İkisini çarptığımızda:

1 x 4 = 4

Dolayısıyla:

```text
a.grad = 4
```

Matematiksel olarak:

dL \ da
=
dL \ doutput
x
doutput \ da

Çarpımın yerel türevi `b` olduğu için:

dL \ da = 1 x 4 = 4

Aynı mantık sağ taraf için de geçerli:

```cpp
*(rhs->m_grad) +=
    output.Grad() * (*lhs);
```

Burada:

dL \ db = 1 x 3 = 3

Sonuç:

```text
a.grad = 4
b.grad = 3
```

### Backward sırasında gradient dağılımı

| Tensor | Gradient hesabı | Sonuç |
|---|---:|---:|
| `a.grad` | `1 × b` = `1 × 4` | **4** |
| `b.grad` | `1 × a` = `1 × 3` | **3** |

---

## 7. Peki neden `+=` kullanıyoruz?

Burası özellikle anlamamız gereken nokta.

Farklı bir örnek düşünelim:

```cpp
Tensor a({}, 5.0f);

a.SetRequiresGrad(true);

Tensor L = a * a;

L.Backward();
```

Matematiksel karşılığı:

L = a x a

L = 5 x 5 = 25

Türevini alalım:

dL \ da = a + a

dL \ da = 5 + 5 = 10

Çarpma işleminin sol tarafında `a`, sağ tarafında da yine `a` bulunuyor:

```cpp
lhs = &a;
rhs = &a;
```

Backward çalıştığında ilk işlem:

```cpp
*(lhs->m_grad) += output.Grad() * (*rhs);
```

Sonuç:

\[
a.grad = 0 + (1 x 5) = 5
\]

İkinci işlem:

```cpp
*(rhs->m_grad) += output.Grad() * (*lhs);
```

Ama `rhs` de aynı `a` değişkeni. Dolayısıyla:

\[
a.grad = 5 + (1 x 5) = 10
\]

İşte `+=` kullanmamızın nedeni bu.

Eğer `=` kullansaydık ikinci hesaplama ilkini ezerdi ve sonuç yanlışlıkla 5 olurdu.

**Gradient bir değişkene birden fazla yoldan ulaşabilir. Bu nedenle gradient değerlerini topluyoruz.**

---

## 8. `Backward()` çağrısı bütün bunları nasıl başlatıyor?

Çarpma işlemi sırasında hesaplama grafiğini oluşturduk ama henüz gradient hesaplamadık.

```cpp
L.Backward();
```

Bu satır geriye doğru hesaplamayı başlatıyor.

Mevcut Tensor mimarimizde `Backward` kabaca şu sırayla ilerliyor:

1. `BuildTopo()` ile hesaplama grafiğini dolaşıyor.
2. Düğümleri geriye doğru çalıştırmaya uygun sıraya yerleştiriyor.
3. Çıktı Tensor'unun başlangıç gradient'ini oluşturuyor.
4. Düğümlerin `backward` fonksiyonlarını çağırıyor.
5. Gradient'ler ilgili Tensor'larda birikiyor.

Peki neden başlangıç gradient'i 1?

Çünkü doğrudan `L` değerinin türevini kendisine göre alıyoruz:

dL \ dL = 1

Bu 1, hesaplamanın başlangıç noktası.

Önceki örneğimizde `d.grad = 1 * 2 * 8` işleminde gördüğümüz 1 de tam olarak bu mantıktan geliyordu.

---

## 9. Anlayıp anlamadığımızı kontrol edelim

Kod yazmamız gerekmiyor. Sadece matematiksel olarak düşünelim.

Aşağıdaki işlemi ele alalım:

```cpp
Tensor a({}, 6.0f);
Tensor b({}, 2.0f);

a.SetRequiresGrad(true);
b.SetRequiresGrad(true);

Tensor L = a * b;

L.Backward();
```

### Sorular

**Soru 1: `L` kaç olur?**

- 8
- 12
- 16

**Soru 2: `a.grad` kaç olur?**

- 2
- 4
- 6

**Soru 3: `b.grad` kaç olur?**

- 2
- 4
- 6

### Cevaplar ve açıklama

| Değer | Sonuç |
|---|---:|
| `L` | **12** |
| `a.grad` | **2** |
| `b.grad` | **6** |

Önemli nokta: Bir Tensor'un gradient'i, çarpımdaki diğer Tensor'un değeridir.

---

## Bu bölümde öğrendiklerimiz

1. `SetRequiresGrad(true)` ile bir Tensor'un gradient takibini etkinleştiriyoruz.
2. `operator*` hem çarpma işlemini yapıyor hem de Autograd düğümünü oluşturuyor.
3. `AutogradNode`, işlemin girdilerini ve geriye doğru hesaplama kuralını saklıyor.
4. `backward` lambda fonksiyonu, gelen gradient'i çarpma işleminin türev kuralına göre girdilere aktarıyor.
5. `+=` kullanarak aynı Tensor'a farklı yollardan gelen gradient katkılarını topluyoruz.
6. `Backward()` hesaplama grafiğini geriye doğru dolaşarak gradient hesaplamasını başlatıyor.

**Bir sonraki geliştirme hedefi:** Broadcasting işlemlerinin Autograd desteği. Şu ana kadar çarpma için aynı şekle sahip Tensor'ları ele aldık. Broadcasting durumunda gradient'lerin özgün Tensor şekline geri indirgenmesini de sağlamamız gerekecek.
