# Kerem için özet: Posterior

> Bu dosya ekip içi bir not. İç makine adları ve yollar içeriyor. **Repoyu public yapmadan önce silin
> ya da taşıyın.** Her şeyin tarihi Ekim 2026.

## Bir cümlede

Posterior, bir veritabanına gelecekle ilgili düz bir soru sormanı sağlıyor ("Önümüzdeki 30 günde hangi
satıcılar satışı bırakır?"). Her varlık için kalibre bir olasılık döndürüyor, soruyu nasıl anladığını
ve hangi kolonları sızıntı yüzünden kullanmadığını gösteriyor.

- Veri bilimcinin kararlarını Jev tarzı bir **decision model** veriyor (Ollaya üzerinden açık model `winnow:e4b`).
- İstatistiği **TabPFN-3.5** yapıyor (TabPFN-Rel üzerinden).
- Hiçbir model SQL yazmıyor. Küçük, tipli bir dilbilgisi kararları RelArena görevine çeviriyor.

**Hackathon:** Prior Labs TabPFN-3.5 Hackathon, https://platform.priorlabs.ai/hackathon-3.5

**Son teslim: 6 Ekim 23:59 CEST** (Türkiye saatiyle 7 Ekim 00:59).

Puanlama:
- %50: TabPFN-3.5'i sergilemek
- %30: yaratıcılık
- %20: teknik kalite ve tekrar üretilebilirlik

## Neden bu fikir

TabPFN, model eğitme adımını saniyelere indirdi. Ama ondan önceki adımlar hâlâ elle yapılıyor:
- soruyu tanımlamak (hedef, pencere, hangi varlıklar risk altında),
- tabloları birleştirmek,
- tahmin anında bilinmeyen kolonları ayıklamak.

Prior Labs kendi RelArena dokümanında bunu açıkça yazıyor: "What even is a predictive task over a
relational database? How to answer this well remains an open problem."

Bu adımlardaki kararların hepsi küçük, tipli sorular: bir seçim ya da bir evet/hayır. Decision model'ler
tam olarak bunu milisaniyede ve olasılıkla yapıyor. TabPFN ise kolon adlarını hiç görmüyor; anlam
bilmiyor ama istatistikte çok güçlü. İkisi birbirinin eksiğini kapatıyor.

## Nerede ne var

- **Repo:** https://github.com/cobanov/posterior (şu an private)
- **Sunum sayfası:** https://claude.ai/artifact/9pzB6pMDBsPE1NE2fdegkU (bağlantıya sahip herkes açabilir). Repodaki kopyası `docs/index.html`.
- **README.md:** Kurulum, kullanım, sonuç tabloları, sınırlamalar. Jüri bunu okuyacak, İngilizce.
- **PLAN.md:** Başlangıç planı ve neyin bittiği.
- **Kod (`src/posterior/`):**

  | Dosya | Görevi |
  |---|---|
  | `schema.py` | Veritabanını DuckDB'ye yükler; anahtar, bağlantı ve olay zamanını bulur |
  | `formulate.py` | Soruyu slotlara böler ve her slotu decision model'e sorar |
  | `spec.py` | Dilbilgisi ve SQL derleyicisi |
  | `clarify.py` | Okumalar çelişirse tek bir netleştirme sorusu üretir |
  | `audit.py` | Sızıntı denetimi |
  | `learn.py` | TabPFN-Rel ile öğrenme, backtest ve canlı tahmin |
  | `engine.py` | Hepsini birleştirir |
  | `cli.py`, `api.py`, `mcp_server.py` | Arayüzler |

- **Değerlendirmeler:**
  - `eval/relbench_formulation.py`
  - `eval/leakage.py`
  - Ham sonuçlar `eval/results/` içinde.
- **Testler:** `tests/` altında 24 çevrimdışı test. Model sunucusu gerekmiyor.

## Altyapı (Mert'in ortamı)

- **Çalışma makinesi `white` (RTX 3090):**
  - Ollaya 0.9.0 systemd servisi olarak çalışıyor (`127.0.0.1:11435`); `winnow:e4b`, `clef` ve `laya` çekili.
  - Kod `~/Developer/posterior`'da, veri `~/data/olist` ve `~/data/relbench/{rel-f1,rel-event,rel-hm,rel-trial}` altında.
- **TabPFN:**
  - Prior Labs API anahtarıyla, hosted API üzerinden çalışıyor (`POSTERIOR_TABPFN=client`).
  - Yerel GPU için hesabın TabPFN-3.5 lisansını kabul etmiş olması gerekiyor (ux.priorlabs.ai, Licenses sekmesi). Bu henüz yapılmadı.
  - Anahtar kişiye özel; kendi makinende kendi anahtarını kullan.
- **4090 ve 5090:** Başka eğitimler (earth-dreams LoRA) için kullanılıyor; bu projede kullanılmadı.

## Kendi makinende çalıştırmak

```sh
git clone https://github.com/cobanov/posterior && cd posterior
uv sync --extra api --extra serve --group dev       # GPU ile yerel TabPFN için: --extra local
uv run pytest                                       # 24 test, model gerekmiyor

# Decision model: Ollaya (https://github.com/ollaya-dev/ollaya)
curl -fsSL https://ollaya.dev/install.sh | sh && ollaya pull winnow:e4b
# Ollaya başka bir makinedeyse: export POSTERIOR_DECIDER_URL=http://<host>:11435

# TabPFN anahtarı (ux.priorlabs.ai)
mkdir -p ~/.config/posterior && echo "TABPFN_TOKEN=<kendi anahtarın>" > ~/.config/posterior/env

./scripts/get_olist.sh data/olist                   # Olist, Kaggle hesabı gerekmez
uv run posterior formulate --db data/olist "Which sellers will stop selling in the next 30 days?"
uv run posterior ask       --db data/olist "Which sellers will stop selling in the next 30 days?"
uv run posterior serve     --db data/olist --port 8787     # HTTP API
uv run python scripts/mcp_smoke.py data/olist              # MCP'yi gerçek istemciyle dener
```

Notlar:
- Bir `ask` koşusu yaklaşık 2,5 dakika sürüyor: biri backtest, biri canlı tahmin için iki TabPFN-Rel fit'i.
- Aynı soru tekrar sorulunca sonuç `runs/` klasöründen geliyor.
- Decision model cevapları `cache/decisions.jsonl`'de saklanıyor.

## Son bulgular (ölçülmüş)

| Ölçüm | Sonuç |
|---|---|
| Olist, tek İngilizce cümleden satıcı churn'ü, sızıntı denetimiyle | ROC-AUC **0.777-0.780** |
| Karşılaştırma | Sabit tahmin 0.500, satıcının kendi geçmişi 0.670, Prior Labs'in elle yazdığı görev (bizim koşumuz) 0.776 |
| Derleyici doğruluğu | Prior Labs'in elle yazdığı SQL ile 6.136 satırın 6.136'sında aynı etiket |
| Olist'e geleceği bilen 4 kolon enjekte edildi | Sadece anlam 3/4, sadece veri (TabPFN) 2/4, **ikisi birlikte 4/4** yakaladı. Normal kolonların hiçbiri düşmedi. Prior Labs'in elle seçtiği kolon listesiyle %88 uyum. |
| Sızıntılı veride TabPFN-3.5 | Denetimsiz **1.000** (geleceği görüyor), denetimli **0.774** |
| RelBench, 11 görev | Cevap tipi ve pencere 11/11, varlık 9/11. H&M'nin iki görevi ilk okumada, F1 top-3 bir netleştirmeyle birebir doğru. Dilbilgisinin ifade edebildiği 5 görevde netleştirmeyle ortalama 0.67 etiket uyumu. |
| Jev uyumlu tek istek | "Satıcı bırakır mı" TabPFN'den 0.717, "not şikâyet mi" decision model'den 0.983 |

Öğrendiklerimiz:

1. **Decision model'e atomik ve yalnız soru cümlesiyle soru sor.** State'e şemayı koymak cevabı
   bozuyor. "Satışı bırakır" sorusu 0.37'den 0.89'a çıktı. RelBench'te varlık seçimi 4/6'dan 6/6'ya çıktı.
2. **Veri sızıntıyı doğrulayabilir ama bir kolonu aklayamaz.** Olist, teslimatlar bittikten sonra dışa
   aktarılmış; yeni satırlarda boş teslim tarihi yok. Teslim tarihlerini yalnız anlam kanıtı yakaladı.
3. **Anahtarlı bir tablodaki zaman kolonu otomatik olarak olay zamanı sayılmamalı.** "Son sipariş
   tarihi" satıcılar tablosunun olay zamanı sanılınca denetim kör kaldı. Enjekte sızıntılardan 0/4
   yakalandı; düzeltmeden sonra 4/4.
4. **Netleştirme sorusu güvenlik ağı.** Örnek: "Her ürün gelecek ay ne kadar gelir?" sorusunda "tüm
   ürünler" ile "satışı olan ürünler" okumaları etiketlerde yalnızca %12 uyuşuyor. Sistem sessizce
   tahmin etmek yerine tek bir soru soruyor.

Zayıf noktalar (README'de de yazıyor):
- **Dilbilgisinin dışında kalanlar:** rel-event ve rel-trial görevlerinin çoğu bunları gerektiriyor:
  - küme filtresi (`status IN (...)`),
  - iki adımlı bağlantı,
  - başka tablodan değer,
  - tablo birleşimi,
  - pencere fonksiyonu.
- **Çalıştırılmayan veri setleri:** rel-amazon, rel-stack ve rel-avito, 16 GB RAM'li makinede çalıştırılmadı. hope (47 GB RAM) bunun için daha uygun.
- **Sınırdaki kolon:** Tahmini teslim tarihinin sızıntı skoru (~0.77) 0.75 eşiğine çok yakın, bu yüzden sonuç koşudan koşuya 0.777 ile 0.780 arasında oynuyor.

## Yapılacaklar

Öncelik sırasıyla:

- [ ] **ux.priorlabs.ai → Licenses:** TabPFN-3.5 lisansını kabul et. Hesap sahibi yapmalı. Sonra `POSTERIOR_TABPFN=local` ile yerel GPU'ya geçilir.
- [ ] **Hackathon'a katıl** (platform.priorlabs.ai) ve ek API kredisi iste.
- [ ] **Demo video:** Önerilen akış:
  1. `posterior ask` ile tek cümleden tahmin,
  2. sızıntı deneyi (1.000 → 0.774),
  3. Jev uyumlu karışık istek,
  4. Claude Code içinde MCP.
- [ ] **Bu dosyayı kaldır, repoyu public yap, gönder.** Gönderim formu: repo bağlantısı (Apache-2.0), proje açıklaması, isteğe bağlı video bağlantısı.
- [ ] **İsteğe bağlı, vakit kalırsa:**
  - Dilbilgisine küme filtresi ve iki adımlı bağlantı ekle; rel-event'i yükseltir.
  - hope'ta rel-amazon değerlendirmesini çalıştır.

---

## Claude'un son mesajı (aynen)

Evet, çalışıyor. Bu sabah white'ta GitHub'dan sıfırdan klonlayıp kurdum ve her şeyi yeniden doğruladım:

| Kontrol | Sonuç |
|---|---|
| Testler | 24/24 geçti |
| CLI | `schema` ve `formulate` çalışıyor; soru doğru okunuyor (0.57) |
| HTTP API | `/health` tamam; Jev uyumlu istekte iki soru decision model'e gitti (şikâyet 0.98, iade 0.99) |
| Uçtan uca koşu | 167 saniyede bitti, **ROC-AUC 0.777**, 1.278 satıcı için tahmin üretti |

Doğrulama sırasında bir şey öğrendim: sonuçlar koşudan koşuya 0.777 ile 0.780 arasında oynayabiliyor. Sebebi, "tahmini teslim tarihi" kolonunun sızıntı skorunun (~0.77) 0.75 eşiğinin hemen üstünde olması. Temiz klonda decision model'in kayıtlı cevapları olmadığı için bu kolon bu kez düşürüldü. README'de de sayfada da bunu açıkça yazdım.

**Sunum sayfası güncel (sürüm 4):** https://claude.ai/artifact/9pzB6pMDBsPE1NE2fdegkU
- **Gerçek sonuçlar:** Gece güncellediğim sayılar yerinde. Sayfada artık "temsili" değer kalmadı.
- **Yeni bölümler:**
  - **Gece ne oldu:** 8 adım. İçlerinde sonradan en faydalı bulguya dönüşen iki hata var: hiçbir kolonu düşürmeyen sızıntı denetimi ve denetimi kör eden "son sipariş tarihi" hatası.
  - **Bulgular:** 4 madde, her biri ölçümle destekli.
  - **Sırada ne var.**
- **Mobil düzen:**
  - Karşılaştırma tablosu telefonda yatay kaydırma yerine her adım için bir karta dönüşüyor.
  - Olasılık etiketleri artık kesilmiyor, çubuklarının üstüne iniyor.
  - Sızıntı tablosu küçük ekranda alt alta diziliyor.
  - Başlıklar ve dokunma alanları küçük ekrana göre ayarlandı.
  - 320, 360, 390 ve 430 piksellik genişliklerde ölçtüm, yatay taşma yok. Ölçüm sırasında uzun risk etiketlerinin 439 piksele taştığını buldum ve düzelttim.

Değişiklikler push edildi; sayfanın repodaki kopyası `docs/index.html`'de. Obsidian notuna sabahki doğrulama satırını ekledim.

Senden beklenenler aynı: TabPFN-3.5 lisansını kabul etmek (yerel GPU için), hackathon kaydı, demo video, repoyu public yapıp göndermek. Son teslim 6 Ekim 23:59 CEST.
