# CLAUDE.md - Otonom Araştırma Ajanı Talimatları

## SEN KIMSİN?
Sen bir yapay zeka araştırma ajanısın. Görevin: `train.py` dosyasını düzenleyerek
**val_bpb** (validation bits-per-byte) değerini düşürmek. Düşük val_bpb = daha iyi model.

## KURAL: prepare.py DOSYASINA ASLA DOKUNMA
`prepare.py` sabitleri, veri yükleme, tokenizer ve değerlendirme içerir. Değiştirme.

## KURAL: results.tsv DOSYASINA COMMIT ATMA
`results.tsv` çalışma dosyasıdır. Git'e commit etme.

---

## ADIM 1: BAŞLANGIÇ KONTROLÜ (Her seferinde yap)

```bash
# Hangi branch'teyiz?
git branch --show-current

# Son commit ne?
git log --oneline -3

# Sonuç dosyası var mı?
cat results.tsv
```

---

## ADIM 2: TRENİ BAŞLAT

```bash
cd D:/autoresearch

# Çalıştır ve log'a yaz
uv run train.py > run.log 2>&1
```

Bu komut bittiğinde dosya çalışmış olacak. Bekle.

---

## ADIM 3: SONUCU OKU

```bash
# Sonuçları oku
grep "^val_bpb:" run.log
grep "^peak_vram_mb:" run.log
grep "^num_steps:" run.log
grep "^num_params_M:" run.log
```

Eğer `grep "^val_bpb:" run.log` boş dönerse, program çökmüştür:
```bash
# Çökme nedenini gör
tail -n 30 run.log
```

---

## ADIM 4: KARAR VER (KEEP mi, DISCARD mi?)

### KABUL KRİTERLERİ (BUNLARI EZBERLE)

Sonuçları `results.tsv` dosyasındaki en iyi sonucu bul:
```bash
# En düşük val_bpb'yi bul
awk -F'\t' 'NR>1 && $2>0 {print $0}' results.tsv | sort -t$'\t' -k2 -n | head -1
```

Karar tablosu:

| Durum | Karar | Neden |
|---|---|---|
| val_bpb < en iyi sonuç | **KEEP** | İyileşme var |
| val_bpb == en iyi sonuç (±0.005) | **DISCARD** | Fark yok, zaman kaybı |
| val_bpb > en iyi sonuç | **DISCARD** | Kötüleşti |
| Program çöktü (OOM, hata) | **CRASH** | Kod çalışmıyor |

### ÖRNEK:
- En iyi sonuç: 6.271457
- Yeni sonuç: 6.150000 → **KEEP** (düşük = iyi)
- Yeni sonuç: 6.280000 → **DISCARD** (aynı seviye)
- Yeni sonuç: 6.500000 → **DISCARD** (kötüleşti)
- Yeni sonuç: 0.000000 → **CRASH** (çöktü)

---

## ADIM 5: SONUCU KAYDET

### 5a. KEEP (İyileşme varsa):

```bash
# Kısa commit hash al
COMMIT=$(git rev-parse --short HEAD)

# val_bpb, memory_gb, status, description bilgilerini al
# Aşağıdaki satırı results.tsv'ye ekle:
echo -e "${COMMIT}\t$(grep '^val_bpb:' run.log | awk '{print $2}')\t$(grep '^peak_vram_mb:' run.log | awk '{printf \"%.1f\", $2/1024}')\tkeep\t$(date +%H:%M) deney açıklaması" >> results.tsv

# Git'te tut (zaten commit edilmiş durumda, bir şey yapma)
```

### 5b. DISCARD (İyileşme yoksa):

```bash
# Son commit'i geri al
git reset --hard HEAD~1

# results.tsv'ye discard olarak kaydet
echo -e "discard\t$(grep '^val_bpb:' run.log | awk '{print $2}')\t0.0\tdiscard\t$(date +%H:%M) deney açıklaması" >> results.tsv
```

### 5c. CRASH (Program çöktüyse):

```bash
# Son commit'i geri al
git reset --hard HEAD~1

# results.tsv'ye crash olarak kaydet
echo -e "crash\t0.000000\t0.0\tcrash\t$(date +%H:%M) deney açıklaması" >> results.tsv
```

---

## ADIM 6: YENİ DENEY TASARLA

Aşağıdaki değişikliklerden BİRİNİ seç ve uygula. Her seferinde SADECE BİR değişiklik yap.

### Değişiklik Seçenekleri (Sırayla dene):

**Seçenek 1: WINDOW_PATTERN değiştir**
```python
# train.py'de bul:
WINDOW_PATTERN = "SSSL"
# Şunu yap:
WINDOW_PATTERN = "L"
```
Neden: "L" tüm katmanlarda tam bağlam kullanır. Basit ve etkili.

**Seçenek 2: DEPTH değiştir**
```python
# train.py'de bul:
DEPTH = 6
# Şunu yap:
DEPTH = 8
```
Neden: Daha derin model daha çok şey öğrenebilir.

**Seçenek 3: DEPTH düşür**
```python
# train.py'de bul:
DEPTH = 6
# Şunu yap:
DEPTH = 4
```
Neden: Daha küçük model daha hızlı eğitilir, aynı sürede daha çok adım atar.

**Seçenek 4: DEVICE_BATCH_SIZE değiştir**
```python
# train.py'de bul:
DEVICE_BATCH_SIZE = 4
# Şunu yap:
DEVICE_BATCH_SIZE = 2
```
Neden: Daha küçük batch daha sık güncelleme yapar.

**Seçenek 5: TOTAL_BATCH_SIZE değiştir**
```python
# train.py'de bul:
TOTAL_BATCH_SIZE = 2**17
# Şunu yap:
TOTAL_BATCH_SIZE = 2**16
```
Neden: Daha küçük total batch daha sık optimizer adımı.

**Seçenek 6: Learning Rate değiştir**
```python
# train.py'de bul:
EMBEDDING_LR = 0.6
# Şunu yap:
EMBEDDING_LR = 0.3
```
veya
```python
EMBEDDING_LR = 1.0
```

**Seçenek 7: MATRIX_LR değiştir**
```python
# train.py'de bul:
MATRIX_LR = 0.04
# Şunu yap:
MATRIX_LR = 0.02
```
veya
```python
MATRIX_LR = 0.08
```

**Seçenek 8: Softcap kaldır**
```python
# train.py'de bul:
softcap = 15
logits = softcap * torch.tanh(logits / softcap)
# Şunu yap:
logits = logits  # softcap kaldırıldı
```

**Seçenek 9: Head dimension değiştir**
```python
# train.py'de bul:
HEAD_DIM = 64
# Şunu yap:
HEAD_DIM = 128
```

**Seçenek 10: Weight decay değiştir**
```python
# train.py'de bul:
WEIGHT_DECAY = 0.2
# Şunu yap:
WEIGHT_DECAY = 0.0
```

---

## ADIM 7: DEĞİŞİKLİĞİ COMMIT ET VE ÇALIŞTIR

```bash
# Değişikliği commit et
git add train.py
git commit -m "deney: [kısa açıklama]"

# Çalıştır
uv run train.py > run.log 2>&1

# ADIM 3'e dön
```

---

## DÖNGÜ: ADIM 3 → 4 → 5 → 6 → 7 → 3 → ...

Bu döngüyü gece boyunca tekrarla. Her deney ~1 saat sürer.

---

## HATA DURUMLARI

### OOM (Out of Memory):
```python
# DEVICE_BATCH_SIZE düşür
DEVICE_BATCH_SIZE = 2  # veya 1
```

### Loss NaN oldu:
```python
# Learning rate düşür
EMBEDDING_LR = 0.3
MATRIX_LR = 0.02
```

### Loss hiç düşmüyor:
```python
# Learning rate artır
EMBEDDING_LR = 1.0
MATRIX_LR = 0.08
```

### Program açılmıyor (import error):
```bash
# Bağımlılıkları kontrol et
uv sync
```

---

## SONLANDIRMA KRİTERİ

Aşağıdaki durumlarda dur:
1. val_bpb **5.0**'nın altına düştüyse → iyi sonuç, dur ve raporla
2. 10 deney ardışık DISCARD ise → farklı bir yaklaşım dene
3. 3 ardışık CRASH varsa → son stabil commit'e dön

---

## GITHUB BİLGİLERİ

```bash
# Token (zaten yapılandırılmış olmalı)
# Remote: https://github.com/aysberk/autoresearch.git
# Branch: autoresearch/rx580-mar26

# Eğer remote yoksa:
git remote add origin https://github.com/aysberk/autoresearch.git
```

---

## ÖRNEK DÖNGÜ AKIŞI

```
1. git branch --show-current     → autoresearch/rx580-mar26
2. cat results.tsv               → 6.271457 en iyi
3. WINDOW_PATTERN = "L" yap
4. git commit -m "deney: L pattern"
5. uv run train.py > run.log 2>&1
6. grep "^val_bpb:" run.log      → 6.150000
7. 6.15 < 6.27 → KEEP
8. echo "... 6.150000 ... keep ..." >> results.tsv
9. Sonraki deneye geç
```

---

## DİKKAT!

- Her seferinde SADECE BİR değişiklik yap
- prepare.py'ye ASLA dokunma
- results.tsv'ye commit atma
- Çöktüğünde HEAD~1'e geri dön
- Gece boyunca durma, devam et
