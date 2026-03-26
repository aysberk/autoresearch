# AGENTS.md - Bu Projedeki Ajanlar İçin Talimatlar

## Proje Hakkında
Bu repo, Karpathy'nin `autoresearch` projesinin RX 580 DirectML adaptasyonudur.
Amaç: Bir ajanın (ben veya yerel AI) `train.py`'yi düzenleyerek val_bpb'yi düşürmesi.

## Dosya Yapısı
```
train.py        ← Ajan BUNU düzenler (model, optimizer, hiperparametreler)
prepare.py      ← DOKUNMA (sabitler, veri, tokenizer, değerlendirme)
program.md      ← Otonom döngü talimatları (Karpathy orijinali)
CLAUDE.md       ← Yerel ajan (LM Studio) için talimatlar
AGENTS.md       ← Bu dosya (benim talimatlarım)
results.tsv     ← Deney sonuçları (git'e commit ETME)
ralph.sh        ← Otonom döngü scripti
```

## Benim (OpenCode) Rolüm
1. Kullanıcıya yardımcı olmak
2. CLAUDE.md ve ralph.sh'yi oluşturmak
3. İlk baseline'ı kaydetmek
4. Kullanıcıya yerel ajanı nasıl başlatacağını anlatmak

## Yerel Ajanın (Qwen/LM Studio) Rolüm
1. CLAUDE.md'deki talimatları takip etmek
2. train.py'yi düzenlemek
3. uv run train.py çalıştırmak
4. Sonuçları results.tsv'ye yazmak
5. İyileşme varsa commit'i tutmak, yoksa geri almak
6. Tekrar başa dönmek (Ralph döngüsü)
