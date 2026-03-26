# LM Studio Kurulum Rehberi

## Adım 1: LM Studio İndir ve Kur
https://lmstudio.ai adresinden indir.

## Adım 2: Model İndir
LM Studio içinde arama yap: `unsloth/qwen3.5-35b-a3b`
İndir (21.6GB). Q4_K_M quantization önerilir.

## Adım 3: Yerel Sunucuyu Başlat
1. LM Studio'da "Developer" sekmesine git
2. Modeli yükle
3. Server'ı başlat (port 1234)
4. "Status: Running" görmelisin

## Adım 4: Environment Variables Ayarla

PowerShell'de (tek seferlik):
```powershell
# Kalıcı olarak ayarla
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_BASE_URL", "http://localhost:1234", "User")
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_AUTH_TOKEN", "lmstudio", "User")
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_MODEL", "unsloth/qwen3.5-35b-a3b", "User")
```

Veya her oturum başında:
```powershell
$env:ANTHROPIC_BASE_URL = "http://localhost:1234"
$env:ANTHROPIC_AUTH_TOKEN = "lmstudio"
$env:ANTHROPIC_MODEL = "unsloth/qwen3.5-35b-a3b"
```

## Adım 5: Claude Code Kur (Eğer yüklü değilse)
```powershell
npm install -g @anthropic-ai/claude-code
```

## Adım 6: Test Et
```powershell
claude -p "Merhaba, çalışır mısın?" --allowedTools Read,Write,Bash
```

## Adım 7: Ralph Döngüsünü Başlat

### PowerShell (Önerilen):
```powershell
cd D:\autoresearch
$env:ANTHROPIC_BASE_URL = "http://localhost:1234"
$env:ANTHROPIC_AUTH_TOKEN = "lmstudio"
$env:ANTHROPIC_MODEL = "unsloth/qwen3.5-35b-a3b"
.\ralph.ps1 -MaxExperiments 24
```

### Git Bash / WSL:
```bash
cd /mnt/d/autoresearch
export ANTHROPIC_BASE_URL=http://localhost:1234
export ANTHROPIC_AUTH_TOKEN=lmstudio
export ANTHROPIC_MODEL="unsloth/qwen3.5-35b-a3b"
chmod +x ralph.sh
./ralph.sh 24
```

## Dikkat Edilecekler
- LM Studio açık kalmalı (sunucu kapanmasın)
- Her deney ~1 saat sürer (RX 580)
- 24 deney ≈ 24 saat
- Gece boyunca bırakabilirsin
- Sabah results.tsv dosyasına bak
