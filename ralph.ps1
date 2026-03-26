# ralph.ps1 - Otonom Araştırma Döngüsü (Windows + LM Studio + Claude Code)
#
# KULLANIM:
#   .\ralph.ps1
#   .\ralph.ps1 -MaxExperiments 50
#
# ÖN KOŞULLAR:
#   1. LM Studio açık ve model yüklü (port 1234)
#   2. Claude Code kurulu
#   3. Aşağıdaki environment variable'lar ayarlı:
#      $env:ANTHROPIC_BASE_URL = "http://localhost:1234"
#      $env:ANTHROPIC_AUTH_TOKEN = "lmstudio"
#      $env:ANTHROPIC_MODEL = "unsloth/qwen3.5-35b-a3b"

param(
    [int]$MaxExperiments = 24
)

$LogDir = "ralph_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  AUTORESEARCH - Otonom Arastirma Dongusu" -ForegroundColor Cyan
Write-Host "  Max deney: $MaxExperiments" -ForegroundColor Cyan
Write-Host "  Log dizini: $LogDir/" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

for ($i = 1; $i -le $MaxExperiments; $i++) {
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogFile = "$LogDir\experiment_${i}_${Timestamp}.log"

    Write-Host "[$i/$MaxExperiments] Deney baslatiliyor... ($(Get-Date))" -ForegroundColor Yellow

    # Mevcut durumu al
    $Branch = git branch --show-current 2>$null
    $LastCommit = git log --oneline -1 2>$null
    $ResultsTail = if (Test-Path "results.tsv") { Get-Content results.tsv -Tail 5 } else { "Henuz sonuc yok" }

    # CLAUDE.md icerigini oku
    $ClaudeMd = Get-Content "CLAUDE.md" -Raw

    # Prompt olustur
    $Prompt = @"
Sen bir arastirma ajanisin. Asagidaki talimatlari sirayla uygula.

$ClaudeMd

Su anki durum:
- Branch: $Branch
- Son commit: $LastCommit
- results.tsv son 5 satir:
$ResultsTail

Simdi ADIM 1'den basla. Bir sonraki deneyi tasarla, calistir, sonucu kaydet.
Sadece BIR deney yap, sonra dur.
"@

    # Claude Code'u calistir
    claude -p $Prompt --allowedTools Read,Write,Edit,Bash 2>&1 | Tee-Object -FilePath $LogFile

    Write-Host ""
    Write-Host "[$i/$MaxExperiments] Deney tamamlandi. Sonuc:" -ForegroundColor Green
    if (Test-Path "results.tsv") {
        Get-Content results.tsv -Tail 1
    } else {
        Write-Host "Sonuc kaydedilemedi" -ForegroundColor Red
    }
    Write-Host ""

    # Kisa bekle
    Start-Sleep -Seconds 5
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Tum deneyler tamamlandi!" -ForegroundColor Cyan
Write-Host "  Toplam: $MaxExperiments deney" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# En iyi sonucu goster
Write-Host ""
Write-Host "En iyi sonuc:" -ForegroundColor Green
if (Test-Path "results.tsv") {
    Get-Content results.tsv | Where-Object { $_ -match "`t" -and ($_ -split "`t")[1] -gt 0 } | Sort-Object { [double]($_ -split "`t")[1] } | Select-Object -First 1
}
