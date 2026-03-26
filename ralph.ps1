# ralph.ps1 - Otonom Arastirma Dongusu (LM Studio + Claude Code)
# Kullanim: .\ralph.ps1 [-MaxExperiments 24] [-DryRun]
#
# ON KOSULLAR:
#   1. Claude Code native installer ile kurulmus olmali
#   2. LM Studio acik, model yuklu, server baslatilmis (port 1234)
#   3. git init + ilk commit yapilmis
#   4. results.tsv dosyasi olusturulmus

param(
    [int]$MaxExperiments = 24,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

# ─── Ayarlar ────────────────────────────────────────────────────────────────
$ClaudeExe    = "C:\Users\aysbe\.local\bin\claude.exe"
$ProjectDir   = "D:\autoresearch"
$LogDir       = "$ProjectDir\ralph_logs"
$ResultsFile  = "$ProjectDir\results.tsv"
$ClaudeMd     = "$ProjectDir\CLAUDE.md"

# LM Studio baglantisi
$LMStudioUrl  = "http://localhost:1234"
$LMStudioAuth = "lmstudio"

# LM Studio'da yuklu olan model adi (LM Studio'daki tam adini yaz)
$ModelName    = "unsloth/qwen3.5-35b-a3b"

# claude -p icin tool izinleri
$AllowedTools = "Bash,Read,Write,Edit"

# ─── Renkli log fonksiyonu ──────────────────────────────────────────────────
function Log {
    param([string]$Msg, [string]$Color = "White")
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[$ts] $Msg" -ForegroundColor $Color
}

# ─── LM Studio kontrolu ─────────────────────────────────────────────────────
function Test-LMStudio {
    try {
        $resp = Invoke-RestMethod -Uri "$LMStudioUrl/v1/models" `
            -Headers @{ Authorization = "Bearer $LMStudioAuth" } `
            -TimeoutSec 5
        $models = $resp.data | ForEach-Object { $_.id }
        Log "LM Studio baglantisi OK. Modeller: $($models -join ', ')" "Green"
        return $true
    } catch {
        Log "LM Studio'ya baglanamadi ($LMStudioUrl). Sunucu acik mi?" "Red"
        return $false
    }
}

# ─── Claude Code kontrolu ───────────────────────────────────────────────────
function Test-ClaudeCode {
    if (Test-Path $ClaudeExe) {
        Log "Claude Code bulundu: $ClaudeExe" "Green"
        return $true
    }
    # PATH'te ara
    $found = Get-Command "claude" -ErrorAction SilentlyContinue
    if ($found) {
        $script:ClaudeExe = $found.Source
        Log "Claude Code PATH'te bulundu: $($found.Source)" "Green"
        return $true
    }
    Log "claude.exe bulunamadı. Konum: $ClaudeExe" "Red"
    return $false
}

# ─── Mevcut durumu topla ────────────────────────────────────────────────────
function Get-CurrentState {
    $branch    = git -C $ProjectDir branch --show-current 2>$null
    $lastLog   = git -C $ProjectDir log --oneline -3 2>$null
    $results   = if (Test-Path $ResultsFile) { Get-Content $ResultsFile -Tail 10 } else { "Henuz sonuc yok" }
    $bestLine  = if (Test-Path $ResultsFile) {
        Get-Content $ResultsFile |
            Where-Object { $_ -match "`t" } |
            Where-Object { ($_ -split "`t")[3] -eq "keep" } |
            Sort-Object { [double]($_ -split "`t")[1] } |
            Select-Object -First 1
    } else { "" }
    $bestVal   = if ($bestLine) { ($bestLine -split "`t")[1] } else { "henuz_yok" }

    return @{
        Branch   = $branch
        LastLog  = ($lastLog -join "`n")
        Results  = ($results -join "`n")
        BestVal  = $bestVal
    }
}

# ─── Tek deney calistir ─────────────────────────────────────────────────────
function Run-Experiment {
    param([int]$ExpNum, [string]$LogFile)

    $state = Get-CurrentState

    # CLAUDE.md icerigini oku
    $instructions = Get-Content $ClaudeMd -Raw -ErrorAction Stop

    # Ajan prompt'u
    $prompt = @"
Asagidaki talimatlari uygula. Yalnizca BIR deney yap, sonra dur.

=== TALIMATLAR ===
$instructions

=== SU ANKI DURUM ===
Proje: $ProjectDir
Branch: $($state.Branch)
Son commitler:
$($state.LastLog)

results.tsv son satirlar:
$($state.Results)

En iyi val_bpb simdi: $($state.BestVal)

=== GOREV ===
1. results.tsv'ye bak, henuz denenmemis bir degisikligi sec
2. train.py'yi duzenle (SADECE BIR degisiklik)
3. git add train.py && git commit -m "deney: [aciklama]"
4. uv run train.py > run.log 2>&1   (bu ~1 saat surer, bekle)
5. grep "^val_bpb:" run.log ile sonucu oku
6. Karar ver: yeni val_bpb < $($state.BestVal) ise KEEP, degilse DISCARD
7. results.tsv'ye sonucu yaz (TAB ile ayrilmis: commit val mem status aciklama)
8. DISCARD/CRASH ise: git reset --hard HEAD~1

Simdi basla.
"@

    Log "Claude Code baslatiliyor (deney $ExpNum)..." "Yellow"

    if ($DryRun) {
        Log "[DRY RUN] Prompt hazir, gercek calistirma atlaniyor." "Cyan"
        Log "[DRY RUN] Prompt uzunlugu: $($prompt.Length) karakter" "Gray"
        return $true
    }

    # Ortam degiskenlerini ayarla (LM Studio icin)
    $env:ANTHROPIC_BASE_URL    = $LMStudioUrl
    $env:ANTHROPIC_AUTH_TOKEN  = $LMStudioAuth
    # Tum model isimlerini lokal modele yonlendir
    $env:ANTHROPIC_DEFAULT_OPUS_MODEL   = $ModelName
    $env:ANTHROPIC_DEFAULT_SONNET_MODEL = $ModelName
    $env:ANTHROPIC_DEFAULT_HAIKU_MODEL  = $ModelName
    $env:ANTHROPIC_MODEL        = $ModelName
    $env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
    $env:DISABLE_TELEMETRY     = "1"
    $env:DISABLE_ERROR_REPORTING = "1"

    try {
        # claude -p = headless/non-interactive mod
        # --dangerously-skip-permissions = her onay icin duraklamaz
        # --allowedTools = sadece bu araclara izin ver
        # --max-turns = sonsuz donguye girmesin
        $claudeArgs = @(
            "-p", $prompt,
            "--allowedTools", $AllowedTools,
            "--dangerously-skip-permissions",
            "--max-turns", "50",
            "--output-format", "text"
        )

        # Calistir ve log'a yaz
        Set-Location $ProjectDir
        & $ClaudeExe @claudeArgs 2>&1 | Tee-Object -FilePath $LogFile

        return ($LASTEXITCODE -eq 0)
    } catch {
        Log "Claude Code hatasi: $_" "Red"
        return $false
    }
}

# ─── Ana dongu ──────────────────────────────────────────────────────────────

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  RALPH - Otonom Arastirma Dongusu"        -ForegroundColor Cyan
Write-Host "  Proje : $ProjectDir"                     -ForegroundColor Cyan
Write-Host "  Model : $ModelName"                      -ForegroundColor Cyan
Write-Host "  Max   : $MaxExperiments deney"           -ForegroundColor Cyan
if ($DryRun) {
Write-Host "  MOD   : DRY RUN (gercek calisma yok)"   -ForegroundColor Yellow }
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# On kontroller
if (-not (Test-ClaudeCode)) { exit 1 }
if (-not (Test-LMStudio))   { exit 1 }
if (-not (Test-Path $ProjectDir)) {
    Log "Proje klasoru bulunamadi: $ProjectDir" "Red"; exit 1
}
if (-not (Test-Path $ClaudeMd)) {
    Log "CLAUDE.md bulunamadi: $ClaudeMd" "Red"; exit 1
}
if (-not (Test-Path $ResultsFile)) {
    Log "results.tsv olusturuluyor..." "Yellow"
    "commit`tval_bpb`tmemory_gb`tstatus`tdescription" | Out-File $ResultsFile -Encoding utf8
}

Set-Location $ProjectDir

# ─── Deney dongusu ──────────────────────────────────────────────────────────
$successCount = 0
$failCount    = 0

for ($i = 1; $i -le $MaxExperiments; $i++) {
    $ts      = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile = "$LogDir\exp_${i}_${ts}.log"

    Write-Host ""
    Log "--- Deney $i / $MaxExperiments ---" "Cyan"

    $state = Get-CurrentState
    Log "En iyi val_bpb: $($state.BestVal)" "White"
    Log "Branch: $($state.Branch)" "White"

    $ok = Run-Experiment -ExpNum $i -LogFile $logFile

    # Sonucu goster
    Write-Host ""
    if (Test-Path $ResultsFile) {
        $lastResult = Get-Content $ResultsFile -Tail 1
        if ($lastResult -and $lastResult -ne "commit`tval_bpb`tmemory_gb`tstatus`tdescription") {
            $cols   = $lastResult -split "`t"
            $status = if ($cols.Count -ge 4) { $cols[3] } else { "?" }
            $valBpb = if ($cols.Count -ge 2) { $cols[1] } else { "?" }

            switch ($status) {
                "keep"    { Log "SONUC: KEEP - val_bpb=$valBpb" "Green"; $successCount++ }
                "discard" { Log "SONUC: DISCARD - val_bpb=$valBpb" "Yellow"; $failCount++ }
                "crash"   { Log "SONUC: CRASH" "Red"; $failCount++ }
                default   { Log "SONUC: $lastResult" "White" }
            }
        }
    }

    if (-not $ok) {
        Log "Claude Code basarisiz oldu. Devam ediliyor..." "Red"
        $failCount++
    }

    # Ardisik basarisizlik kontrolu
    if ($failCount -ge 15) {
        Log "15 ardisik basarisizlik! Dongu durduruluyor." "Red"
        break
    }

    # Deneylerin arasinda kisa bekleme
    if ($i -lt $MaxExperiments) {
        Log "Sonraki deney icin 10 saniye bekleniyor..." "Gray"
        Start-Sleep -Seconds 10
    }
}

# ─── Ozet ───────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  TAMAMLANDI" -ForegroundColor Green
Write-Host "  Toplam deney : $MaxExperiments" -ForegroundColor White
Write-Host "  Basarili KEEP: $successCount" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "En iyi sonuclar:" -ForegroundColor Green
if (Test-Path $ResultsFile) {
    Get-Content $ResultsFile |
        Where-Object { $_ -match "`t" -and ($_ -split "`t")[3] -eq "keep" } |
        Sort-Object { [double]($_ -split "`t")[1] } |
        Select-Object -First 5 |
        ForEach-Object { Write-Host "  $_" -ForegroundColor Green }
}
