#!/bin/bash
# ralph.sh - Otonom Araştırma Döngüsü (LM Studio + Claude Code)
#
# KULLANIM:
#   chmod +x ralph.sh
#   ./ralph.sh [max_deney_sayisi]
#
# ÖN KOŞULLAR:
#   1. LM Studio açık ve model yüklü (port 1234)
#   2. Claude Code kurulu
#   3. Aşağıdaki environment variable'lar ayarlı:
#      export ANTHROPIC_BASE_URL=http://localhost:1234
#      export ANTHROPIC_AUTH_TOKEN=lmstudio
#      export ANTHROPIC_MODEL="unsloth/qwen3.5-35b-a3b"

MAX_EXPERIMENTS=${1:-24}
EXPERIMENT=0
LOG_DIR="ralph_logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  AUTORESEARCH - Otonom Araştırma Döngüsü"
echo "  Max deney: $MAX_EXPERIMENTS"
echo "  Model: $ANTHROPIC_MODEL"
echo "  Log dizini: $LOG_DIR/"
echo "=========================================="
echo ""

while [ $EXPERIMENT -lt $MAX_EXPERIMENTS ]; do
    EXPERIMENT=$((EXPERIMENT + 1))
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/experiment_${EXPERIMENT}_${TIMESTAMP}.log"

    echo "[$EXPERIMENT/$MAX_EXPERIMENTS] Deney başlatılıyor... ($(date))"

    # CLAUDE.md içeriğini prompt'a ekle
    PROMPT="Sen bir araştırma ajanısın. Aşağıdaki talimatları sırayla uygula.

$(cat CLAUDE.md)

Şu anki durum:
- Branch: $(git branch --show-current)
- Son commit: $(git log --oneline -1)
- results.tsv son 5 satır:
$(tail -5 results.tsv 2>/dev/null || echo 'Henüz sonuç yok')

Şimdi ADIM 1'den başla. Bir sonraki deneyi tasarla, çalıştır, sonucu kaydet.
Sadece BİR deney yap, sonra dur."

    # Claude Code'u çalıştır
    claude -p "$PROMPT" \
        --allowedTools Read,Write,Edit,Bash \
        2>&1 | tee "$LOG_FILE"

    echo ""
    echo "[$EXPERIMENT/$MAX_EXPERIMENTS] Deney tamamlandı. Sonuç:"
    tail -1 results.tsv 2>/dev/null || echo "Sonuç kaydedilemedi"
    echo ""

    # Kısa bekle (sistem nefes alsın)
    sleep 5
done

echo "=========================================="
echo "  Tüm deneyler tamamlandı!"
echo "  Toplam: $EXPERIMENT deney"
echo "  Sonuçlar: results.tsv"
echo "=========================================="

# En iyi sonucu göster
echo ""
echo "En iyi sonuç:"
awk -F'\t' 'NR>1 && $2>0 {print $0}' results.tsv | sort -t$'\t' -k2 -n | head -1
