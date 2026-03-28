"""
autoresearch_loop.py — Otonom Araştırma Döngüsü
Yerel LLM sadece "ne değiştireceğine" karar verir.
Git, çalıştırma, sonuç okuma, keep/discard — Python yapar.

Kullanım:
    python autoresearch_loop.py                    # LLM modu (LM Studio gerekli)
    python autoresearch_loop.py --mode grid        # Grid search modu (LLM yok)
    python autoresearch_loop.py --mode grid --dry-run  # Sadece plan göster
"""

import os
import re
import sys
import json
import time
import argparse
import subprocess
import requests
from pathlib import Path
from datetime import datetime

# ─── Yapılandırma ──────────────────────────────────────────────────────────
PROJECT_DIR = Path(r"D:\autoresearch")
TRAIN_PY = PROJECT_DIR / "train.py"
RUN_LOG = PROJECT_DIR / "run.log"
RESULTS_TSV = PROJECT_DIR / "results.tsv"

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_STUDIO_MODEL = "liquid/lfm2-24b-a2b"  # LM Studio'da yüklü modeli yaz
LM_STUDIO_KEY = "lmstudio"

# Şu anki en iyi konfigürasyon (baseline)
CURRENT_BEST_CONFIG = {
    "DEPTH": 5,
    "HEAD_DIM": 128,
    "WINDOW_PATTERN": '"L"',
    "TOTAL_BATCH_SIZE": "2**16",
    "DEVICE_BATCH_SIZE": 4,
    "MATRIX_LR": 0.08,
    "EMBEDDING_LR": 0.6,
    "UNEMBEDDING_LR": 0.004,
    "SCALAR_LR": 0.5,
    "WEIGHT_DECAY": 0.2,
    "WARMUP_RATIO": 0.0,
    "WARMDOWN_RATIO": 0.5,
    "FINAL_LR_FRAC": 0.0,
    # softcap kaldırılmış (train.py'nin forward() metodunda)
}

# Grid search için denenecek kombinasyonlar (LLM olmadan)
GRID_EXPERIMENTS = [
    # (açıklama, {parametre: değer})
    ("DEPTH=4 tekrar", {"DEPTH": 4}),
    ("DEPTH=6 L pattern tekrar", {"DEPTH": 6}),
    ("WARMUP=0.05", {"WARMUP_RATIO": 0.05}),
    ("WARMUP=0.10", {"WARMUP_RATIO": 0.10}),
    ("WARMDOWN=0.3", {"WARMDOWN_RATIO": 0.3}),
    ("WARMDOWN=0.7", {"WARMDOWN_RATIO": 0.7}),
    ("MATRIX_LR=0.06", {"MATRIX_LR": 0.06}),
    ("MATRIX_LR=0.10", {"MATRIX_LR": 0.10}),
    ("EMBEDDING_LR=0.4", {"EMBEDDING_LR": 0.4}),
    ("EMBEDDING_LR=0.8", {"EMBEDDING_LR": 0.8}),
    ("EMBEDDING_LR=1.0", {"EMBEDDING_LR": 1.0}),
    ("WEIGHT_DECAY=0.0", {"WEIGHT_DECAY": 0.0}),
    ("WEIGHT_DECAY=0.3", {"WEIGHT_DECAY": 0.3}),
    ("WEIGHT_DECAY=0.4", {"WEIGHT_DECAY": 0.4}),
    ("BATCH=2^15", {"TOTAL_BATCH_SIZE": "2**15"}),
    ("BATCH=2^14", {"TOTAL_BATCH_SIZE": "2**14"}),
    ("HEAD_DIM=64", {"HEAD_DIM": 64}),
    ("DEVICE_BATCH=2", {"DEVICE_BATCH_SIZE": 2}),
    ("DEVICE_BATCH=6", {"DEVICE_BATCH_SIZE": 6}),
    ("FINAL_LR_FRAC=0.1", {"FINAL_LR_FRAC": 0.1}),
    ("FINAL_LR_FRAC=0.05", {"FINAL_LR_FRAC": 0.05}),
    ("ADAM_BETAS beta1=0.9", {"ADAM_BETAS": "(0.9, 0.95)"}),
    ("ADAM_BETAS beta1=0.7", {"ADAM_BETAS": "(0.7, 0.95)"}),
    ("SCALAR_LR=0.3", {"SCALAR_LR": 0.3}),
    ("SCALAR_LR=0.8", {"SCALAR_LR": 0.8}),
    ("WINDOW=LL", {"WINDOW_PATTERN": '"LL"'}),
    ("DEPTH=5 MATRIX_LR=0.06", {"DEPTH": 5, "MATRIX_LR": 0.06}),
    (
        "DEPTH=5 WARMUP=0.05 WARMDOWN=0.4",
        {"DEPTH": 5, "WARMUP_RATIO": 0.05, "WARMDOWN_RATIO": 0.4},
    ),
    ("DEPTH=5 EMBEDDING_LR=1.0", {"DEPTH": 5, "EMBEDDING_LR": 1.0}),
    ("DEPTH=4 MATRIX_LR=0.10", {"DEPTH": 4, "MATRIX_LR": 0.10}),
    ("DEPTH=6 MATRIX_LR=0.10", {"DEPTH": 6, "MATRIX_LR": 0.10}),
    ("DEPTH=5 HEAD_DIM=64", {"DEPTH": 5, "HEAD_DIM": 64}),
    ("EMBEDDING_LR=0.6 MATRIX_LR=0.06", {"EMBEDDING_LR": 0.6, "MATRIX_LR": 0.06}),
]


# ─── Renkli çıktı ──────────────────────────────────────────────────────────
class C:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def log(msg, color=C.RESET):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{C.GRAY}[{ts}]{C.RESET} {color}{msg}{C.RESET}", flush=True)


# ─── train.py düzenleme ────────────────────────────────────────────────────
def read_train_py():
    return TRAIN_PY.read_text(encoding="utf-8")


def write_train_py(content):
    TRAIN_PY.write_text(content, encoding="utf-8")


def apply_changes(changes: dict) -> bool:
    """
    train.py'deki hiperparametreleri değiştirir.
    changes = {"DEPTH": 5, "MATRIX_LR": 0.08, ...}
    """
    content = read_train_py()
    original = content

    for param, value in changes.items():
        # Değeri Python string'e çevir
        if (
            isinstance(value, str)
            and not value.startswith('"')
            and not value.startswith("(")
            and not value.startswith("2**")
        ):
            val_str = f'"{value}"'
        elif isinstance(value, float):
            val_str = str(value)
        elif isinstance(value, int):
            val_str = str(value)
        else:
            val_str = str(value)

        # Satırı bul ve değiştir
        # Örnek: DEPTH = 6  veya  DEPTH= 6  veya  DEPTH=6
        pattern = rf"^({re.escape(param)}\s*=\s*)(.+)$"
        new_line = rf"\g<1>{val_str}"
        new_content = re.sub(pattern, new_line, content, flags=re.MULTILINE)

        if new_content == content:
            log(f"UYARI: '{param}' bulunamadı, atlanıyor", C.YELLOW)
        else:
            log(f"  {param} = {val_str}", C.CYAN)
            content = new_content

    if content == original:
        log("Hiçbir değişiklik yapılamadı!", C.RED)
        return False

    write_train_py(content)
    return True


def restore_config(config: dict):
    """Konfigürasyonu bilinen en iyi değerlere sıfırla."""
    apply_changes(config)


# ─── Git işlemleri ─────────────────────────────────────────────────────────
def git(cmd: str, check=True) -> str:
    result = subprocess.run(
        f"git -C {PROJECT_DIR} {cmd}", shell=True, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {cmd} başarısız: {result.stderr}")
    return result.stdout.strip()


def git_commit(message: str) -> str:
    git("add train.py")
    git(f'commit -m "deney: {message}"')
    return git("rev-parse --short HEAD")


def git_reset():
    git("reset --hard HEAD~1")


def git_current_hash() -> str:
    return git("rev-parse --short HEAD", check=False)


# ─── Eğitimi çalıştır ──────────────────────────────────────────────────────
def run_training(dry_run=False) -> dict:
    """
    uv run train.py çalıştırır, sonuçları parse eder.
    Döndürür: {"val_bpb": float, "num_steps": int, "seconds": float, "success": bool}
    """
    if dry_run:
        log("[DRY RUN] Eğitim atlanıyor", C.YELLOW)
        return {"val_bpb": 99.0, "num_steps": 0, "seconds": 0, "success": True}

    log("Eğitim başlıyor... (uzun sürebilir)", C.YELLOW)
    start = time.time()

    with open(RUN_LOG, "w") as f:
        proc = subprocess.run(
            "uv run train.py",
            shell=True,
            cwd=PROJECT_DIR,
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=7200,  # max 2 saat
        )

    elapsed = time.time() - start

    # Sonuçları parse et
    log_text = RUN_LOG.read_text(encoding="utf-8", errors="ignore")

    def parse_val(pattern):
        m = re.search(pattern, log_text, re.MULTILINE)
        return float(m.group(1)) if m else None

    val_bpb = parse_val(r"^val_bpb:\s+([\d.]+)")
    num_steps = parse_val(r"^num_steps:\s+(\d+)")
    seconds = parse_val(r"^training_seconds:\s+([\d.]+)")

    success = val_bpb is not None
    if not success:
        # Hata mesajını göster
        last_lines = "\n".join(log_text.strip().split("\n")[-20:])
        log(f"Program çöktü! Son satırlar:\n{last_lines}", C.RED)

    return {
        "val_bpb": val_bpb or 99.0,
        "num_steps": int(num_steps or 0),
        "seconds": seconds or elapsed,
        "success": success,
    }


# ─── Sonuçları kaydet ──────────────────────────────────────────────────────
def init_results_tsv():
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(
            "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n", encoding="utf-8"
        )
        log("results.tsv oluşturuldu", C.GREEN)


def save_result(commit, val_bpb, status, description, memory_gb=0.0):
    line = f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t{status}\t{description}\n"
    with open(RESULTS_TSV, "a", encoding="utf-8") as f:
        f.write(line)
    log(f"Kaydedildi: {line.strip()}", C.GRAY)


def get_best_val_bpb() -> float:
    if not RESULTS_TSV.exists():
        return 999.0
    best = 999.0
    for line in RESULTS_TSV.read_text(encoding="utf-8").splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) >= 4 and cols[3] == "keep":
            try:
                best = min(best, float(cols[1]))
            except ValueError:
                pass
    return best


def validate_changes(changes: dict) -> bool:
    """Değişikliklerin mantıklı olup olmadığını kontrol et."""
    # HEAD_DIM + DEPTH uyumsuzluğu kontrolü
    depth = changes.get("DEPTH", CURRENT_BEST_CONFIG.get("DEPTH", 5))
    head_dim = changes.get("HEAD_DIM", CURRENT_BEST_CONFIG.get("HEAD_DIM", 128))
    aspect_ratio = 64
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    if num_heads < 2:
        log(
            f"GEÇERSİZ: DEPTH={depth} + HEAD_DIM={head_dim} → num_heads={num_heads} (en az 2 gerekli)",
            C.RED,
        )
        return False

    # DEVICE_BATCH_SIZE bölünebilirlik kontrolü
    batch_size = changes.get(
        "DEVICE_BATCH_SIZE", CURRENT_BEST_CONFIG.get("DEVICE_BATCH_SIZE", 4)
    )
    total_batch_str = changes.get(
        "TOTAL_BATCH_SIZE", CURRENT_BEST_CONFIG.get("TOTAL_BATCH_SIZE", "2**16")
    )
    total_batch = eval(str(total_batch_str))
    tokens_per_step = batch_size * 512  # MAX_SEQ_LEN
    if total_batch % tokens_per_step != 0:
        log(
            f"GEÇERSİZ: TOTAL_BATCH={total_batch} % (BATCH={batch_size} * 512) = {total_batch % tokens_per_step} != 0",
            C.RED,
        )
        return False

    return True


def get_tried_experiments() -> list:
    """Daha önce denenmiş deney açıklamalarını döndür."""
    if not RESULTS_TSV.exists():
        return []
    tried = []
    for line in RESULTS_TSV.read_text(encoding="utf-8").splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) >= 5:
            tried.append(cols[4])
    return tried


# ─── LM Studio — LLM kararı ─────────────────────────────────────────────────
def ask_llm_for_change(best_val: float, tried: list) -> dict | None:
    """
    LLM'e sadece şunu sorar:
    'Hangi parametreyi nasıl değiştirmeliyim?'
    Cevap: {"param": "DEPTH", "value": 4, "reason": "..."}
    """
    tried_str = "\n".join(f"- {t}" for t in tried[-10:]) if tried else "(henüz yok)"

    system = """Sen bir ML araştırma uzmanısın.
Görevin: val_bpb değerini düşürmek için train.py'de hangi TEK değişikliğin yapılacağını söylemek.
SADECE JSON formatında cevap ver. Başka hiçbir şey yazma.
Format: {"param": "PARAMETRE_ADI", "value": DEĞERİ, "reason": "kısa açıklama"}

Geçerli parametreler ve mevcut değerleri:
- DEPTH = 5
- HEAD_DIM = 128
- WINDOW_PATTERN = "L"
- TOTAL_BATCH_SIZE = 2**16
- DEVICE_BATCH_SIZE = 4
- MATRIX_LR = 0.08
- EMBEDDING_LR = 0.6
- WEIGHT_DECAY = 0.2
- WARMUP_RATIO = 0.0
- WARMDOWN_RATIO = 0.5
- FINAL_LR_FRAC = 0.0

Kural: Daha önce denenenlerden FARKLI bir şey seç."""

    user = f"""En iyi val_bpb: {best_val:.6f}

Son denenenler:
{tried_str}

Hangi değişikliği önerirsin? Sadece JSON döndür."""

    try:
        resp = requests.post(
            LM_STUDIO_URL,
            headers={"Authorization": f"Bearer {LM_STUDIO_KEY}"},
            json={
                "model": LM_STUDIO_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 200,
                "temperature": 0.7,
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # JSON parse et — LLM bazen ```json ... ``` ile sarar
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            return data
    except Exception as e:
        log(f"LLM hatası: {e}", C.RED)
    return None


# ─── Tek deney çalıştır ────────────────────────────────────────────────────
def run_one_experiment(description: str, changes: dict, dry_run=False) -> str:
    """
    Bir deney çalıştırır.
    Döndürür: "keep" | "discard" | "crash"
    """
    best_before = get_best_val_bpb()
    log(f"\n{'═' * 55}", C.CYAN)
    log(f"Deney: {description}", C.BOLD + C.CYAN)
    log(f"Değişiklikler: {changes}", C.CYAN)
    log(f"Hedef: {best_before:.6f}'nin altına in", C.CYAN)

    # 0. Validasyon
    if not validate_changes(changes):
        log("Değişiklik geçersiz — CRASH", C.RED)
        save_result("none", 0.0, "crash", f"{description} (geçersiz config)")
        return "crash"

    # 1. Değişiklikleri uygula
    if not apply_changes(changes):
        log("Değişiklik uygulanamadı — CRASH", C.RED)
        save_result("none", 0.0, "crash", description)
        return "crash"

    # 2. Git commit
    try:
        commit = git_commit(description)
        log(f"Commit: {commit}", C.GRAY)
    except Exception as e:
        log(f"Git commit hatası: {e} — CRASH", C.RED)
        restore_config(CURRENT_BEST_CONFIG)
        save_result("none", 0.0, "crash", f"{description} (git hatası)")
        return "crash"

    # 3. Eğitimi çalıştır
    result = run_training(dry_run=dry_run)

    if not result["success"]:
        log("Eğitim çöktü — CRASH", C.RED)
        git_reset()
        restore_config(CURRENT_BEST_CONFIG)
        commit_after = git_current_hash()
        save_result(commit_after, 0.0, "crash", description)
        return "crash"

    val_bpb = result["val_bpb"]
    num_steps = result["num_steps"]
    seconds = result["seconds"]

    log(f"val_bpb: {val_bpb:.6f} | steps: {num_steps} | süre: {seconds:.0f}s", C.WHITE)

    # 4. Karar ver
    if val_bpb < best_before - 0.001:  # anlamlı iyileşme
        log(
            f"✅ KEEP! {best_before:.6f} → {val_bpb:.6f} ({best_before - val_bpb:.4f} iyileşme)",
            C.GREEN,
        )
        save_result(commit, val_bpb, "keep", description)
        # En iyi config'i güncelle
        CURRENT_BEST_CONFIG.update(changes)
        log(f"Config güncellendi: {CURRENT_BEST_CONFIG}", C.GRAY)
        return "keep"
    else:
        log(f"❌ DISCARD. {val_bpb:.6f} ≥ {best_before:.6f}", C.YELLOW)
        git_reset()
        restore_config(CURRENT_BEST_CONFIG)
        save_result(commit, val_bpb, "discard", description)
        return "discard"


# ─── Grid search modu ──────────────────────────────────────────────────────
def run_grid_mode(max_experiments=None, dry_run=False):
    """LLM olmadan, sabit liste üzerinden dene."""
    tried = get_tried_experiments()
    experiments = [
        (desc, changes)
        for desc, changes in GRID_EXPERIMENTS
        if not any(desc in t for t in tried)  # denenenleri atla
    ]

    if not experiments:
        log("Tüm grid deneyleri tamamlandı!", C.GREEN)
        return

    if max_experiments:
        experiments = experiments[:max_experiments]

    log(f"Grid modu: {len(experiments)} deney planlandı", C.CYAN)

    consecutive_discard = 0
    for i, (desc, changes) in enumerate(experiments, 1):
        log(f"\n{'─' * 55}")
        log(f"Deney {i}/{len(experiments)}: {desc}", C.BOLD)

        status = run_one_experiment(desc, changes, dry_run=dry_run)

        if status in ("discard", "crash"):
            consecutive_discard += 1
        else:
            consecutive_discard = 0

        if consecutive_discard >= 10:
            log("10 ardışık başarısızlık — duruyorum", C.RED)
            break

        # Deney arası kısa bekleme
        if i < len(experiments):
            time.sleep(3)


# ─── LLM modu ──────────────────────────────────────────────────────────────
def run_llm_mode(max_experiments=20, dry_run=False):
    """LLM kararı + Python mekanik işler."""
    log("LLM modu başlatılıyor...", C.CYAN)

    # LM Studio bağlantısını test et
    try:
        resp = requests.get(
            "http://localhost:1234/v1/models",
            headers={"Authorization": f"Bearer {LM_STUDIO_KEY}"},
            timeout=5,
        )
        models = [m["id"] for m in resp.json().get("data", [])]
        log(f"LM Studio bağlantısı OK. Modeller: {models}", C.GREEN)
    except Exception as e:
        log(f"LM Studio bağlanamadı: {e}", C.RED)
        log("Grid moduna geçiliyor...", C.YELLOW)
        run_grid_mode(max_experiments, dry_run)
        return

    consecutive_fail = 0
    for i in range(1, max_experiments + 1):
        best = get_best_val_bpb()
        tried = get_tried_experiments()

        log(f"\n{'═' * 55}")
        log(f"LLM Deney {i}/{max_experiments} | En iyi: {best:.6f}", C.BOLD)

        # LLM'den değişiklik iste
        log("LLM'e danışılıyor...", C.GRAY)
        suggestion = ask_llm_for_change(best, tried)

        if not suggestion:
            log("LLM cevap vermedi — grid'den seç", C.YELLOW)
            tried_set = set(tried)
            for desc, changes in GRID_EXPERIMENTS:
                if desc not in tried_set:
                    status = run_one_experiment(desc, changes, dry_run)
                    break
            else:
                log("Grid de bitti!", C.RED)
                break
            consecutive_fail += 1
        else:
            param = suggestion.get("param", "")
            value = suggestion.get("value", "")
            reason = suggestion.get("reason", "LLM önerisi")
            description = f"{param}={value} ({reason})"
            log(f"LLM önerisi: {description}", C.CYAN)

            status = run_one_experiment(description, {param: value}, dry_run=dry_run)
            if status == "keep":
                consecutive_fail = 0
            else:
                consecutive_fail += 1

        if consecutive_fail >= 15:
            log("15 ardışık başarısızlık — duruyorum", C.RED)
            break

        time.sleep(2)


# ─── Özet raporu ───────────────────────────────────────────────────────────
def print_summary():
    print(f"\n{'═' * 55}")
    print(f"{C.CYAN}{C.BOLD}ÖZET{C.RESET}")

    if not RESULTS_TSV.exists():
        print("Sonuç yok.")
        return

    lines = RESULTS_TSV.read_text(encoding="utf-8").splitlines()[1:]
    keep = [l for l in lines if "\tkeep\t" in l]
    discard = [l for l in lines if "\tdiscard\t" in l]
    crash = [l for l in lines if "\tcrash\t" in l]

    print(f"Toplam deney: {len(lines)}")
    print(
        f"{C.GREEN}KEEP: {len(keep)}{C.RESET}  {C.YELLOW}DISCARD: {len(discard)}{C.RESET}  {C.RED}CRASH: {len(crash)}{C.RESET}"
    )

    if keep:
        best_line = sorted(keep, key=lambda l: float(l.split("\t")[1]))[0]
        cols = best_line.split("\t")
        print(f"\n{C.GREEN}En iyi sonuç:{C.RESET}")
        print(f"  val_bpb    : {cols[1]}")
        print(f"  commit     : {cols[0]}")
        print(f"  açıklama   : {cols[4]}")

    print(f"{'═' * 55}\n")


# ─── Ana program ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Otonom araştırma döngüsü")
    parser.add_argument(
        "--mode",
        choices=["llm", "grid"],
        default="grid",
        help="llm: LM Studio kullan | grid: sabit liste (varsayılan)",
    )
    parser.add_argument("--max", type=int, default=30, help="Maksimum deney sayısı")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Gerçek eğitim çalıştırma, sadece plan göster",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_DIR)
    init_results_tsv()

    print(f"\n{C.CYAN}{C.BOLD}AUTORESEARCH — Otonom Döngü{C.RESET}")
    print(f"Mod: {args.mode} | Max deney: {args.max} | Dry run: {args.dry_run}")
    print(f"En iyi val_bpb: {get_best_val_bpb():.6f}")
    print()

    try:
        if args.mode == "llm":
            run_llm_mode(args.max, args.dry_run)
        else:
            run_grid_mode(args.max, args.dry_run)
    except KeyboardInterrupt:
        log("\nKullanıcı durdurdu (Ctrl+C)", C.YELLOW)
    finally:
        print_summary()
