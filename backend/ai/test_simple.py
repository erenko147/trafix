"""
TraFix Simple — Model Test Scripti
====================================
Kullanim:
  python backend/ai/test_simple.py
  python backend/ai/test_simple.py --weights path/to/model.pth
"""

import os
import sys
import math
import argparse
import traceback
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

try:
    from backend.ai.trafix_simple import (
        SimplePPOAgent, parse_sumo_observations,
        compute_reward, compute_gae, RewardWeights, NUM_NODE_FEATURES,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from trafix_simple import (
        SimplePPOAgent, parse_sumo_observations,
        compute_reward, compute_gae, RewardWeights, NUM_NODE_FEATURES,
    )


# ══════════════════════════════════════════════════
#  Sabitler
# ══════════════════════════════════════════════════

HIDDEN_DIM  = 64
NUM_ACTIONS = 4
NUM_NODES   = 5

PHASE_NAMES  = {0: "Kuzey-Güney", 1: "Güney-Kuzey", 2: "Doğu-Batı", 3: "Batı-Doğu"}
KAVSAK_NAMES = {i: f"K{i+1}" for i in range(NUM_NODES)}


# ══════════════════════════════════════════════════
#  Test Senaryolari
# ══════════════════════════════════════════════════

SCENARIOS: Dict[str, List[Dict]] = {
    "Normal Trafik": [
        {"intersection_id": 0, "north_count": 4,  "south_count": 3,  "east_count": 12, "west_count": 7,  "queue_length": 15.2, "current_phase": 0, "phase_duration": 12.0},
        {"intersection_id": 1, "north_count": 2,  "south_count": 1,  "east_count": 5,  "west_count": 3,  "queue_length": 0.0,  "current_phase": 2, "phase_duration": 4.0},
        {"intersection_id": 2, "north_count": 10, "south_count": 8,  "east_count": 3,  "west_count": 6,  "queue_length": 45.0, "current_phase": 1, "phase_duration": 22.0},
        {"intersection_id": 3, "north_count": 0,  "south_count": 2,  "east_count": 7,  "west_count": 1,  "queue_length": 2.1,  "current_phase": 3, "phase_duration": 8.0},
        {"intersection_id": 4, "north_count": 5,  "south_count": 4,  "east_count": 2,  "west_count": 8,  "queue_length": 8.5,  "current_phase": 0, "phase_duration": 10.0},
    ],
    "Yogun Trafik (K3 darboğaz)": [
        {"intersection_id": 0, "north_count": 8,  "south_count": 6,  "east_count": 15, "west_count": 10, "queue_length": 35.0,  "current_phase": 0, "phase_duration": 15.0},
        {"intersection_id": 1, "north_count": 5,  "south_count": 3,  "east_count": 8,  "west_count": 6,  "queue_length": 12.0,  "current_phase": 2, "phase_duration": 8.0},
        {"intersection_id": 2, "north_count": 35, "south_count": 28, "east_count": 15, "west_count": 20, "queue_length": 180.0, "current_phase": 1, "phase_duration": 40.0},
        {"intersection_id": 3, "north_count": 3,  "south_count": 5,  "east_count": 10, "west_count": 2,  "queue_length": 8.0,   "current_phase": 3, "phase_duration": 10.0},
        {"intersection_id": 4, "north_count": 10, "south_count": 8,  "east_count": 5,  "west_count": 12, "queue_length": 25.0,  "current_phase": 0, "phase_duration": 12.0},
    ],
    "Gece Trafigi": [
        {"intersection_id": 0, "north_count": 1, "south_count": 0, "east_count": 2, "west_count": 1, "queue_length": 1.0, "current_phase": 0, "phase_duration": 30.0},
        {"intersection_id": 1, "north_count": 0, "south_count": 1, "east_count": 0, "west_count": 0, "queue_length": 0.0, "current_phase": 2, "phase_duration": 30.0},
        {"intersection_id": 2, "north_count": 2, "south_count": 1, "east_count": 1, "west_count": 0, "queue_length": 2.0, "current_phase": 1, "phase_duration": 30.0},
        {"intersection_id": 3, "north_count": 0, "south_count": 0, "east_count": 1, "west_count": 0, "queue_length": 0.5, "current_phase": 3, "phase_duration": 30.0},
        {"intersection_id": 4, "north_count": 1, "south_count": 0, "east_count": 0, "west_count": 2, "queue_length": 1.5, "current_phase": 0, "phase_duration": 30.0},
    ],
    "Tek Yon Baskin (Dogu-Bati)": [
        {"intersection_id": 0, "north_count": 1, "south_count": 1, "east_count": 20, "west_count": 18, "queue_length": 55.0, "current_phase": 0, "phase_duration": 35.0},
        {"intersection_id": 1, "north_count": 0, "south_count": 1, "east_count": 22, "west_count": 15, "queue_length": 60.0, "current_phase": 0, "phase_duration": 35.0},
        {"intersection_id": 2, "north_count": 2, "south_count": 0, "east_count": 25, "west_count": 20, "queue_length": 70.0, "current_phase": 0, "phase_duration": 35.0},
        {"intersection_id": 3, "north_count": 1, "south_count": 0, "east_count": 18, "west_count": 16, "queue_length": 48.0, "current_phase": 0, "phase_duration": 35.0},
        {"intersection_id": 4, "north_count": 0, "south_count": 1, "east_count": 19, "west_count": 17, "queue_length": 52.0, "current_phase": 0, "phase_duration": 35.0},
    ],
}


# ══════════════════════════════════════════════════
#  Yardimcilar
# ══════════════════════════════════════════════════

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []

    def ok(self, msg: str):
        self.passed += 1
        print(f"  ✓ {msg}")

    def fail(self, msg: str, detail: str = ""):
        self.failed += 1
        self.errors.append(msg)
        print(f"  ✗ {msg}")
        if detail:
            print(f"    → {detail}")

    def summary(self) -> bool:
        total = self.passed + self.failed
        print(f"\n{'═' * 60}")
        print(f"  SONUÇ: {self.passed}/{total} test basarili", end="")
        if self.failed > 0:
            print(f" — {self.failed} BASARISIZ")
            for e in self.errors:
                print(f"    ✗ {e}")
        else:
            print(" — HEPSI GECTI")
        print(f"{'═' * 60}")
        return self.failed == 0


def find_weights() -> Optional[str]:
    candidates = [
        "training_outputs_simple/best_model.pth",
        "training_outputs_simple/final_model.pth",
        "coordinated_agent_weights_simple.pth",
    ]
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    for c in candidates:
        p = os.path.join(root, c)
        if os.path.exists(p):
            return p
    return None


def load_agent(weights_path: Optional[str]) -> SimplePPOAgent:
    agent = SimplePPOAgent(
        num_node_features=NUM_NODE_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_actions=NUM_ACTIONS,
        entropy_coef=0.05,
        value_coef=0.25,
    )
    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        try:
            agent.load_state_dict(state)
            print(f"  Agirliklar yuklendi: {weights_path}")
        except RuntimeError as e:
            print(f"  ⚠ Agirlik uyumsuzlugu — rastgele model ile devam ediliyor")
            print(f"    {str(e)[:120]}")
    else:
        print("  Agirlik dosyasi bulunamadi — rastgele baslatilmis model test ediliyor")
    agent.eval()
    return agent


# ══════════════════════════════════════════════════
#  TEST 1: Model Bilgisi
# ══════════════════════════════════════════════════

def test_model_info(agent: SimplePPOAgent, results: TestResult):
    print(f"\n{'═' * 60}")
    print("  TEST 1: Model Bilgisi")
    print(f"{'═' * 60}")

    total     = sum(p.numel() for p in agent.parameters())
    trainable = sum(p.numel() for p in agent.parameters() if p.requires_grad)

    print(f"\n  Toplam parametre : {total:,}")
    print(f"  Egit. parametre  : {trainable:,}")
    print(f"  Giris ozelligi   : {NUM_NODE_FEATURES}")
    print(f"  Gizli boyut      : {HIDDEN_DIM}")
    print(f"  Aksiyon sayisi   : {NUM_ACTIONS}")
    print(f"  Kavsak sayisi    : {NUM_NODES}")

    for name, expected in [("proj", True), ("gru", True), ("attn", True),
                            ("attn_norm", True), ("actor", True), ("critic", True)]:
        if hasattr(agent, name):
            results.ok(f"Katman mevcut: {name}")
        else:
            results.fail(f"Katman eksik: {name}")

    if trainable == total and total > 0:
        results.ok(f"Tum parametreler egit. ({trainable:,})")
    else:
        results.fail(f"Parametre sorunu: {trainable}/{total}")


# ══════════════════════════════════════════════════
#  TEST 2: Ileri Gecis
# ══════════════════════════════════════════════════

def test_forward_pass(agent: SimplePPOAgent, results: TestResult):
    print(f"\n{'═' * 60}")
    print("  TEST 2: Ileri Gecis (tum senaryolar)")
    print(f"{'═' * 60}")

    hidden = agent.init_hidden(NUM_NODES)

    for name, obs_list in SCENARIOS.items():
        print(f"\n  --- {name} ---")
        try:
            x = parse_sumo_observations(obs_list)
        except Exception as e:
            results.fail(f"[{name}] parse hatasi", str(e))
            continue

        if x.shape != (NUM_NODES, NUM_NODE_FEATURES):
            results.fail(f"[{name}] Giris boyutu yanlis: {x.shape}")
            continue

        if x.min() < -0.01 or x.max() > 1.5:
            results.fail(f"[{name}] Normalizasyon hatasi (min={x.min():.3f} max={x.max():.3f})")
        else:
            results.ok(f"[{name}] Normalizasyon dogru (min={x.min():.3f} max={x.max():.3f})")

        try:
            with torch.no_grad():
                actions, log_probs, value, hidden = agent.select_actions(x, hidden)
        except Exception as e:
            results.fail(f"[{name}] select_actions hatasi", str(e))
            continue

        shape_ok = (
            actions.shape   == (NUM_NODES,) and
            log_probs.shape == (NUM_NODES,) and
            value.numel()   == 1 and
            hidden.shape    == (1, NUM_NODES, HIDDEN_DIM)
        )
        if shape_ok:
            results.ok(f"[{name}] Cikti boyutlari dogru")
        else:
            results.fail(f"[{name}] Boyut hatasi — "
                         f"act={actions.shape} lp={log_probs.shape} "
                         f"v={value.shape} h={hidden.shape}")

        if actions.min() >= 0 and actions.max() < NUM_ACTIONS:
            results.ok(f"[{name}] Aksiyonlar gecerli [0,{NUM_ACTIONS})")
        else:
            results.fail(f"[{name}] Aksiyon aralik disi: {actions.tolist()}")

        obs_sorted = sorted(obs_list, key=lambda d: d["intersection_id"])
        for i in range(NUM_NODES):
            o  = obs_sorted[i]
            tot = o["north_count"] + o["south_count"] + o["east_count"] + o["west_count"]
            a   = actions[i].item()
            p   = math.exp(log_probs[i].item()) * 100
            print(f"    {KAVSAK_NAMES[i]}: {int(tot):3d} arac → "
                  f"Faz {a} ({PHASE_NAMES[a]}) [%{p:.1f}]")
        print(f"    Ag degeri: {value.item():.4f}")


# ══════════════════════════════════════════════════
#  TEST 3: Odul Fonksiyonu
# ══════════════════════════════════════════════════

def test_reward_function(results: TestResult):
    print(f"\n{'═' * 60}")
    print("  TEST 3: Odul Fonksiyonu")
    print(f"{'═' * 60}")

    normal  = SCENARIOS["Normal Trafik"]
    heavy   = SCENARIOS["Yogun Trafik (K3 darboğaz)"]
    night   = SCENARIOS["Gece Trafigi"]
    a_same  = torch.tensor([0, 0, 0, 0, 0])
    a_diff  = torch.tensor([1, 2, 3, 0, 1])

    r_normal = compute_reward(normal, None, None, a_same).item()
    r_heavy  = compute_reward(heavy,  None, None, a_same).item()
    r_night  = compute_reward(night,  None, None, a_same).item()

    print(f"\n  Ilk adim odulleri:")
    print(f"    Normal  : {r_normal:.4f}")
    print(f"    Yogun   : {r_heavy:.4f}")
    print(f"    Gece    : {r_night:.4f}")

    if r_heavy < r_normal:
        results.ok("Yogun trafik daha dusuk odul aliyor")
    else:
        results.fail("Yogun trafik odulu beklenenden yuksek",
                     f"heavy={r_heavy:.4f} normal={r_normal:.4f}")

    if r_night > r_normal:
        results.ok("Gece trafigi (az arac) daha yuksek odul aliyor")
    else:
        results.fail("Gece trafigi odulu beklenenden dusuk",
                     f"night={r_night:.4f} normal={r_normal:.4f}")

    r_impr = compute_reward(normal, heavy,  a_same, a_same).item()
    r_wors = compute_reward(heavy,  normal, a_same, a_same).item()
    print(f"\n  Throughput testi:")
    print(f"    Iyilesme (yogun→normal) : {r_impr:.4f}")
    print(f"    Kotuleşme (normal→yogun): {r_wors:.4f}")
    if r_impr > r_wors:
        results.ok("Throughput bileseni dogru calisiyor")
    else:
        results.fail("Throughput bileseni hatali")

    r_stable = compute_reward(normal, normal, a_same, a_same).item()
    r_change = compute_reward(normal, normal, a_same, a_diff).item()
    print(f"\n  Faz stabilitesi testi:")
    print(f"    Stabil : {r_stable:.4f}")
    print(f"    Degisim: {r_change:.4f}")
    if r_stable >= r_change:
        results.ok("Gereksiz faz degisimi cezalandiriliyor")
    else:
        results.fail("Faz stabilitesi cezasi calısmiyor")

    long_wait = [{**o, "phase_duration": 90.0} for o in normal]
    r_long  = compute_reward(long_wait, None, None, a_same).item()
    r_short = compute_reward(normal,    None, None, a_same).item()
    print(f"\n  Bekleme cezasi testi:")
    print(f"    Normal sure  : {r_short:.4f}")
    print(f"    Uzun (90 sn) : {r_long:.4f}")
    if r_long < r_short:
        results.ok("Uzun bekleme cezalandiriliyor")
    else:
        results.fail("Bekleme cezasi calısmiyor")


# ══════════════════════════════════════════════════
#  TEST 4: Attention Etkisi
# ══════════════════════════════════════════════════

def test_attention_effect(agent: SimplePPOAgent, results: TestResult):
    print(f"\n{'═' * 60}")
    print("  TEST 4: Attention Katmanı Etkisi")
    print(f"{'═' * 60}")

    obs = SCENARIOS["Yogun Trafik (K3 darboğaz)"]
    x   = parse_sumo_observations(obs)
    h   = agent.init_hidden(NUM_NODES)

    with torch.no_grad():
        proj_out = agent.proj(x)                          # (N, H)
        gru_out, _ = agent.gru(proj_out.unsqueeze(1), h)
        gru_out = gru_out.squeeze(1)                      # (N, H) — before attention

        attn_in  = gru_out.unsqueeze(0)
        attn_out, attn_weights = agent.attn(attn_in, attn_in, attn_in)
        after = agent.attn_norm(gru_out + attn_out.squeeze(0))  # (N, H)

    diff = (after - gru_out).abs().mean().item()
    print(f"\n  GRU oncesi/sonrasi attention farki: {diff:.6f}")
    if diff > 1e-6:
        results.ok("Attention katmani ozellikleri degistiriyor")
    else:
        results.fail("Attention etkisiz (fark ≈ 0)")

    # K3 (index 2) en yogun kavşak — temsili en cok degismeli
    k3_change = (after[2] - gru_out[2]).abs().mean().item()
    print(f"  K3 (darboğaz) ozellik degisimi: {k3_change:.6f}")
    if k3_change > 1e-6:
        results.ok("Darboğaz kavsağin temsili attention ile guncellendi")
    else:
        results.fail("Darboğaz kavsak temsili degismedi")

    # Attention agırlıkları: (1, N, N)
    print(f"\n  Attention agirlik matrisi (K→K):")
    w = attn_weights.squeeze(0)  # (N, N)
    for i in range(NUM_NODES):
        row = " ".join(f"{w[i, j].item():.3f}" for j in range(NUM_NODES))
        print(f"    {KAVSAK_NAMES[i]}: [{row}]")
    results.ok("Attention agirliklari hesaplandi")


# ══════════════════════════════════════════════════
#  TEST 5: Tutarlilik
# ══════════════════════════════════════════════════

def test_consistency(agent: SimplePPOAgent, results: TestResult):
    print(f"\n{'═' * 60}")
    print("  TEST 5: Tutarlilik (Deterministik Mod)")
    print(f"{'═' * 60}")

    obs = SCENARIOS["Normal Trafik"]
    x   = parse_sumo_observations(obs)

    decisions_list = []
    values_list    = []

    for _ in range(5):
        h = agent.init_hidden(NUM_NODES)
        with torch.no_grad():
            probs, value, _ = agent(x, h)
            decisions_list.append(torch.argmax(probs, dim=-1).tolist())
            values_list.append(value.item())

    print()
    for i, (d, v) in enumerate(zip(decisions_list, values_list)):
        labels = [f"{KAVSAK_NAMES[j]}={d[j]}" for j in range(NUM_NODES)]
        print(f"    Deneme {i+1}: [{', '.join(labels)}]  V={v:.4f}")

    if all(d == decisions_list[0] for d in decisions_list):
        results.ok("Deterministik kararlar tutarli")
    else:
        results.fail("Deterministik kararlar tutarsiz")

    if max(values_list) - min(values_list) < 1e-5:
        results.ok("Value tahminleri tutarli")
    else:
        results.fail(f"Value tahminleri degisken: {[f'{v:.4f}' for v in values_list]}")


# ══════════════════════════════════════════════════
#  TEST 6: Entropi ve Kesif
# ══════════════════════════════════════════════════

def test_entropy(agent: SimplePPOAgent, results: TestResult):
    print(f"\n{'═' * 60}")
    print("  TEST 6: Entropi ve Kesif Davranisi")
    print(f"{'═' * 60}")

    obs = SCENARIOS["Normal Trafik"]
    x   = parse_sumo_observations(obs)
    h   = agent.init_hidden(NUM_NODES)

    with torch.no_grad():
        probs, _, _ = agent(x, h)

    max_entropy = math.log(NUM_ACTIONS)
    print(f"\n  Maksimum entropi (uniform): {max_entropy:.4f}")
    print()

    entropies = []
    all_valid = True
    for i in range(NUM_NODES):
        p = probs[i]
        if p.min() < 0 or abs(p.sum().item() - 1.0) > 1e-4:
            results.fail(f"{KAVSAK_NAMES[i]} gecersiz dagilim: {p.tolist()}")
            all_valid = False
            continue
        ent   = -(p * p.log().clamp(min=-100)).sum().item()
        ratio = ent / max_entropy * 100
        bar   = "█" * int(ratio / 5) + "░" * (20 - int(ratio / 5))
        entropies.append(ent)
        print(f"    {KAVSAK_NAMES[i]}: {ent:.4f} ({ratio:.1f}%) {bar}")
        print(f"          olasiliklar: [{', '.join(f'{p[j]:.3f}' for j in range(NUM_ACTIONS))}]")

    if all_valid:
        results.ok("Tum olasilik dagilimlari gecerli")

    if entropies:
        avg = sum(entropies) / len(entropies)
        print(f"\n  Ortalama entropi: {avg:.4f}")
        if avg < max_entropy * 0.95:
            results.ok(f"Model ogrenilmis tercihler gosteriyor (H={avg:.4f} < max={max_entropy:.4f})")
        else:
            results.fail(f"Entropi hala maksimuma yakin — model random olabilir (H={avg:.4f})")
        if avg > 0.05:
            results.ok("Model hala kesif yapiyor (entropi > 0)")
        else:
            results.fail("Entropi cok dusuk — politika cokmus olabilir")


# ══════════════════════════════════════════════════
#  TEST 7: Uc Durumlar
# ══════════════════════════════════════════════════

def test_edge_cases(agent: SimplePPOAgent, results: TestResult):
    print(f"\n{'═' * 60}")
    print("  TEST 7: Uc Durumlar")
    print(f"{'═' * 60}")

    h = agent.init_hidden(NUM_NODES)

    # 7a: Bos trafik
    empty = [{"intersection_id": i, "north_count": 0, "south_count": 0,
               "east_count": 0, "west_count": 0, "queue_length": 0.0,
               "current_phase": 0, "phase_duration": 0.0} for i in range(NUM_NODES)]
    try:
        x = parse_sumo_observations(empty)
        agent.select_actions(x, h)
        results.ok("Bos trafik senaryosu calisti")
    except Exception as e:
        results.fail("Bos trafik coktu", str(e))

    # 7b: Asiri yogun
    extreme = [{"intersection_id": i, "north_count": 999, "south_count": 999,
                 "east_count": 999, "west_count": 999, "queue_length": 9999.0,
                 "current_phase": 3, "phase_duration": 999.0} for i in range(NUM_NODES)]
    try:
        x = parse_sumo_observations(extreme)
        _, _, value, _ = agent.select_actions(x, h)
        if torch.isnan(value) or torch.isinf(value):
            results.fail("Asiri yogunlukta NaN/Inf olustu")
        else:
            results.ok(f"Asiri yogunlukta stabil (value={value.item():.4f})")
    except Exception as e:
        results.fail("Asiri yogun senaryo coktu", str(e))

    # 7c: Tek kavsak spike
    spike = [{"intersection_id": 0, "north_count": 50, "south_count": 40,
               "east_count": 60, "west_count": 30, "queue_length": 200.0,
               "current_phase": 1, "phase_duration": 55.0}] + [
              {"intersection_id": i, "north_count": 0, "south_count": 0,
               "east_count": 0, "west_count": 0, "queue_length": 0.0,
               "current_phase": 0, "phase_duration": 10.0} for i in range(1, NUM_NODES)]
    try:
        x = parse_sumo_observations(spike)
        agent.select_actions(x, h)
        results.ok("Tek kavsak spike senaryosu calisti")
    except Exception as e:
        results.fail("Spike senaryo coktu", str(e))

    # 7d: GRU hidden state 20 adim stabilitesi
    print(f"\n  Hidden state stabilitesi (20 adim):")
    obs = SCENARIOS["Normal Trafik"]
    x   = parse_sumo_observations(obs)
    hh  = agent.init_hidden(NUM_NODES)
    norms = []
    for _ in range(20):
        with torch.no_grad():
            _, _, _, hh = agent.select_actions(x, hh)
            norms.append(hh.norm().item())
    print(f"    Norm: {' → '.join(f'{n:.2f}' for n in norms[::4])} ...")
    if not any(math.isnan(n) or math.isinf(n) for n in norms):
        results.ok("Hidden state 20 adimda stabil (NaN/Inf yok)")
    else:
        results.fail("Hidden state karsiz hale geldi")
    if norms[-1] < norms[0] * 100:
        results.ok("Hidden state norm patlamiyor")
    else:
        results.fail(f"Hidden state norm buyuyor: {norms[0]:.2f} → {norms[-1]:.2f}")


# ══════════════════════════════════════════════════
#  TEST 8: GAE Hesaplama
# ══════════════════════════════════════════════════

def test_gae(results: TestResult):
    print(f"\n{'═' * 60}")
    print("  TEST 8: GAE Hesaplama")
    print(f"{'═' * 60}")

    rewards    = [torch.tensor(r) for r in [1.0, 0.5, -0.3, 0.8, 0.2]]
    values     = [torch.tensor(v) for v in [0.5, 0.4,  0.3, 0.6, 0.1]]
    next_value = torch.tensor(0.2)

    try:
        advantages, returns = compute_gae(rewards, values, next_value)
        print(f"\n  Rewards    : {[f'{r.item():.2f}' for r in rewards]}")
        print(f"  Values     : {[f'{v.item():.2f}' for v in values]}")
        print(f"  Advantages : {[f'{a.item():.4f}' for a in advantages]}")
        print(f"  Returns    : {[f'{r.item():.4f}' for r in returns]}")

        if not torch.isnan(advantages).any():
            results.ok("GAE hesaplama basarili (NaN yok)")
        else:
            results.fail("GAE'de NaN olustu")

        adv_mean = advantages.mean().item()
        if abs(adv_mean) < 0.1:
            results.ok(f"Advantage normalize edilmis (mean={adv_mean:.4f})")
        else:
            results.fail(f"Advantage normalize degil (mean={adv_mean:.4f})")

    except Exception as e:
        results.fail("GAE hesaplama hatasi", traceback.format_exc())


# ══════════════════════════════════════════════════
#  TEST 9: PPO Loss ve Gradient Akisi
# ══════════════════════════════════════════════════

def test_ppo_loss(agent: SimplePPOAgent, results: TestResult):
    print(f"\n{'═' * 60}")
    print("  TEST 9: PPO Loss ve Gradient Akisi")
    print(f"{'═' * 60}")

    obs = SCENARIOS["Normal Trafik"]
    x   = parse_sumo_observations(obs)
    h   = agent.init_hidden(NUM_NODES)

    with torch.no_grad():
        actions, log_probs, value, _ = agent.select_actions(x, h)

    advantages = torch.randn(NUM_NODES)
    returns    = torch.tensor([1.0])

    agent.train()
    try:
        losses = agent.compute_ppo_loss(
            x=x, hidden_state=h,
            old_actions=actions, old_log_probs=log_probs,
            advantages=advantages, returns=returns,
        )
        print(f"\n  Total loss  : {losses['total'].item():.6f}")
        print(f"  Policy loss : {losses['policy'].item():.6f}")
        print(f"  Value loss  : {losses['value'].item():.6f}")
        print(f"  Entropy     : {losses['entropy'].item():.6f}")

        if not torch.isnan(losses["total"]):
            results.ok("PPO loss hesaplandi (NaN yok)")
        else:
            results.fail("PPO loss NaN")

        losses["total"].backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in agent.parameters()
        )
        if has_grad:
            results.ok("Gradientler basariyla hesaplandi")
        else:
            results.fail("Gradient akisi yok")

        max_grad = max(p.grad.abs().max().item() for p in agent.parameters() if p.grad is not None)
        print(f"  Maks gradient: {max_grad:.6f}")
        if max_grad < 100:
            results.ok(f"Gradient buyuklugu makul ({max_grad:.4f})")
        else:
            results.fail(f"Gradient patlamasi: {max_grad:.2f}")

    except Exception as e:
        results.fail("PPO loss hatasi", traceback.format_exc())
    finally:
        agent.eval()
        agent.zero_grad()


# ══════════════════════════════════════════════════
#  Ana Fonksiyon
# ══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TraFix Simple Model Test")
    parser.add_argument("--weights", type=str, default=None, help="Model agirlik dosyasi (.pth)")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  TraFix Simple — MODEL TEST SUITI")
    print("═" * 60)

    weights_path = args.weights or find_weights()
    agent        = load_agent(weights_path)
    results      = TestResult()

    test_model_info(agent, results)
    test_forward_pass(agent, results)
    test_reward_function(results)
    test_attention_effect(agent, results)
    test_consistency(agent, results)
    test_entropy(agent, results)
    test_edge_cases(agent, results)
    test_gae(results)
    test_ppo_loss(agent, results)

    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
