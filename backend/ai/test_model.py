"""
TraFix v2 — Model Test Scripti
================================
CoordinatedPPOAgent için kapsamlı test suite.

Testler:
  1. Model bilgisi  (GRU removed, GCN-only)
  2. SUMO JSON formatında ileri geçiş  (no hidden_state)
  3. Ödül fonksiyonu — (N,) per-node tensor
  4. Koordinasyon katmanı etkisi
  5. Tutarlılık (deterministik mod)
  6. Entropi ve keşif davranışı
  7. Uç durumlar
  8. GAE — (T, N) tensors
  9. PPO Loss hesaplama

Kullanım:
  python backend/ai/test_model.py
"""

import os
import sys
import math
import torch
import traceback
from typing import Dict, List, Optional

# ── Model import ──
try:
    from backend.ai.trafix_v2 import (
        CoordinatedPPOAgent,
        parse_sumo_observations,
        compute_reward,
        compute_gae,
        RewardWeights,
        NUM_NODE_FEATURES,
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        from backend.ai.trafix_v2 import (
            CoordinatedPPOAgent,
            parse_sumo_observations,
            compute_reward,
            compute_gae,
            RewardWeights,
            NUM_NODE_FEATURES,
        )
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from trafix_v2 import (
            CoordinatedPPOAgent,
            parse_sumo_observations,
            compute_reward,
            compute_gae,
            RewardWeights,
            NUM_NODE_FEATURES,
        )


# ══════════════════════════════════════════════════
#  Sabitler
# ══════════════════════════════════════════════════

HIDDEN_DIM  = 128
NUM_ACTIONS = 4
NUM_NODES   = 5
NUM_HEADS   = 4

EDGE_INDEX = torch.tensor([
    [0, 1, 1, 2, 1, 3, 2, 4, 3, 4],
    [1, 0, 2, 1, 3, 1, 4, 2, 4, 3],
], dtype=torch.long)

PHASE_NAMES  = {0: "N-only", 1: "E-only", 2: "S-only", 3: "W-only"}
KAVSAK_NAMES = {0: "K1", 1: "K2", 2: "K3", 3: "K4", 4: "K5"}


# ══════════════════════════════════════════════════
#  Test Senaryoları
# ══════════════════════════════════════════════════

SCENARIOS: Dict[str, List[Dict]] = {
    "Normal Trafik": [
        {"intersection_id": 0, "north_count": 4,  "south_count": 3,
         "east_count": 12, "west_count": 7,  "queue_length": 15.2,
         "current_phase": 0, "phase_duration": 12.0},
        {"intersection_id": 1, "north_count": 2,  "south_count": 1,
         "east_count": 5,  "west_count": 3,  "queue_length": 0.0,
         "current_phase": 2, "phase_duration": 4.0},
        {"intersection_id": 2, "north_count": 10, "south_count": 8,
         "east_count": 3,  "west_count": 6,  "queue_length": 45.0,
         "current_phase": 1, "phase_duration": 22.0},
        {"intersection_id": 3, "north_count": 0,  "south_count": 2,
         "east_count": 7,  "west_count": 1,  "queue_length": 2.1,
         "current_phase": 3, "phase_duration": 8.0},
        {"intersection_id": 4, "north_count": 5,  "south_count": 4,
         "east_count": 2,  "west_count": 8,  "queue_length": 8.5,
         "current_phase": 0, "phase_duration": 10.0},
    ],
    "Yoğun Trafik (K3 darboğaz)": [
        {"intersection_id": 0, "north_count": 8,  "south_count": 6,
         "east_count": 15, "west_count": 10, "queue_length": 35.0,
         "current_phase": 0, "phase_duration": 15.0},
        {"intersection_id": 1, "north_count": 5,  "south_count": 3,
         "east_count": 8,  "west_count": 6,  "queue_length": 12.0,
         "current_phase": 2, "phase_duration": 8.0},
        {"intersection_id": 2, "north_count": 35, "south_count": 28,
         "east_count": 15, "west_count": 20, "queue_length": 180.0,
         "current_phase": 1, "phase_duration": 40.0},
        {"intersection_id": 3, "north_count": 3,  "south_count": 5,
         "east_count": 10, "west_count": 2,  "queue_length": 8.0,
         "current_phase": 3, "phase_duration": 10.0},
        {"intersection_id": 4, "north_count": 10, "south_count": 8,
         "east_count": 5,  "west_count": 12, "queue_length": 25.0,
         "current_phase": 0, "phase_duration": 12.0},
    ],
    "Gece Trafiği": [
        {"intersection_id": 0, "north_count": 1, "south_count": 0,
         "east_count": 2,  "west_count": 1,  "queue_length": 1.0,
         "current_phase": 0, "phase_duration": 30.0},
        {"intersection_id": 1, "north_count": 0, "south_count": 1,
         "east_count": 0,  "west_count": 0,  "queue_length": 0.0,
         "current_phase": 2, "phase_duration": 30.0},
        {"intersection_id": 2, "north_count": 2, "south_count": 1,
         "east_count": 1,  "west_count": 0,  "queue_length": 2.0,
         "current_phase": 1, "phase_duration": 30.0},
        {"intersection_id": 3, "north_count": 0, "south_count": 0,
         "east_count": 1,  "west_count": 0,  "queue_length": 0.5,
         "current_phase": 3, "phase_duration": 30.0},
        {"intersection_id": 4, "north_count": 1, "south_count": 0,
         "east_count": 0,  "west_count": 2,  "queue_length": 1.5,
         "current_phase": 0, "phase_duration": 30.0},
    ],
    "Tek Yön Baskın (Doğu-Batı)": [
        {"intersection_id": 0, "north_count": 1,  "south_count": 1,
         "east_count": 20, "west_count": 18, "queue_length": 55.0,
         "current_phase": 0, "phase_duration": 35.0},
        {"intersection_id": 1, "north_count": 0,  "south_count": 1,
         "east_count": 22, "west_count": 15, "queue_length": 60.0,
         "current_phase": 0, "phase_duration": 35.0},
        {"intersection_id": 2, "north_count": 2,  "south_count": 0,
         "east_count": 25, "west_count": 20, "queue_length": 70.0,
         "current_phase": 0, "phase_duration": 35.0},
        {"intersection_id": 3, "north_count": 1,  "south_count": 0,
         "east_count": 18, "west_count": 16, "queue_length": 48.0,
         "current_phase": 0, "phase_duration": 35.0},
        {"intersection_id": 4, "north_count": 0,  "south_count": 1,
         "east_count": 19, "west_count": 17, "queue_length": 52.0,
         "current_phase": 0, "phase_duration": 35.0},
    ],
}


# ══════════════════════════════════════════════════
#  Yardımcı
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

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"  SONUÇ: {self.passed}/{total} test başarılı", end="")
        if self.failed > 0:
            print(f" — {self.failed} BAŞARISIZ")
            for e in self.errors:
                print(f"    ✗ {e}")
        else:
            print(" — HEPSİ GEÇTİ")
        print(f"{'=' * 60}")
        return self.failed == 0


def find_weights() -> Optional[str]:
    candidates = [
        "coordinated_agent_weights.pth",
        "core_agent_weights.pth",
        "core_agent_weights_latest.pth",
    ]
    dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(__file__), "..", ".."),
        ".",
    ]
    for d in dirs:
        for c in candidates:
            p = os.path.abspath(os.path.join(d, c))
            if os.path.exists(p):
                return p
    return None


def create_agent(weights_path: Optional[str] = None) -> CoordinatedPPOAgent:
    agent = CoordinatedPPOAgent(
        num_node_features=NUM_NODE_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_actions=NUM_ACTIONS,
        num_heads=NUM_HEADS,
        entropy_coef=0.005,
        value_coef=0.25,
    )
    if weights_path:
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            agent.load_state_dict(state_dict)
            print(f"  Ağırlıklar yüklendi: {weights_path}")
        except RuntimeError as e:
            print(f"  ⚠ Ağırlık dosyası uyumsuz: {weights_path}")
            print(f"    Hata: {str(e)[:200]}...")
            print("  → Rastgele başlatılmış model ile test ediliyor")
    else:
        print("  Ağırlık dosyası bulunamadı — rastgele başlatılmış model test ediliyor")
    agent.eval()
    return agent


# ══════════════════════════════════════════════════
#  TEST 1: Model Bilgisi
# ══════════════════════════════════════════════════

def test_model_info(agent: CoordinatedPPOAgent, results: TestResult):
    print(f"\n{'=' * 60}")
    print("  TEST 1: Model Bilgisi")
    print(f"{'=' * 60}")

    total_params = sum(p.numel() for p in agent.parameters())
    trainable    = sum(p.numel() for p in agent.parameters() if p.requires_grad)

    print(f"\n  Toplam parametre  : {total_params:,}")
    print(f"  Giriş özelliği    : {NUM_NODE_FEATURES}  (beklenen: 10)")
    print(f"  Gizli boyut       : {HIDDEN_DIM}")
    print(f"  Aksiyon sayısı    : {NUM_ACTIONS}")

    # GCN present
    if hasattr(agent, "st_gnn"):
        results.ok("SpatioTemporalGNN (GCN) mevcut")
    else:
        results.fail("SpatioTemporalGNN bulunamadı")

    # GRU must be gone
    if hasattr(agent, "st_gnn") and not hasattr(agent.st_gnn, "gru"):
        results.ok("GRU kaldırıldı (GCN-only)")
    else:
        results.fail("GRU hâlâ mevcut — kaldırılmalı")

    # init_hidden must be gone
    if not hasattr(agent, "init_hidden"):
        results.ok("init_hidden kaldırıldı")
    else:
        results.fail("init_hidden hâlâ mevcut")

    # Coordinator and heads
    if hasattr(agent, "coordinator"):
        results.ok("IntersectionCoordinator (Attention) mevcut")
    else:
        results.fail("IntersectionCoordinator bulunamadı")

    if hasattr(agent, "actor") and hasattr(agent, "critic"):
        results.ok("Actor ve Critic head'leri mevcut")
    else:
        results.fail("Actor/Critic eksik")

    # NUM_NODE_FEATURES
    if NUM_NODE_FEATURES == 10:
        results.ok(f"NUM_NODE_FEATURES == 10")
    else:
        results.fail(f"NUM_NODE_FEATURES yanlış: {NUM_NODE_FEATURES} (beklenen 10)")

    if total_params > 0 and trainable == total_params:
        results.ok(f"Tüm parametreler eğitilebilir ({trainable:,})")
    else:
        results.fail(f"Parametre sorunu: {trainable}/{total_params}")


# ══════════════════════════════════════════════════
#  TEST 2: SUMO Verisi ile İleri Geçiş
# ══════════════════════════════════════════════════

def test_forward_pass(agent: CoordinatedPPOAgent, results: TestResult):
    print(f"\n{'=' * 60}")
    print("  TEST 2: SUMO Verisi ile İleri Geçiş")
    print(f"{'=' * 60}")

    for name, obs_list in SCENARIOS.items():
        print(f"\n  --- {name} ---")

        try:
            x = parse_sumo_observations(obs_list)
        except Exception as e:
            results.fail(f"[{name}] parse_sumo_observations hatası", str(e))
            continue

        # Shape check: (N, 10)
        if x.shape != (NUM_NODES, NUM_NODE_FEATURES):
            results.fail(
                f"[{name}] Girdi boyutu yanlış",
                f"Beklenen: ({NUM_NODES}, {NUM_NODE_FEATURES}), Gelen: {x.shape}",
            )
            continue

        # Normalisation range [0, 1]
        if x.min() < -0.01 or x.max() > 1.01:
            results.fail(
                f"[{name}] Normalizasyon sorunu",
                f"min={x.min().item():.4f}, max={x.max().item():.4f}",
            )
        else:
            results.ok(f"[{name}] Girdi şekil/normaliz doğru")

        try:
            with torch.no_grad():
                actions, log_probs, value = agent.select_actions(x, EDGE_INDEX)
        except Exception as e:
            results.fail(f"[{name}] İleri geçiş hatası", str(e))
            continue

        assert_ok = True
        if actions.shape != (NUM_NODES,):
            results.fail(f"[{name}] Aksiyon boyutu yanlış: {actions.shape}")
            assert_ok = False
        if log_probs.shape != (NUM_NODES,):
            results.fail(f"[{name}] Log-prob boyutu yanlış: {log_probs.shape}")
            assert_ok = False
        if value.numel() != 1:
            results.fail(f"[{name}] Value boyutu yanlış: {value.shape}")
            assert_ok = False
        if assert_ok:
            results.ok(f"[{name}] Çıktı boyutları doğru (actions={actions.shape}, value scalar)")

        if actions.min() >= 0 and actions.max() < NUM_ACTIONS:
            results.ok(f"[{name}] Aksiyonlar geçerli [0, {NUM_ACTIONS})")
        else:
            results.fail(f"[{name}] Aksiyon aralık dışı: {actions.tolist()}")

        for i in range(NUM_NODES):
            obs   = sorted(obs_list, key=lambda d: d["intersection_id"])[i]
            total = obs["north_count"] + obs["south_count"] + obs["east_count"] + obs["west_count"]
            prob  = math.exp(log_probs[i].item()) * 100
            print(f"    {KAVSAK_NAMES[i]}: {int(total)} araç "
                  f"→ {PHASE_NAMES[actions[i].item()]} [%{prob:.1f}]")
        print(f"    Ağ değeri: {value.item():.4f}")


# ══════════════════════════════════════════════════
#  TEST 3: Ödül Fonksiyonu — (N,) tensor
# ══════════════════════════════════════════════════

def test_reward_function(results: TestResult):
    print(f"\n{'=' * 60}")
    print("  TEST 3: Ödül Fonksiyonu — per-node (N,) tensor")
    print(f"{'=' * 60}")

    normal  = SCENARIOS["Normal Trafik"]
    heavy   = SCENARIOS["Yoğun Trafik (K3 darboğaz)"]
    night   = SCENARIOS["Gece Trafiği"]

    actions_same = torch.tensor([0, 0, 0, 0, 0])
    actions_diff = torch.tensor([1, 2, 3, 0, 1])

    r_normal = compute_reward(normal, None, None, actions_same)
    r_heavy  = compute_reward(heavy,  None, None, actions_same)
    r_night  = compute_reward(night,  None, None, actions_same)

    # Shape check
    if r_normal.shape == (NUM_NODES,):
        results.ok(f"compute_reward şekli (N,) = ({NUM_NODES},) doğru")
    else:
        results.fail(f"compute_reward şekli yanlış: {r_normal.shape}")

    print(f"\n  Normal trafik  (per-node): {r_normal.tolist()}")
    print(f"  Yoğun trafik   (per-node): {r_heavy.tolist()}")
    print(f"  Gece trafiği   (per-node): {r_night.tolist()}")

    if r_heavy.mean() < r_normal.mean():
        results.ok("Yoğun trafik ortalaması < normal (doğru yön)")
    else:
        results.fail("Yoğun trafik beklenenden yüksek ödül aldı")

    if r_night.mean() > r_normal.mean():
        results.ok("Gece trafiği ortalaması > normal (az araç → iyi)")
    else:
        results.fail("Gece trafiği beklenenden düşük ödül aldı")

    # Throughput
    r_improve = compute_reward(normal, heavy, actions_same, actions_same)
    r_worsen  = compute_reward(heavy,  normal, actions_same, actions_same)
    if r_improve.mean() > r_worsen.mean():
        results.ok("Throughput bileşeni doğru çalışıyor")
    else:
        results.fail("Throughput bileşeni hatalı")

    # Phase stability
    r_stable = compute_reward(normal, normal, actions_same, actions_same)
    r_change = compute_reward(normal, normal, actions_same, actions_diff)
    if r_stable.mean() >= r_change.mean():
        results.ok("Gereksiz faz değişimi cezalandırılıyor")
    else:
        results.fail("Faz stabilitesi cezası çalışmıyor")

    # Wait penalty
    long_wait = [{**obs, "phase_duration": 90.0} for obs in normal]
    r_long  = compute_reward(long_wait, None, None, actions_same)
    r_short = compute_reward(normal,    None, None, actions_same)
    if r_long.mean() < r_short.mean():
        results.ok("Uzun bekleme cezalandırılıyor")
    else:
        results.fail("Bekleme cezası çalışmıyor")


# ══════════════════════════════════════════════════
#  TEST 4: Koordinasyon Katmanı Etkisi
# ══════════════════════════════════════════════════

def test_coordination_effect(agent: CoordinatedPPOAgent, results: TestResult):
    print(f"\n{'=' * 60}")
    print("  TEST 4: Koordinasyon Katmanı Etkisi")
    print(f"{'=' * 60}")

    obs = SCENARIOS["Yoğun Trafik (K3 darboğaz)"]
    x   = parse_sumo_observations(obs)

    with torch.no_grad():
        gnn_features  = agent.st_gnn(x, EDGE_INDEX)          # (N, H)
        coord_features = agent.coordinator(gnn_features)     # (N, H)

    diff = (coord_features - gnn_features).abs().mean().item()
    print(f"\n  GNN → Coord fark: {diff:.6f}")

    if diff > 1e-6:
        results.ok("Koordinasyon katmanı özellikleri değiştiriyor")
    else:
        results.fail("Koordinasyon katmanı etkisiz (fark ≈ 0)")

    k3_before = gnn_features[2].norm().item()
    k3_after  = coord_features[2].norm().item()
    k3_change = abs(k3_after - k3_before) / max(k3_before, 1e-8)

    print(f"  K3 (darboğaz) temsil değişimi: %{k3_change * 100:.2f}")

    if k3_change > 0.001:
        results.ok("Darboğaz kavşağın temsili attention ile güncellenmiş")
    else:
        results.fail("Darboğaz kavşak temsili değişmemiş")

    results.ok("Koordinasyon analizi tamamlandı")


# ══════════════════════════════════════════════════
#  TEST 5: Tutarlılık
# ══════════════════════════════════════════════════

def test_consistency(agent: CoordinatedPPOAgent, results: TestResult):
    print(f"\n{'=' * 60}")
    print("  TEST 5: Tutarlılık (Deterministik Mod)")
    print(f"{'=' * 60}")

    obs = SCENARIOS["Normal Trafik"]
    x   = parse_sumo_observations(obs)

    decisions_list = []
    values_list    = []

    for _ in range(5):
        with torch.no_grad():
            probs, value = agent(x, EDGE_INDEX)
            decisions = torch.argmax(probs, dim=-1).tolist()
            decisions_list.append(decisions)
            values_list.append(value.item())

    all_same     = all(d == decisions_list[0] for d in decisions_list)
    values_stable = max(values_list) - min(values_list) < 1e-5

    print()
    for i, (d, v) in enumerate(zip(decisions_list, values_list)):
        labels = [f"{KAVSAK_NAMES[j]}={PHASE_NAMES[d[j]]}" for j in range(NUM_NODES)]
        print(f"    Deneme {i + 1}: {', '.join(labels)}  (V={v:.4f})")

    if all_same:
        results.ok("Deterministik kararlar tutarlı")
    else:
        results.fail("Deterministik kararlar tutarsız")

    if values_stable:
        results.ok("Value tahminleri tutarlı")
    else:
        results.fail(f"Value tahminleri değişken: {values_list}")


# ══════════════════════════════════════════════════
#  TEST 6: Entropi ve Keşif
# ══════════════════════════════════════════════════

def test_entropy_exploration(agent: CoordinatedPPOAgent, results: TestResult):
    print(f"\n{'=' * 60}")
    print("  TEST 6: Entropi ve Keşif Davranışı")
    print(f"{'=' * 60}")

    obs = SCENARIOS["Normal Trafik"]
    x   = parse_sumo_observations(obs)

    with torch.no_grad():
        probs, _ = agent(x, EDGE_INDEX)

    max_entropy = math.log(NUM_ACTIONS)
    print(f"\n  Maksimum entropi: {max_entropy:.4f} (uniform)")
    print()

    entropies = []
    for i in range(NUM_NODES):
        p = probs[i]
        entropy = -(p * p.log().clamp(min=-100)).sum().item()
        entropies.append(entropy)
        ratio   = entropy / max_entropy * 100
        bar_len = int(ratio / 5)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {KAVSAK_NAMES[i]}: entropi={entropy:.4f} ({ratio:.1f}%) {bar}")
        print(f"          probs: [{', '.join(f'{p[j]:.3f}' for j in range(NUM_ACTIONS))}]")

    avg_entropy = sum(entropies) / len(entropies)
    print(f"\n  Ortalama entropi: {avg_entropy:.4f}")

    for i in range(NUM_NODES):
        p = probs[i]
        if p.min() < 0:
            results.fail(f"{KAVSAK_NAMES[i]} negatif olasılık")
            return
        if abs(p.sum().item() - 1.0) > 1e-4:
            results.fail(f"{KAVSAK_NAMES[i]} olasılık toplamı ≠ 1: {p.sum():.6f}")
            return

    results.ok("Tüm olasılık dağılımları geçerli (≥0, toplam=1)")

    if avg_entropy > 0.01:
        results.ok(f"Model keşif yapıyor (entropi={avg_entropy:.4f})")
    else:
        results.fail("Entropi çok düşük — model çökmüş olabilir")

    if agent.entropy_coef > 0:
        results.ok(f"Entropi bonusu aktif (coef={agent.entropy_coef})")
    else:
        results.fail("Entropi bonusu kapalı")


# ══════════════════════════════════════════════════
#  TEST 7: Uç Durumlar
# ══════════════════════════════════════════════════

def test_edge_cases(agent: CoordinatedPPOAgent, results: TestResult):
    print(f"\n{'=' * 60}")
    print("  TEST 7: Uç Durumlar")
    print(f"{'=' * 60}")

    # 7a: Tüm kavşaklar boş
    empty_obs = [
        {"intersection_id": i, "north_count": 0, "south_count": 0,
         "east_count": 0, "west_count": 0, "queue_length": 0.0,
         "current_phase": 0, "phase_duration": 0.0}
        for i in range(NUM_NODES)
    ]
    try:
        x = parse_sumo_observations(empty_obs)
        actions, _, _ = agent.select_actions(x, EDGE_INDEX)
        results.ok("Boş trafik senaryosu çökmedi")
    except Exception as e:
        results.fail("Boş trafik senaryosu çöktü", str(e))

    # 7b: Aşırı yoğun trafik
    extreme_obs = [
        {"intersection_id": i, "north_count": 999, "south_count": 999,
         "east_count": 999, "west_count": 999, "queue_length": 9999.0,
         "current_phase": 3, "phase_duration": 999.0}
        for i in range(NUM_NODES)
    ]
    try:
        x = parse_sumo_observations(extreme_obs)
        actions, _, value = agent.select_actions(x, EDGE_INDEX)
        if torch.isnan(value) or torch.isinf(value):
            results.fail("Aşırı değerlerde NaN/Inf oluştu")
        else:
            results.ok(f"Aşırı yoğunlukta stabil (value={value.item():.4f})")
    except Exception as e:
        results.fail("Aşırı yoğun senaryo çöktü", str(e))

    # 7c: Tek kavşak yoğun
    spike_obs = [
        {"intersection_id": 0, "north_count": 50, "south_count": 40,
         "east_count": 60, "west_count": 30, "queue_length": 200.0,
         "current_phase": 1, "phase_duration": 55.0},
    ] + [
        {"intersection_id": i, "north_count": 0, "south_count": 0,
         "east_count": 0, "west_count": 0, "queue_length": 0.0,
         "current_phase": 0, "phase_duration": 10.0}
        for i in range(1, NUM_NODES)
    ]
    try:
        x = parse_sumo_observations(spike_obs)
        actions, _, _ = agent.select_actions(x, EDGE_INDEX)
        results.ok("Tek kavşak spike senaryosu çökmedi")
    except Exception as e:
        results.fail("Spike senaryo çöktü", str(e))

    # 7d: Per-node reward shape check
    obs      = SCENARIOS["Normal Trafik"]
    actions_t = torch.tensor([0, 1, 2, 3, 0])
    reward = compute_reward(obs, None, None, actions_t)
    if reward.shape == (NUM_NODES,):
        results.ok(f"compute_reward (N,) = ({NUM_NODES},) doğru")
    else:
        results.fail(f"compute_reward beklenen ({NUM_NODES},) ama {reward.shape} geldi")

    # 7e: One-hot phase normalisation — phase bits in {0,1}
    x = parse_sumo_observations(SCENARIOS["Normal Trafik"])
    phase_cols = x[:, 5:9]  # columns 5–8 are one-hot
    unique_vals = phase_cols.unique().tolist()
    if set(unique_vals).issubset({0.0, 1.0}):
        results.ok("One-hot phase bits ∈ {0, 1}")
    else:
        results.fail(f"One-hot phase bits yanlış değerler: {unique_vals}")


# ══════════════════════════════════════════════════
#  TEST 8: GAE — (T, N) Tensors
# ══════════════════════════════════════════════════

def test_gae(results: TestResult):
    print(f"\n{'=' * 60}")
    print("  TEST 8: GAE (T, N) şekil testi")
    print(f"{'=' * 60}")

    T = 5
    N = NUM_NODES

    rewards    = [torch.full((N,), r) for r in [1.0, 0.5, -0.3, 0.8, 0.2]]
    values     = [torch.tensor(v)     for v in [0.5, 0.4,  0.3, 0.6, 0.1]]
    next_value = torch.tensor(0.2)

    try:
        advantages, returns = compute_gae(rewards, values, next_value)

        print(f"\n  advantages şekli: {advantages.shape}  (beklenen: ({T}, {N}))")
        print(f"  returns    şekli: {returns.shape}  (beklenen: ({T}, {N}))")

        if advantages.shape == (T, N):
            results.ok(f"advantages (T, N) = ({T}, {N}) doğru")
        else:
            results.fail(f"advantages boyutu yanlış: {advantages.shape}")

        if returns.shape == (T, N):
            results.ok(f"returns (T, N) = ({T}, {N}) doğru")
        else:
            results.fail(f"returns boyutu yanlış: {returns.shape}")

        if not torch.isnan(advantages).any():
            results.ok("GAE hesaplama başarılı (NaN yok)")
        else:
            results.fail("GAE'de NaN oluştu")

        adv_mean = advantages.mean().item()
        if abs(adv_mean) < 0.1:
            results.ok(f"Advantage normalize edilmiş (mean={adv_mean:.4f})")
        else:
            results.fail(f"Advantage normalize değil (mean={adv_mean:.4f})")

    except Exception as e:
        results.fail("GAE hesaplama hatası", str(e))


# ══════════════════════════════════════════════════
#  TEST 9: PPO Loss
# ══════════════════════════════════════════════════

def test_ppo_loss(agent: CoordinatedPPOAgent, results: TestResult):
    print(f"\n{'=' * 60}")
    print("  TEST 9: PPO Loss Hesaplama")
    print(f"{'=' * 60}")

    obs = SCENARIOS["Normal Trafik"]
    x   = parse_sumo_observations(obs)

    with torch.no_grad():
        actions, log_probs, value = agent.select_actions(x, EDGE_INDEX)

    advantages = torch.randn(NUM_NODES)
    returns    = torch.randn(NUM_NODES)

    agent.train()
    try:
        losses = agent.compute_ppo_loss(
            x=x,
            edge_index=EDGE_INDEX,
            old_actions=actions,
            old_log_probs=log_probs,
            advantages=advantages,
            returns=returns,
        )

        print(f"\n  Total loss  : {losses['total'].item():.6f}")
        print(f"  Policy loss : {losses['policy'].item():.6f}")
        print(f"  Value loss  : {losses['value'].item():.6f}")
        print(f"  Entropy     : {losses['entropy'].item():.6f}")

        if not torch.isnan(losses["total"]):
            results.ok("PPO loss hesaplandı (NaN yok)")
        else:
            results.fail("PPO loss NaN")

        losses["total"].backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in agent.parameters()
        )
        if has_grad:
            results.ok("Gradientler başarıyla hesaplandı")
        else:
            results.fail("Gradient akışı yok")

    except Exception as e:
        results.fail("PPO loss hatası", traceback.format_exc())
    finally:
        agent.eval()
        agent.zero_grad()


# ══════════════════════════════════════════════════
#  Ana Fonksiyon
# ══════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  TraFix v2 — KAPSAMLİ MODEL TESTİ")
    print("═" * 60)

    results = TestResult()

    weights_path = find_weights()
    agent = create_agent(weights_path)

    test_model_info(agent, results)
    test_forward_pass(agent, results)
    test_reward_function(results)
    test_coordination_effect(agent, results)
    test_consistency(agent, results)
    test_entropy_exploration(agent, results)
    test_edge_cases(agent, results)
    test_gae(results)
    test_ppo_loss(agent, results)

    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
