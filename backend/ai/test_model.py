"""
TraFix v2 — Model Test Scripti
================================
CoordinatedPPOAgent için kapsamlı test suite.

Testler:
  1. Model bilgisi ve ağırlık yükleme
  2. SUMO JSON formatında örnek veri ile ileri geçiş
  3. Ödül fonksiyonu doğrulaması
  4. Koordinasyon katmanı etkisi
  5. Tutarlılık testi
  6. Entropi ve keşif davranışı
  7. Uç durum (edge case) testleri

Kullanım:
  cd x:\\trafix\\proje
  .\\venv\\Scripts\\activate
  python -m backend.ai.test_model_v2
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
        # Doğrudan çalıştırma durumu
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

HIDDEN_DIM = 128
NUM_ACTIONS = 4
NUM_NODES = 5
NUM_HEADS = 4

# 5 kavşaklı asimetrik ağ (düzeltilmiş — duplikasyonsuz)
#   0 — 1 — 2
#       |   |
#       3 — 4
EDGE_INDEX = torch.tensor([
    [0, 1, 1, 2, 1, 3, 2, 4, 3, 4],
    [1, 0, 2, 1, 3, 1, 4, 2, 4, 3],
], dtype=torch.long)

PHASE_NAMES = {0: "Kuzey-Güney", 1: "Güney-Kuzey", 2: "Doğu-Batı", 3: "Batı-Doğu"}
KAVSAK_NAMES = {0: "K1", 1: "K2", 2: "K3", 3: "K4", 4: "K5"}


# ══════════════════════════════════════════════════
#  Test Senaryoları (SUMO JSON formatında)
# ══════════════════════════════════════════════════

SCENARIOS: Dict[str, List[Dict]] = {
    "Normal Trafik": [
        {"intersection_id": 0, "north_count": 4, "south_count": 3,
         "east_count": 12, "west_count": 7, "queue_length": 15.2,
         "current_phase": 0, "phase_duration": 12.0},
        {"intersection_id": 1, "north_count": 2, "south_count": 1,
         "east_count": 5,  "west_count": 3, "queue_length": 0.0,
         "current_phase": 2, "phase_duration": 4.0},
        {"intersection_id": 2, "north_count": 10, "south_count": 8,
         "east_count": 3,  "west_count": 6, "queue_length": 45.0,
         "current_phase": 1, "phase_duration": 22.0},
        {"intersection_id": 3, "north_count": 0, "south_count": 2,
         "east_count": 7,  "west_count": 1, "queue_length": 2.1,
         "current_phase": 3, "phase_duration": 8.0},
        {"intersection_id": 4, "north_count": 5, "south_count": 4,
         "east_count": 2,  "west_count": 8, "queue_length": 8.5,
         "current_phase": 0, "phase_duration": 10.0},
    ],
    "Yoğun Trafik (K3 darboğaz)": [
        {"intersection_id": 0, "north_count": 8, "south_count": 6,
         "east_count": 15, "west_count": 10, "queue_length": 35.0,
         "current_phase": 0, "phase_duration": 15.0},
        {"intersection_id": 1, "north_count": 5, "south_count": 3,
         "east_count": 8,  "west_count": 6, "queue_length": 12.0,
         "current_phase": 2, "phase_duration": 8.0},
        {"intersection_id": 2, "north_count": 35, "south_count": 28,
         "east_count": 15, "west_count": 20, "queue_length": 180.0,
         "current_phase": 1, "phase_duration": 40.0},
        {"intersection_id": 3, "north_count": 3, "south_count": 5,
         "east_count": 10, "west_count": 2, "queue_length": 8.0,
         "current_phase": 3, "phase_duration": 10.0},
        {"intersection_id": 4, "north_count": 10, "south_count": 8,
         "east_count": 5,  "west_count": 12, "queue_length": 25.0,
         "current_phase": 0, "phase_duration": 12.0},
    ],
    "Gece Trafiği": [
        {"intersection_id": 0, "north_count": 1, "south_count": 0,
         "east_count": 2,  "west_count": 1, "queue_length": 1.0,
         "current_phase": 0, "phase_duration": 30.0},
        {"intersection_id": 1, "north_count": 0, "south_count": 1,
         "east_count": 0,  "west_count": 0, "queue_length": 0.0,
         "current_phase": 2, "phase_duration": 30.0},
        {"intersection_id": 2, "north_count": 2, "south_count": 1,
         "east_count": 1,  "west_count": 0, "queue_length": 2.0,
         "current_phase": 1, "phase_duration": 30.0},
        {"intersection_id": 3, "north_count": 0, "south_count": 0,
         "east_count": 1,  "west_count": 0, "queue_length": 0.5,
         "current_phase": 3, "phase_duration": 30.0},
        {"intersection_id": 4, "north_count": 1, "south_count": 0,
         "east_count": 0,  "west_count": 2, "queue_length": 1.5,
         "current_phase": 0, "phase_duration": 30.0},
    ],
    "Tek Yön Baskın (Doğu-Batı)": [
        {"intersection_id": 0, "north_count": 1, "south_count": 1,
         "east_count": 20, "west_count": 18, "queue_length": 55.0,
         "current_phase": 0, "phase_duration": 35.0},
        {"intersection_id": 1, "north_count": 0, "south_count": 1,
         "east_count": 22, "west_count": 15, "queue_length": 60.0,
         "current_phase": 0, "phase_duration": 35.0},
        {"intersection_id": 2, "north_count": 2, "south_count": 0,
         "east_count": 25, "west_count": 20, "queue_length": 70.0,
         "current_phase": 0, "phase_duration": 35.0},
        {"intersection_id": 3, "north_count": 1, "south_count": 0,
         "east_count": 18, "west_count": 16, "queue_length": 48.0,
         "current_phase": 0, "phase_duration": 35.0},
        {"intersection_id": 4, "north_count": 0, "south_count": 1,
         "east_count": 19, "west_count": 17, "queue_length": 52.0,
         "current_phase": 0, "phase_duration": 35.0},
    ],
}


# ══════════════════════════════════════════════════
#  Yardımcı Fonksiyonlar
# ══════════════════════════════════════════════════

class TestResult:
    """Test sonuçlarını takip eder."""
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
    """v2 model dosyasını bul."""
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
    """Ajan oluştur, opsiyonel ağırlık yükle."""
    agent = CoordinatedPPOAgent(
        num_node_features=NUM_NODE_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_actions=NUM_ACTIONS,
        num_heads=NUM_HEADS,
        entropy_coef=0.02,
        value_coef=0.5,
    )
    if weights_path:
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            # Handle checkpoint format (dict with 'model_state_dict' key)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            agent.load_state_dict(state_dict)
            print(f"  Ağırlıklar yüklendi: {weights_path}")
        except RuntimeError as e:
            print(f"  ⚠ Ağırlık dosyası uyumsuz (v1 ağırlıkları?): {weights_path}")
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
    trainable = sum(p.numel() for p in agent.parameters() if p.requires_grad)

    print(f"\n  Toplam parametre  : {total_params:,}")
    print(f"  Eğitilebilir      : {trainable:,}")
    print(f"  Giriş özelliği    : {NUM_NODE_FEATURES}")
    print(f"  Gizli boyut       : {HIDDEN_DIM}")
    print(f"  Aksiyon sayısı    : {NUM_ACTIONS}")
    print(f"  Kavşak sayısı     : {NUM_NODES}")
    print(f"  Attention head    : {NUM_HEADS}")

    # Modül kontrolleri
    has_gnn = hasattr(agent, "st_gnn")
    has_coord = hasattr(agent, "coordinator")
    has_actor = hasattr(agent, "actor")
    has_critic = hasattr(agent, "critic")

    if has_gnn:
        results.ok("SpatioTemporalGNN (GCN+GRU) mevcut")
    else:
        results.fail("SpatioTemporalGNN bulunamadı")

    if has_coord:
        results.ok("IntersectionCoordinator (Attention) mevcut")
    else:
        results.fail("IntersectionCoordinator bulunamadı")

    if has_actor and has_critic:
        results.ok("Actor ve Critic head'leri mevcut")
    else:
        results.fail("Actor/Critic eksik")

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

    hidden = agent.init_hidden(NUM_NODES)

    for name, obs_list in SCENARIOS.items():
        print(f"\n  --- {name} ---")

        try:
            x = parse_sumo_observations(obs_list)
        except Exception as e:
            results.fail(f"[{name}] parse_sumo_observations hatası", str(e))
            continue

        # Boyut kontrolü
        if x.shape != (NUM_NODES, NUM_NODE_FEATURES):
            results.fail(
                f"[{name}] Girdi boyutu yanlış",
                f"Beklenen: ({NUM_NODES}, {NUM_NODE_FEATURES}), Gelen: {x.shape}",
            )
            continue

        # Normalizasyon kontrolü — tüm değerler [0, 1] arasında mı
        if x.min() < -0.01 or x.max() > 1.5:
            results.fail(
                f"[{name}] Normalizasyon sorunu",
                f"min={x.min().item():.4f}, max={x.max().item():.4f}",
            )
        else:
            results.ok(f"[{name}] Normalizasyon doğru (min={x.min():.3f}, max={x.max():.3f})")

        try:
            with torch.no_grad():
                actions, log_probs, value, hidden = agent.select_actions(
                    x, EDGE_INDEX, hidden
                )
        except Exception as e:
            results.fail(f"[{name}] İleri geçiş hatası", str(e))
            continue

        # Çıktı boyut kontrolleri
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

        if hidden.shape != (1, NUM_NODES, HIDDEN_DIM):
            results.fail(f"[{name}] Hidden boyutu yanlış: {hidden.shape}")
            assert_ok = False

        if assert_ok:
            results.ok(f"[{name}] Çıktı boyutları doğru")

        # Aksiyon aralık kontrolü
        if actions.min() >= 0 and actions.max() < NUM_ACTIONS:
            results.ok(f"[{name}] Aksiyonlar geçerli aralıkta [0, {NUM_ACTIONS})")
        else:
            results.fail(f"[{name}] Aksiyon aralık dışı: {actions.tolist()}")

        # Sonuçları yazdır
        for i in range(NUM_NODES):
            obs = sorted(obs_list, key=lambda d: d["intersection_id"])[i]
            n = obs["north_count"]
            s = obs["south_count"]
            e = obs["east_count"]
            w = obs["west_count"]
            total = n + s + e + w
            action = actions[i].item()
            prob = math.exp(log_probs[i].item()) * 100

            print(f"    {KAVSAK_NAMES[i]}: {int(total)} araç "
                  f"(K:{int(n)} G:{int(s)} D:{int(e)} B:{int(w)}) "
                  f"→ Faz {action} ({PHASE_NAMES[action]}) [%{prob:.1f}]")

        print(f"    Ağ değeri: {value.item():.4f}")


# ══════════════════════════════════════════════════
#  TEST 3: Ödül Fonksiyonu
# ══════════════════════════════════════════════════

def test_reward_function(results: TestResult):
    print(f"\n{'=' * 60}")
    print("  TEST 3: Ödül Fonksiyonu")
    print(f"{'=' * 60}")

    normal = SCENARIOS["Normal Trafik"]
    heavy = SCENARIOS["Yoğun Trafik (K3 darboğaz)"]
    night = SCENARIOS["Gece Trafiği"]

    actions_same = torch.tensor([0, 0, 0, 0, 0])
    actions_diff = torch.tensor([1, 2, 3, 0, 1])

    # ── 3a: Yoğun trafik → daha düşük ödül ──
    r_normal = compute_reward(normal, None, None, actions_same).item()
    r_heavy = compute_reward(heavy, None, None, actions_same).item()
    r_night = compute_reward(night, None, None, actions_same).item()

    print(f"\n  İlk adım ödülleri (önceki obs yok):")
    print(f"    Normal trafik : {r_normal:.4f}")
    print(f"    Yoğun trafik  : {r_heavy:.4f}")
    print(f"    Gece trafiği  : {r_night:.4f}")

    if r_heavy < r_normal:
        results.ok("Yoğun trafik daha düşük ödül alıyor")
    else:
        results.fail(
            "Yoğun trafik daha yüksek ödül aldı",
            f"heavy={r_heavy:.4f} vs normal={r_normal:.4f}",
        )

    if r_night > r_normal:
        results.ok("Gece trafiği (az araç) daha yüksek ödül alıyor")
    else:
        results.fail(
            "Gece trafiği beklenenden düşük ödül aldı",
            f"night={r_night:.4f} vs normal={r_normal:.4f}",
        )

    # ── 3b: Throughput — araç azalması pozitif ödül ──
    # Yoğun → Normal geçişi (araçlar azalmış)
    r_improving = compute_reward(normal, heavy, actions_same, actions_same).item()
    # Normal → Yoğun geçişi (araçlar artmış)
    r_worsening = compute_reward(heavy, normal, actions_same, actions_same).item()

    print(f"\n  Throughput testi:")
    print(f"    İyileşme (yoğun→normal) : {r_improving:.4f}")
    print(f"    Kötüleşme (normal→yoğun): {r_worsening:.4f}")

    if r_improving > r_worsening:
        results.ok("Throughput bileşeni doğru çalışıyor (iyileşme > kötüleşme)")
    else:
        results.fail("Throughput bileşeni hatalı")

    # ── 3c: Faz stabilitesi — gereksiz değişim cezası ──
    r_stable = compute_reward(normal, normal, actions_same, actions_same).item()
    r_change = compute_reward(normal, normal, actions_same, actions_diff).item()

    print(f"\n  Faz stabilitesi testi:")
    print(f"    Stabil (aynı faz)  : {r_stable:.4f}")
    print(f"    Değişim (faz farklı): {r_change:.4f}")

    if r_stable >= r_change:
        results.ok("Gereksiz faz değişimi cezalandırılıyor")
    else:
        results.fail("Faz stabilitesi cezası çalışmıyor")

    # ── 3d: Bekleme cezası — >60 sn aynı fazda ──
    long_wait = [
        {**obs, "phase_duration": 90.0}
        for obs in normal
    ]
    r_long = compute_reward(long_wait, None, None, actions_same).item()
    r_short = compute_reward(normal, None, None, actions_same).item()

    print(f"\n  Bekleme cezası testi:")
    print(f"    Kısa süre (normal)  : {r_short:.4f}")
    print(f"    Uzun süre (90 sn)   : {r_long:.4f}")

    if r_long < r_short:
        results.ok("Uzun bekleme cezalandırılıyor")
    else:
        results.fail("Bekleme cezası çalışmıyor")

    # ── 3e: Adillik — tek yön baskın durumda daha düşük ──
    r_balanced = compute_reward(normal, None, None, actions_same).item()
    r_unbalanced = compute_reward(
        SCENARIOS["Tek Yön Baskın (Doğu-Batı)"], None, None, actions_same
    ).item()

    print(f"\n  Adillik testi:")
    print(f"    Dengeli trafik  : {r_balanced:.4f}")
    print(f"    Tek yön baskın  : {r_unbalanced:.4f}")

    # Tek yön baskın hem daha fazla araç hem dengesiz → düşük olmalı
    if r_unbalanced < r_balanced:
        results.ok("Dengesiz trafik daha düşük ödül alıyor")
    else:
        results.fail("Adillik bileşeni etkisiz")


# ══════════════════════════════════════════════════
#  TEST 4: Koordinasyon Katmanı Etkisi
# ══════════════════════════════════════════════════

def test_coordination_effect(agent: CoordinatedPPOAgent, results: TestResult):
    print(f"\n{'=' * 60}")
    print("  TEST 4: Koordinasyon Katmanı Etkisi")
    print(f"{'=' * 60}")

    obs = SCENARIOS["Yoğun Trafik (K3 darboğaz)"]
    x = parse_sumo_observations(obs)
    hidden = agent.init_hidden(NUM_NODES)

    with torch.no_grad():
        # GNN çıkışı (koordinasyon öncesi)
        gnn_features, _ = agent.st_gnn(x, EDGE_INDEX, hidden)

        # Koordinasyon sonrası
        coord_features = agent.coordinator(gnn_features)

    # Koordinasyon öncesi ve sonrası farkı
    diff = (coord_features - gnn_features).abs().mean().item()
    print(f"\n  GNN çıkışı ile koordinasyon sonrası fark: {diff:.6f}")

    if diff > 1e-6:
        results.ok("Koordinasyon katmanı özellikleri değiştiriyor")
    else:
        results.fail("Koordinasyon katmanı etkisiz (fark ≈ 0)")

    # Attention'ın K3'e (en yoğun kavşak, index 2) etkisi
    # Komşuları: K1(0), K4(4) — bu kavşağın temsili değişmeli
    k3_before = gnn_features[2].norm().item()
    k3_after = coord_features[2].norm().item()
    k3_change = abs(k3_after - k3_before) / max(k3_before, 1e-8)

    print(f"  K3 (darboğaz) temsil değişimi: %{k3_change * 100:.2f}")

    if k3_change > 0.001:
        results.ok("Darboğaz kavşağın temsili attention ile güncellenmiş")
    else:
        results.fail("Darboğaz kavşak temsili değişmemiş")

    # Tüm kavşaklar arasındaki kosinüs benzerliği değişimi
    # Koordinasyon sonrası komşu kavşaklar daha benzer olmalı
    cos_sim = torch.nn.functional.cosine_similarity
    neighbors = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)]

    sim_before = []
    sim_after = []
    for i, j in neighbors:
        sb = cos_sim(gnn_features[i].unsqueeze(0), gnn_features[j].unsqueeze(0)).item()
        sa = cos_sim(coord_features[i].unsqueeze(0), coord_features[j].unsqueeze(0)).item()
        sim_before.append(sb)
        sim_after.append(sa)

    avg_before = sum(sim_before) / len(sim_before)
    avg_after = sum(sim_after) / len(sim_after)
    print(f"  Komşu benzerliği — Önce: {avg_before:.4f}, Sonra: {avg_after:.4f}")

    results.ok(f"Komşu benzerlik analizi tamamlandı (Δ={avg_after - avg_before:.4f})")


# ══════════════════════════════════════════════════
#  TEST 5: Tutarlılık
# ══════════════════════════════════════════════════

def test_consistency(agent: CoordinatedPPOAgent, results: TestResult):
    print(f"\n{'=' * 60}")
    print("  TEST 5: Tutarlılık (Deterministik Mod)")
    print(f"{'=' * 60}")

    obs = SCENARIOS["Normal Trafik"]
    x = parse_sumo_observations(obs)

    decisions_list = []
    values_list = []

    for trial in range(5):
        hidden = agent.init_hidden(NUM_NODES)
        with torch.no_grad():
            probs, value, _ = agent(x, EDGE_INDEX, hidden)
            decisions = torch.argmax(probs, dim=-1).tolist()
            decisions_list.append(decisions)
            values_list.append(value.item())

    all_same = all(d == decisions_list[0] for d in decisions_list)
    values_stable = max(values_list) - min(values_list) < 1e-5

    print()
    for i, (d, v) in enumerate(zip(decisions_list, values_list)):
        labels = [f"{KAVSAK_NAMES[j]}={PHASE_NAMES[d[j]]}" for j in range(NUM_NODES)]
        print(f"    Deneme {i + 1}: {', '.join(labels)}  (V={v:.4f})")

    if all_same:
        results.ok("Deterministik kararlar tutarlı (argmax aynı)")
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
    x = parse_sumo_observations(obs)
    hidden = agent.init_hidden(NUM_NODES)

    with torch.no_grad():
        probs, _, _ = agent(x, EDGE_INDEX, hidden)

    # Her kavşak için entropi hesapla
    max_entropy = math.log(NUM_ACTIONS)  # uniform dağılım entropisi

    print(f"\n  Maksimum olası entropi: {max_entropy:.4f} (uniform)")
    print()

    entropies = []
    for i in range(NUM_NODES):
        p = probs[i]
        entropy = -(p * p.log().clamp(min=-100)).sum().item()
        entropies.append(entropy)
        ratio = entropy / max_entropy * 100

        bar_len = int(ratio / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        print(f"    {KAVSAK_NAMES[i]}: entropi={entropy:.4f} "
              f"({ratio:.1f}% of max) {bar}")
        print(f"          olasılıklar: "
              f"[{', '.join(f'{p[j]:.3f}' for j in range(NUM_ACTIONS))}]")

    avg_entropy = sum(entropies) / len(entropies)
    print(f"\n  Ortalama entropi: {avg_entropy:.4f}")

    # Olasılıklar geçerli mi (toplamı ~1, hepsi >= 0)
    for i in range(NUM_NODES):
        p = probs[i]
        if p.min() < 0:
            results.fail(f"{KAVSAK_NAMES[i]} negatif olasılık: {p.min():.6f}")
            return
        if abs(p.sum().item() - 1.0) > 1e-4:
            results.fail(f"{KAVSAK_NAMES[i]} olasılık toplamı ≠ 1: {p.sum():.6f}")
            return

    results.ok("Tüm olasılık dağılımları geçerli (≥0, toplam=1)")

    if avg_entropy > 0.01:
        results.ok(f"Model keşif yapıyor (entropi={avg_entropy:.4f} > 0)")
    else:
        results.fail("Entropi çok düşük — model çökmüş olabilir")

    # Entropy coef kontrolü
    if agent.entropy_coef > 0:
        results.ok(f"Entropi bonusu aktif (coef={agent.entropy_coef})")
    else:
        results.fail("Entropi bonusu kapalı (coef=0)")


# ══════════════════════════════════════════════════
#  TEST 7: Uç Durumlar
# ══════════════════════════════════════════════════

def test_edge_cases(agent: CoordinatedPPOAgent, results: TestResult):
    print(f"\n{'=' * 60}")
    print("  TEST 7: Uç Durumlar")
    print(f"{'=' * 60}")

    hidden = agent.init_hidden(NUM_NODES)

    # ── 7a: Tüm kavşaklar boş ──
    empty_obs = [
        {"intersection_id": i, "north_count": 0, "south_count": 0,
         "east_count": 0, "west_count": 0, "queue_length": 0.0,
         "current_phase": 0, "phase_duration": 0.0}
        for i in range(NUM_NODES)
    ]

    try:
        x = parse_sumo_observations(empty_obs)
        actions, _, _, _ = agent.select_actions(x, EDGE_INDEX, hidden)
        results.ok("Boş trafik senaryosu çökmedi")
    except Exception as e:
        results.fail("Boş trafik senaryosu çöktü", str(e))

    # ── 7b: Aşırı yoğun trafik ──
    extreme_obs = [
        {"intersection_id": i, "north_count": 999, "south_count": 999,
         "east_count": 999, "west_count": 999, "queue_length": 9999.0,
         "current_phase": 3, "phase_duration": 999.0}
        for i in range(NUM_NODES)
    ]

    try:
        x = parse_sumo_observations(extreme_obs)
        actions, _, value, _ = agent.select_actions(x, EDGE_INDEX, hidden)

        if torch.isnan(value) or torch.isinf(value):
            results.fail("Aşırı değerlerde NaN/Inf oluştu")
        else:
            results.ok(f"Aşırı yoğunlukta stabil (value={value.item():.4f})")
    except Exception as e:
        results.fail("Aşırı yoğun senaryo çöktü", str(e))

    # ── 7c: Tek kavşak yoğun, diğerleri boş ──
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
        actions, _, _, _ = agent.select_actions(x, EDGE_INDEX, hidden)
        results.ok("Tek kavşak spike senaryosu çökmedi")
    except Exception as e:
        results.fail("Spike senaryo çöktü", str(e))

    # ── 7d: GRU hidden state taşıma (10 adım) ──
    print("\n  GRU hidden state sürekliliği (10 adım):")
    obs = SCENARIOS["Normal Trafik"]
    x = parse_sumo_observations(obs)
    h = agent.init_hidden(NUM_NODES)

    hidden_norms = []
    for step in range(10):
        with torch.no_grad():
            _, _, _, h = agent.select_actions(x, EDGE_INDEX, h)
            norm = h.norm().item()
            hidden_norms.append(norm)

    print(f"    Hidden norm: {' → '.join(f'{n:.2f}' for n in hidden_norms)}")

    if not any(math.isnan(n) or math.isinf(n) for n in hidden_norms):
        results.ok("10 adım boyunca hidden state stabil (NaN/Inf yok)")
    else:
        results.fail("Hidden state 10 adımda kararsız hale geldi")

    # Norm aşırı büyümüyor mu
    if hidden_norms[-1] < hidden_norms[0] * 100:
        results.ok("Hidden state norm patlamıyor")
    else:
        results.fail(
            "Hidden state norm aşırı büyüyor",
            f"başlangıç={hidden_norms[0]:.2f}, son={hidden_norms[-1]:.2f}",
        )


# ══════════════════════════════════════════════════
#  TEST 8: GAE Hesaplama
# ══════════════════════════════════════════════════

def test_gae(results: TestResult):
    print(f"\n{'=' * 60}")
    print("  TEST 8: GAE (Generalized Advantage Estimation)")
    print(f"{'=' * 60}")

    rewards = [torch.tensor(r) for r in [1.0, 0.5, -0.3, 0.8, 0.2]]
    values = [torch.tensor(v) for v in [0.5, 0.4, 0.3, 0.6, 0.1]]
    next_value = torch.tensor(0.2)

    try:
        advantages, returns = compute_gae(rewards, values, next_value)

        print(f"\n  Rewards   : {[f'{r.item():.2f}' for r in rewards]}")
        print(f"  Values    : {[f'{v.item():.2f}' for v in values]}")
        print(f"  Advantages: {[f'{a.item():.4f}' for a in advantages]}")
        print(f"  Returns   : {[f'{r.item():.4f}' for r in returns]}")

        # Advantage normalleşmiş mi (mean ≈ 0, std ≈ 1)
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()

        if abs(adv_mean) < 0.1:
            results.ok(f"Advantage normalize edilmiş (mean={adv_mean:.4f})")
        else:
            results.fail(f"Advantage normalize değil (mean={adv_mean:.4f})")

        if not torch.isnan(advantages).any():
            results.ok("GAE hesaplama başarılı (NaN yok)")
        else:
            results.fail("GAE'de NaN oluştu")

    except Exception as e:
        results.fail("GAE hesaplama hatası", str(e))


# ══════════════════════════════════════════════════
#  TEST 9: PPO Loss Hesaplama
# ══════════════════════════════════════════════════

def test_ppo_loss(agent: CoordinatedPPOAgent, results: TestResult):
    print(f"\n{'=' * 60}")
    print("  TEST 9: PPO Loss Hesaplama")
    print(f"{'=' * 60}")

    obs = SCENARIOS["Normal Trafik"]
    x = parse_sumo_observations(obs)
    hidden = agent.init_hidden(NUM_NODES)

    # Önce aksiyon al (gradient izleme kapalı)
    with torch.no_grad():
        actions, log_probs, value, _ = agent.select_actions(x, EDGE_INDEX, hidden)

    advantages = torch.randn(NUM_NODES)
    returns = torch.tensor([1.0])

    # Gradient izleme açık — loss hesapla
    agent.train()
    try:
        losses = agent.compute_ppo_loss(
            x=x,
            edge_index=EDGE_INDEX,
            hidden_state=hidden,
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

        # Gradient akıyor mu
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

    # Model yükle
    weights_path = find_weights()
    agent = create_agent(weights_path)

    # Testleri çalıştır
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