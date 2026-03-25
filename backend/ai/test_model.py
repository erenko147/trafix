"""
TraFix — Egitilmis Model Test Scripti
=======================================
Kaydedilmis model agirliklarini yukler ve ornek verilerle test eder.

Kullanim:
  cd x:\\trafix\\proje
  .\\venv\\Scripts\\activate
  python -m backend.ai.test_model
"""

import os
import sys
import torch
from torch.distributions import Categorical

# Model import
try:
    from backend.ai.model import Core_PPO_Agent
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from backend.ai.model import Core_PPO_Agent


# ── Konfigurasyon ──
NUM_FEATURES = 7
HIDDEN_DIM = 64
NUM_ACTIONS = 4
NUM_NODES = 5

EDGE_INDEX = torch.tensor([
    [0, 1, 0, 2, 1, 3, 2, 3, 2, 4, 1, 0, 2, 0, 3, 1, 3, 2, 4, 2],
    [1, 0, 2, 0, 3, 1, 3, 2, 4, 2, 0, 1, 0, 2, 1, 3, 2, 3, 2, 4],
], dtype=torch.long)

PHASE_NAMES = {0: "Kuzey", 1: "Guney", 2: "Dogu", 3: "Bati"}
KAVSAK_NAMES = {0: "K1", 1: "K2", 2: "K3", 3: "K4", 4: "K5"}


def find_weights():
    """Model dosyasini bul."""
    paths = [
        os.path.join(os.path.dirname(__file__), "core_agent_weights.pth"),
        os.path.join(os.path.dirname(__file__), "..", "..", "core_agent_weights.pth"),
        "core_agent_weights.pth",
        "core_agent_weights_latest.pth",
        os.path.join(os.path.dirname(__file__), "..", "..", "core_agent_weights_latest.pth"),
    ]
    for p in paths:
        ap = os.path.abspath(p)
        if os.path.exists(ap):
            return ap
    return None


def test_with_sample_data(agent):
    """Ornek veri seti ile modeli test eder."""
    print("\n" + "=" * 60)
    print("  ORNEK VERi iLE TEST")
    print("=" * 60)

    # Dashboard State JSON'daki ornek veri
    test_scenarios = [
        {
            "name": "Normal Trafik",
            "data": torch.tensor([
                [4.0,  3.0,  12.0, 7.0,  15.2,  0.0, 12.0],   # K1
                [2.0,  1.0,  5.0,  3.0,  0.0,   2.0, 4.0],    # K2
                [10.0, 8.0,  3.0,  6.0,  45.0,  1.0, 22.0],   # K3
                [0.0,  2.0,  7.0,  1.0,  2.1,   3.0, 8.0],    # K4
                [5.0,  4.0,  2.0,  8.0,  8.5,   0.0, 10.0],   # K5
            ], dtype=torch.float32),
        },
        {
            "name": "Yogun Trafik (K3 darbogazli)",
            "data": torch.tensor([
                [8.0,  6.0,  15.0, 10.0, 35.0,  0.0, 15.0],
                [5.0,  3.0,  8.0,  6.0,  12.0,  2.0, 8.0],
                [35.0, 28.0, 15.0, 20.0, 180.0, 1.0, 40.0],  # K3 cok yogun
                [3.0,  5.0,  10.0, 2.0,  8.0,   3.0, 10.0],
                [10.0, 8.0,  5.0,  12.0, 25.0,  0.0, 12.0],
            ], dtype=torch.float32),
        },
        {
            "name": "Gece Trafigi (Dusuk yogunluk)",
            "data": torch.tensor([
                [1.0,  0.0,  2.0,  1.0,  1.0,   0.0, 30.0],
                [0.0,  1.0,  0.0,  0.0,  0.0,   2.0, 30.0],
                [2.0,  1.0,  1.0,  0.0,  2.0,   1.0, 30.0],
                [0.0,  0.0,  1.0,  0.0,  0.5,   3.0, 30.0],
                [1.0,  0.0,  0.0,  2.0,  1.5,   0.0, 30.0],
            ], dtype=torch.float32),
        },
    ]

    hidden = agent.init_hidden(NUM_NODES)

    for scenario in test_scenarios:
        print(f"\n--- Senaryo: {scenario['name']} ---")
        x = scenario["data"]

        with torch.no_grad():
            action_probs, state_value, hidden = agent(x, EDGE_INDEX, hidden)

        print(f"  Ag Degeri (Critic): {state_value.item():.4f}")
        print()

        for i in range(NUM_NODES):
            probs = action_probs[i]
            best_phase = torch.argmax(probs).item()
            confidence = probs[best_phase].item() * 100

            # Araclari goster
            n, s, e, w = x[i][0].item(), x[i][1].item(), x[i][2].item(), x[i][3].item()
            total = n + s + e + w

            print(f"  {KAVSAK_NAMES[i]}: Toplam {int(total)} arac "
                  f"(K:{int(n)} G:{int(s)} D:{int(e)} B:{int(w)})")
            print(f"        -> Karar: Faz {best_phase} ({PHASE_NAMES[best_phase]}) "
                  f"[Guven: %{confidence:.1f}]")
            print(f"           Olasiliklar: "
                  f"K:%{probs[0]*100:.1f}  G:%{probs[1]*100:.1f}  "
                  f"D:%{probs[2]*100:.1f}  B:%{probs[3]*100:.1f}")


def test_consistency(agent):
    """Ayni girdi icin tutarli cikti verdigini test eder."""
    print("\n" + "=" * 60)
    print("  TUTARLILIK TESTi")
    print("=" * 60)

    x = torch.tensor([
        [10.0, 5.0, 3.0, 7.0, 20.0, 0.0, 15.0],
        [2.0,  1.0, 8.0, 3.0, 5.0,  2.0, 8.0],
        [15.0, 12.0,4.0, 9.0, 60.0, 1.0, 25.0],
        [1.0,  3.0, 6.0, 0.0, 3.0,  3.0, 10.0],
        [7.0,  4.0, 2.0, 10.0,12.0, 0.0, 12.0],
    ], dtype=torch.float32)

    results = []
    for trial in range(5):
        hidden = agent.init_hidden(NUM_NODES)
        with torch.no_grad():
            probs, _, _ = agent(x, EDGE_INDEX, hidden)
            decisions = [torch.argmax(probs[i]).item() for i in range(NUM_NODES)]
            results.append(decisions)

    all_same = all(r == results[0] for r in results)
    print(f"\n  5 deneme sonucu: {'TUTARLI' if all_same else 'TUTARSIZ'}")
    for i, r in enumerate(results):
        labels = [f"{KAVSAK_NAMES[j]}={PHASE_NAMES[r[j]]}" for j in range(NUM_NODES)]
        print(f"    Deneme {i+1}: {', '.join(labels)}")


def test_model_info(agent, weights_path):
    """Model hakkinda bilgi verir."""
    print("=" * 60)
    print("  TraFix AI MODEL TESTi")
    print("=" * 60)

    file_size = os.path.getsize(weights_path) / 1024
    total_params = sum(p.numel() for p in agent.parameters())
    trainable = sum(p.numel() for p in agent.parameters() if p.requires_grad)

    print(f"\n  Model dosyasi : {weights_path}")
    print(f"  Dosya boyutu  : {file_size:.1f} KB")
    print(f"  Toplam param  : {total_params:,}")
    print(f"  Egitilir param: {trainable:,}")
    print(f"  Giris ozelligi: {NUM_FEATURES}")
    print(f"  Gizli boyut   : {HIDDEN_DIM}")
    print(f"  Aksiyon sayisi: {NUM_ACTIONS}")
    print(f"  Kavsak sayisi : {NUM_NODES}")


def main():
    # 1. Model dosyasini bul
    weights_path = find_weights()
    if weights_path is None:
        print("HATA: Model dosyasi bulunamadi!")
        print("Once egitim yap: python -m backend.ai.train")
        sys.exit(1)

    # 2. Modeli yukle
    agent = Core_PPO_Agent(NUM_FEATURES, HIDDEN_DIM, NUM_ACTIONS)
    agent.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    agent.eval()

    # 3. Testleri calistir
    test_model_info(agent, weights_path)
    test_with_sample_data(agent)
    test_consistency(agent)

    print("\n" + "=" * 60)
    print("  TUM TESTLER TAMAMLANDI")
    print("=" * 60)


if __name__ == "__main__":
    main()
