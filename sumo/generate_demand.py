import os
import sys
import subprocess
import random

def generate_dynamic_demand():
    """
    SUMO'nun randomTrips.py aracini kullanarak, her cagrildiginda
    farkli yogunluk paternlerine sahip (Dalgalanan/Mantili) bir trafik talebi olusturur.
    """
    if 'SUMO_HOME' not in os.environ:
        print("HATA: SUMO_HOME ortam degiskeni tanimli degil!")
        return

    sumo_tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    random_trips_script = os.path.join(sumo_tools, 'randomTrips.py')
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    net_file = os.path.join(script_dir, "map.net.xml")
    out_file = os.path.join(script_dir, "training_demand.rou.xml")

    # 1. Episode icin rastgele bir zorluk profili sec
    # - "Normal": Saniyede 1-2 arac (period=0.6 - 1.0)
    # - "Asiri Yogun": Saniyede 3-4 arac (period=0.2 - 0.4)
    # - "Dalgalı": Bazen cok durgun, bazen aniden patlayan trafik
    
    profile = random.choices(
        ["normal", "rush_hour", "chaotic"],
        weights=[0.4, 0.4, 0.2]
    )[0]

    if profile == "normal":
        period = round(random.uniform(0.7, 1.2), 2)
    elif profile == "rush_hour":
        period = round(random.uniform(0.2, 0.4), 2)
    else: # chaotic
        period = round(random.uniform(0.15, 0.8), 2)

    # randomTrips.py komutunu hazirla
    # --binomial n : N=Fringe edges sayisi. Trafik agin disindan gelip disina gitsin.
    cmd = [
        sys.executable, random_trips_script,
        "-n", net_file,
        "-o", out_file,
        "-e", "3600",         # 3600 saniyelik trafik (1 saat)
        "-p", str(period),    # Arac uretme periyodu (Daha kucuk = daha yogun)
        "--fringe-factor", "10", # Araclari mapin kenarlarindan baslat
        "--seed", str(random.randint(1, 99999)) # Her episode farkli arabalar
    ]

    print(f"[DEMAND] Yeni egitim trafigi uretiliyor... (Profil: {profile} | Yogunluk: {period})")
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_file
    except subprocess.CalledProcessError as e:
        print(f"randomTrips calistirilirken hata: {e}")
        return None

if __name__ == "__main__":
    generate_dynamic_demand()
