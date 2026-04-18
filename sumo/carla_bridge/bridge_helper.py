"""
Bridge Helper — SUMO <-> CARLA koordinat ve blueprint donusumleri.

SUMO: sol-elli koordinat sistemi, kuzey=yukari, aci saat yonuyle (0=kuzey)
CARLA: sag-elli (Unreal Engine), Y ekseni ters, aci CCW (0=dogu)
"""

import carla

# SUMO arac sinifi → CARLA blueprint oncelik listesi
VEHICLE_BLUEPRINT_MAP = {
    "passenger":  [
        "vehicle.tesla.model3",
        "vehicle.audi.a2",
        "vehicle.volkswagen.t2",
        "vehicle.bmw.grandtourer",
    ],
    "truck": [
        "vehicle.carlamotors.carlacola",
        "vehicle.tesla.cybertruck",
    ],
    "bus": [
        "vehicle.mitsubishi.futuristic_bus",
    ],
    "motorcycle": [
        "vehicle.kawasaki.ninja",
        "vehicle.yamaha.yzf",
    ],
    "bicycle": [
        "vehicle.bh.crossbike",
        "vehicle.gazelle.omafiets",
    ],
    "emergency": [
        "vehicle.dodge.charger_police",
        "vehicle.ford.ambulance",
    ],
    "DEFAULT": [
        "vehicle.tesla.model3",
        "vehicle.audi.a2",
        "vehicle.citroen.c3",
    ],
}

# SUMO TLS karakter → CARLA TrafficLightState
# SUMO: rRgGyYoOu — kucuk harf = yavaslama (g,y) veya kirmizi-sari (u)
TLS_STATE_MAP = {
    "G": carla.TrafficLightState.Green,
    "g": carla.TrafficLightState.Green,
    "Y": carla.TrafficLightState.Yellow,
    "y": carla.TrafficLightState.Yellow,
    "R": carla.TrafficLightState.Red,
    "r": carla.TrafficLightState.Red,
    "s": carla.TrafficLightState.Red,
    "u": carla.TrafficLightState.Yellow,  # kirmizi+sari (hazirlik) — sariya yakin
    "o": getattr(carla.TrafficLightState, "Off", carla.TrafficLightState.Unknown),
    "O": getattr(carla.TrafficLightState, "Off", carla.TrafficLightState.Unknown),
}


def get_carla_transform(
    sumo_pos,
    sumo_angle,
    sumo_offset=(0.0, 0.0),
    z=0.5,
):
    """
    SUMO (x, y, angle) → carla.Transform donusumu.

    sumo_offset: net.xml'deki netOffset degeri — koordinat sistemlerini hizalar.
    z: aracin CARLA dunyasindaki yuksekligi (yol ustu).
    """
    x = sumo_pos[0] - sumo_offset[0]
    y = -(sumo_pos[1] - sumo_offset[1])  # SUMO Y → CARLA -Y
    ang = float(sumo_angle) % 360.0
    yaw = -(ang - 90.0)  # SUMO 0=kuzey → CARLA 0=dogu

    return carla.Transform(
        carla.Location(x=x, y=y, z=z),
        carla.Rotation(yaw=yaw),
    )


def get_blueprint(blueprint_library, sumo_vclass):
    """SUMO arac sinifina gore uygun CARLA blueprint sec."""
    candidates = VEHICLE_BLUEPRINT_MAP.get(
        sumo_vclass, VEHICLE_BLUEPRINT_MAP["DEFAULT"]
    )
    for bp_id in candidates:
        try:
            bp = blueprint_library.find(bp_id)
            if bp is not None:
                return bp
        except Exception:
            continue
    # Son care: rastgele bir arac blueprint'i
    vehicles = blueprint_library.filter("vehicle.*")
    return vehicles[0] if vehicles else None


def sumo_color_to_carla(sumo_color_str):
    """'R,G,B' formatindaki SUMO renk string'ini carla.Color'a donusturur."""
    try:
        parts = [int(v) for v in sumo_color_str.split(",")][:3]
        return carla.Color(r=parts[0], g=parts[1], b=parts[2])
    except Exception:
        return carla.Color(r=180, g=180, b=180)
