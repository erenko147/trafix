using System;

// ── Araç (kisa JSON anahtarlari: i/x/y/a/s/c) ────────────────────────────
[Serializable]
public class VehicleData
{
    public string i;   // id
    public float  x;
    public float  y;
    public float  a;   // angle (SUMO: 0=Kuzey, 90=Dogu, saat yonu)
    public float  s;   // speed m/s
    public string c;   // vehicle class: passenger/truck/bus/motorcycle
}

// ── Trafik isigi (kisa: i/p/s) ────────────────────────────────────────────
[Serializable]
public class TLSData
{
    public string i;   // id
    public int    p;   // phase (0=NS-yesil, 1=sari, 2=EW-yesil, 3=sari)
    public string s;   // SUMO state string "GGGgrrrrGGGgrrrr"
}

// ── Frame mesaji ──────────────────────────────────────────────────────────
[Serializable]
public class FrameMessage
{
    public string       type;  // "frame"
    public int          t;     // step
    public VehicleData[] v;
    public TLSData[]    tls;
}

// ── Init: kavşak ──────────────────────────────────────────────────────────
[Serializable]
public class JunctionInfo
{
    public string id;
    public float  x;
    public float  y;
    public int    tls;  // 1 = trafik isikli
}

// ── Init: kenar (yol) ─────────────────────────────────────────────────────
[Serializable]
public class EdgeInfo
{
    public string  id;
    public int     ln;     // serit sayisi
    public float   w;      // toplam yol genisligi (m)
    public float[] sh;     // shape: [x1,y1, x2,y2, ...] ciftleri
}

// ── Init mesaji ───────────────────────────────────────────────────────────
[Serializable]
public class InitMessage
{
    public string        type;      // "init"
    public float[]       bounds;    // [xmin, ymin, xmax, ymax]
    public JunctionInfo[] junctions;
    public EdgeInfo[]    edges;
}
