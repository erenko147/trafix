using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Arac havuzu — her arac icin bir GameObject (govde + kabin).
///
/// Optimizasyonlar:
///   • Dead-reckoning: SUMO frame'leri arasi hareket smooth tahmin edilir
///   • Spawn-snap: ilk gorundugunde aninda dogru pozisyona gelir (ucma yok)
///   • GC-free iterasyon: _poolKeys listesi ile Dictionary enumerator heap allocasyonu yok
///   • Frame-rate bagimsiz lerp: Mathf.Exp ile dogru zaman sabiti
/// </summary>
public class VehiclePool : MonoBehaviour
{
    public int ActiveCount { get; private set; }

    // ── Boyutlar: (en, yukseklik, uzunluk) ───────────────────────────────
    private static readonly (float w, float h, float l)[] SIZE = {
        (2.0f, 1.55f, 4.5f),   // 0: passenger
        (2.5f, 2.30f, 7.8f),   // 1: truck
        (2.6f, 3.10f, 12.5f),  // 2: bus
        (0.9f, 1.20f, 2.3f),   // 3: motorcycle
    };

    // ── Baz ton (HSV hue) arac tipine gore ───────────────────────────────
    private static readonly float[] BASE_HUE = {
        -1f,    // 0: passenger — rastgele renk (tam yelpazen)
        0.07f,  // 1: truck     — kahverengi/turuncu ton
        0.12f,  // 2: bus       — sari/turuncu ton
        0.75f,  // 3: motorcycle— mor/pembe ton
    };

    // ── Kabin rengi karanlik pencere simulasyonu ──────────────────────────
    private static readonly Color CABIN_TINT = new Color(0.14f, 0.15f, 0.18f);
    private const float CABIN_TINT_AMT = 0.40f;  // 0=tam araç rengi, 1=tam siyah

    // ── Dead-reckoning sabiti ─────────────────────────────────────────────
    // kPos = 1 - exp(-TAU_POS * dt)  — pozisyon duzeltme hizi
    // kRot = 1 - exp(-TAU_ROT * dt)  — rotasyon duzeltme hizi
    private const float TAU_POS = 15f;
    private const float TAU_ROT = 10f;

    // ── Ic veri yapisi ────────────────────────────────────────────────────
    private class VehicleGO
    {
        public GameObject root;
        public int        typeIdx;
        public Vector3    velocity;    // m/s, dead-reckoning icin
        public bool       justSpawned; // ilk gozukme → snap, lerp degil
    }

    // GC-free iterasyon: Dictionary ve ona paralel List
    private readonly Dictionary<string, VehicleGO> _pool     = new Dictionary<string, VehicleGO>(128);
    private readonly List<string>                   _poolKeys = new List<string>(128);
    private readonly HashSet<string>                _active   = new HashSet<string>(128);

    private VehicleData[] _pending;

    // ── Dis API ──────────────────────────────────────────────────────────

    public void UpdateVehicles(VehicleData[] data)
    {
        _pending = data;
    }

    // ── Her frame ────────────────────────────────────────────────────────

    private void Update()
    {
        if (_pending == null)
        {
            ActiveCount = 0;
            return;
        }

        float dt   = Time.deltaTime;
        float kPos = 1f - Mathf.Exp(-TAU_POS * dt);   // frame-rate bagimsiz
        float kRot = 1f - Mathf.Exp(-TAU_ROT * dt);

        _active.Clear();

        foreach (var v in _pending)
        {
            if (string.IsNullOrEmpty(v.i)) continue;
            _active.Add(v.i);

            // Pool'da yoksa yeni olustur
            if (!_pool.TryGetValue(v.i, out var vgo))
            {
                vgo = Spawn(v.c, v.i);
                _pool[v.i]   = vgo;
                _poolKeys.Add(v.i);
            }

            // SUMO hedef pozisyon/rotasyon
            float      halfH = SIZE[vgo.typeIdx].h * 0.5f;
            Vector3    tPos  = new Vector3(v.x, halfH, v.y);
            // SUMO: 0=Kuzey, 90=Dogu, saat yonu → Unity Y ekseni
            Quaternion tRot  = Quaternion.Euler(0f, v.a, 0f);

            var tr = vgo.root.transform;

            if (vgo.justSpawned)
            {
                // ── Ilk gozukme: snap (ucma efekti yok) ──────────────
                tr.SetPositionAndRotation(tPos, tRot);
                vgo.velocity    = (tRot * Vector3.forward) * v.s;
                vgo.justSpawned = false;
                vgo.root.SetActive(true);
            }
            else
            {
                // ── Dead-reckoning: ileri tahmin + SUMO'ya dogru duzeltme ──
                // Arac son bilinen hiz yonunde ilerler; hedeften sapma lerp ile duzeltilir
                vgo.velocity   = (tRot * Vector3.forward) * v.s;
                Vector3 predicted = tr.position + vgo.velocity * dt;
                tr.position = Vector3.Lerp(predicted, tPos, kPos);
                tr.rotation = Quaternion.Slerp(tr.rotation, tRot, kRot);

                if (!vgo.root.activeSelf)
                    vgo.root.SetActive(true);
            }
        }

        // ── Simden cikan araclari gizle (GC-free: List iterasyonu) ───────
        for (int i = 0; i < _poolKeys.Count; i++)
        {
            string key = _poolKeys[i];
            if (!_active.Contains(key))
            {
                var vgo = _pool[key];
                if (vgo.root.activeSelf)
                {
                    vgo.root.SetActive(false);
                    vgo.velocity    = Vector3.zero;
                    vgo.justSpawned = true;   // tekrar gelince snap yapsin
                }
            }
        }

        ActiveCount = _active.Count;
    }

    // ── GameObject uret ──────────────────────────────────────────────────

    private VehicleGO Spawn(string cls, string id)
    {
        int   t   = TypeIdx(cls);
        (float w, float h, float l) = SIZE[t];
        Color col = PickColor(id, t);

        var root = new GameObject("V_" + id);
        root.transform.SetParent(transform, false);
        root.SetActive(false);

        float bodyH  = h * 0.58f;
        float cabinH = h * 0.50f;
        // Kabin hafif arkaya kayik (on kaplama alanini genisletir)
        float cabinZ = -l * 0.06f;

        // ── Govde ─────────────────────────────────────────────────────
        var bodyGO = MakeBox(root.transform,
            new Vector3(w, bodyH, l),
            new Vector3(0f, bodyH * 0.5f, 0f),
            MakeMat(col, 0.45f, 0.15f));

        // ── Kabin (koyu pencere rengi) ────────────────────────────────
        Color cabinCol = Color.Lerp(col, CABIN_TINT, CABIN_TINT_AMT);
        var cabGO = MakeBox(root.transform,
            new Vector3(w * 0.80f, cabinH, l * 0.50f),
            new Vector3(0f, bodyH + cabinH * 0.5f, cabinZ),
            MakeMat(cabinCol, 0.70f, 0.05f));

        return new VehicleGO
        {
            root        = root,
            typeIdx     = t,
            velocity    = Vector3.zero,
            justSpawned = true,
        };
    }

    // ── Yardimcilar ──────────────────────────────────────────────────────

    private static int TypeIdx(string cls)
    {
        if (string.IsNullOrEmpty(cls)) return 0;
        if (cls.Contains("truck")    || cls.Contains("delivery") ||
            cls.Contains("heavy")    || cls.Contains("trailer"))   return 1;
        if (cls.Contains("bus")      || cls.Contains("coach")    ||
            cls.Contains("tram")     || cls.Contains("trolleybus")) return 2;
        if (cls.Contains("motor")    || cls.Contains("moped")    ||
            cls.Contains("bike")     || cls.Contains("bicycle"))   return 3;
        return 0;
    }

    /// <summary>
    /// Araç ID hash'inden belirleyici ama cesitli renk secer.
    /// Passenger: tüm yelpaze. Truck: sicak/koyu. Bus: sari. Motorcycle: canli.
    /// </summary>
    private static Color PickColor(string id, int typeIdx)
    {
        int hash = id == null ? 17 : (id.GetHashCode() ^ (id.GetHashCode() >> 16));
        hash = Mathf.Abs(hash);

        float hue, sat, val;

        if (BASE_HUE[typeIdx] < 0f)
        {
            // Passenger: tum renk yelpazesi, yuksek saturation
            hue = (hash % 360) / 360f;
            sat = 0.62f + (hash % 30) / 100f;  // 0.62 – 0.92
            val = 0.72f + (hash % 22) / 100f;  // 0.72 – 0.94
        }
        else
        {
            // Diger tipler: baz ton etrafinda sinirli varyasyon
            hue = Mathf.Repeat(BASE_HUE[typeIdx] + (hash % 24 - 12) / 360f, 1f);
            sat = 0.55f + (hash % 35) / 100f;
            val = 0.65f + (hash % 25) / 100f;
        }

        return Color.HSVToRGB(hue, sat, val);
    }

    private static GameObject MakeBox(Transform parent,
                                       Vector3 scale, Vector3 localPos,
                                       Material mat)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        Object.Destroy(go.GetComponent<BoxCollider>());
        go.transform.SetParent(parent, false);
        go.transform.localScale    = scale;
        go.transform.localPosition = localPos;
        go.GetComponent<Renderer>().material = mat;
        return go;
    }

    private static Material MakeMat(Color col, float glossiness, float metallic)
    {
        var mat = new Material(Shader.Find("Standard"));
        mat.color = col;
        mat.SetFloat("_Glossiness", glossiness);
        mat.SetFloat("_Metallic",   metallic);
        return mat;
    }
}
