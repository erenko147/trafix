using UnityEngine;

/// <summary>
/// Kavsak baslari icin L-sekilli trafik isigi direkleri.
/// Her direk yol kenarina yerlestirilmis, yatay kol serit uzerine uzanir,
/// sinyal kutusu kol ucundan asagiya sarkarak trafiğe bakar.
/// SUMO state string ile gercek zamanli senkronize edilir.
/// </summary>
public class TrafficLightController : MonoBehaviour
{
    // ── Direk konumlari: her yol girisinin sol kenari, kavsak merkezine gore ──
    // Siralama: Kuzey, Dogu, Guney, Bati
    private static readonly Vector3[] POLES =
    {
        new Vector3(-5f,  0f,  13f),  // 0: Kuzey
        new Vector3( 13f, 0f,   5f),  // 1: Dogu
        new Vector3(  5f, 0f, -13f),  // 2: Guney
        new Vector3(-13f, 0f,  -5f),  // 3: Bati
    };

    // ── Her diregin yuz yonu: yerel +Z yaklasan trafiğe bakar ────────────────
    private static readonly float[] FACE_Y = { 0f, 90f, 180f, -90f };

    // ── Geometri ──────────────────────────────────────────────────────────────
    private const float POLE_H   = 5.8f;   // direk yuksekligi (m)
    private const float ARM_LEN  = 3.2f;   // yatay kol uzunlugu, yerel +X
    private const float ARM_SZ   = 0.14f;  // kol kesiti
    private const float BOX_W    = 0.60f;  // sinyal kutusu genisligi
    private const float BOX_H    = 2.00f;  // sinyal kutusu yuksekligi
    private const float BOX_D    = 0.48f;  // sinyal kutusu derinligi
    private const float LAMP_R   = 0.18f;  // lamba yariçapi
    private const float LAMP_OFF = 0.10f;  // lambalar kutu yuzunden ne kadar cikiyor

    // ── Lamba renkleri ────────────────────────────────────────────────────────
    private static readonly Color C_RED    = new Color(1.00f, 0.05f, 0.05f);
    private static readonly Color C_YELLOW = new Color(1.00f, 0.78f, 0.00f);
    private static readonly Color C_GREEN  = new Color(0.05f, 1.00f, 0.15f);
    private static readonly Color C_DIM    = new Color(0.08f, 0.08f, 0.08f);

    private const float GLOW_ON  = 4.0f;
    private const float GLOW_OFF = 0.06f;

    // ── Materyal referanslari (4 direk × 3 renk) ─────────────────────────────
    private Material[] _redMats;
    private Material[] _yellowMats;
    private Material[] _greenMats;

    // ── Degisim tespiti ───────────────────────────────────────────────────────
    private int    _lastPhase = -99;
    private string _lastState = null;

    // ── Fabrika ───────────────────────────────────────────────────────────────

    public static TrafficLightController Build(GameObject parent)
    {
        var c = parent.AddComponent<TrafficLightController>();
        c.CreatePoles();
        return c;
    }

    // ── Guncelleme (SimulationManager'dan cagrili) ────────────────────────────

    public void UpdatePhase(int phase, string stateStr)
    {
        bool phaseChanged = phase    != _lastPhase;
        bool stateChanged = stateStr != _lastState;
        if (!phaseChanged && !stateChanged) return;

        _lastPhase = phase;
        _lastState = stateStr;

        if (!string.IsNullOrEmpty(stateStr) && stateStr.Length >= 4)
            ApplyStateString(stateStr);
        else
            ApplyPhaseFallback(phase);
    }

    // ── SUMO state string senkronizasyonu ─────────────────────────────────────

    private void ApplyStateString(string s)
    {
        int total = s.Length;
        int chunk = Mathf.Max(1, total / 4);

        for (int pole = 0; pole < 4; pole++)
        {
            int  start = pole * chunk;
            int  end   = Mathf.Min(start + chunk, total);
            char dom   = DominantChar(s, start, end);

            SetLamp(_redMats[pole],    C_RED,    dom == 'r');
            SetLamp(_yellowMats[pole], C_YELLOW, dom == 'y');
            SetLamp(_greenMats[pole],  C_GREEN,  dom == 'G');
        }
    }

    // ── Faz numarasina gore yedek mantik ──────────────────────────────────────

    private void ApplyPhaseFallback(int phase)
    {
        bool yellow  = (phase & 1) == 1;
        bool nsGreen = phase == 0 || phase == 4 || phase == 6;
        bool ewGreen = phase == 2;

        for (int pole = 0; pole < 4; pole++)
        {
            bool isNS = (pole == 0 || pole == 2);
            bool r, y, g;

            if (yellow)
            {
                r = false; y = true; g = false;
            }
            else if (isNS && nsGreen)
            {
                r = false; y = false; g = true;
            }
            else if (!isNS && ewGreen)
            {
                r = false; y = false; g = true;
            }
            else
            {
                r = true; y = false; g = false;
            }

            SetLamp(_redMats[pole],    C_RED,    r);
            SetLamp(_yellowMats[pole], C_YELLOW, y);
            SetLamp(_greenMats[pole],  C_GREEN,  g);
        }
    }

    // ── Dominant karakter: Y > G > R ──────────────────────────────────────────

    private static char DominantChar(string s, int start, int end)
    {
        bool hasY = false, hasG = false;
        for (int i = start; i < end; i++)
        {
            char c = s[i];
            if      (c == 'y' || c == 'Y' || c == 'u') hasY = true;
            else if (c == 'G' || c == 'g')              hasG = true;
        }
        if (hasY) return 'y';
        if (hasG) return 'G';
        return 'r';
    }

    // ── Lamba emisyon guncelleme ──────────────────────────────────────────────

    private static void SetLamp(Material mat, Color col, bool on)
    {
        mat.color = on ? col : C_DIM;
        mat.SetColor("_EmissionColor", col * (on ? GLOW_ON : GLOW_OFF));
        mat.EnableKeyword("_EMISSION");
    }

    // ── Direk insasi ──────────────────────────────────────────────────────────

    private void CreatePoles()
    {
        _redMats    = new Material[4];
        _yellowMats = new Material[4];
        _greenMats  = new Material[4];

        Color poleCol = new Color(0.20f, 0.20f, 0.22f);
        Color boxCol  = new Color(0.10f, 0.10f, 0.12f);

        for (int i = 0; i < 4; i++)
        {
            // Kok nesne: yola gore konum + yuz yonu
            var root = new GameObject("TLS_" + i);
            root.transform.SetParent(transform, false);
            root.transform.localPosition = POLES[i];
            root.transform.localRotation = Quaternion.Euler(0f, FACE_Y[i], 0f);

            // ── Dikey silindir direk ──────────────────────────────────────────
            // Cylinder yerel yuksekligi 2 birim → scale.y = POLE_H / 2
            MakePrim(PrimitiveType.Cylinder, root.transform,
                new Vector3(0f, POLE_H * 0.5f, 0f),
                new Vector3(0.18f, POLE_H * 0.5f, 0.18f),
                poleCol);

            // ── Yatay kol (yerel +X yonunde, direk tepesinde) ────────────────
            MakePrim(PrimitiveType.Cube, root.transform,
                new Vector3(ARM_LEN * 0.5f, POLE_H, 0f),
                new Vector3(ARM_LEN, ARM_SZ, ARM_SZ),
                poleCol);

            // ── Sinyal kutusu (kol ucunda, asagiya sarkar) ───────────────────
            float boxCY = POLE_H - BOX_H * 0.5f;
            MakePrim(PrimitiveType.Cube, root.transform,
                new Vector3(ARM_LEN, boxCY, 0f),
                new Vector3(BOX_W, BOX_H, BOX_D),
                boxCol);

            // ── Lambalar: kutu on yuzune (yerel +Z = trafiğe bakar) ──────────
            float lz = BOX_D * 0.5f + LAMP_OFF;
            float yR = POLE_H - BOX_H * 0.22f;   // kirmizi — ust
            float yY = POLE_H - BOX_H * 0.50f;   // sari    — orta
            float yG = POLE_H - BOX_H * 0.78f;   // yesil   — alt

            _redMats[i]    = MakeLamp(root.transform, new Vector3(ARM_LEN, yR, lz), C_RED);
            _yellowMats[i] = MakeLamp(root.transform, new Vector3(ARM_LEN, yY, lz), C_YELLOW);
            _greenMats[i]  = MakeLamp(root.transform, new Vector3(ARM_LEN, yG, lz), C_GREEN);
        }

        // Baslangic durumu
        ApplyPhaseFallback(0);
    }

    // ── Primitif yardimcilar ──────────────────────────────────────────────────

    private static Material MakeLamp(Transform parent, Vector3 localPos, Color baseCol)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        go.transform.SetParent(parent, false);
        go.transform.localPosition = localPos;
        go.transform.localScale    = Vector3.one * LAMP_R;
        Object.Destroy(go.GetComponent<SphereCollider>());

        var mat = new Material(Shader.Find("Standard"));
        mat.SetFloat("_Glossiness", 0.85f);
        mat.EnableKeyword("_EMISSION");
        go.GetComponent<Renderer>().material = mat;

        SetLamp(mat, baseCol, false);
        return go.GetComponent<Renderer>().material;
    }

    private static void MakePrim(PrimitiveType type, Transform parent,
                                  Vector3 pos, Vector3 scale, Color col)
    {
        var go = GameObject.CreatePrimitive(type);
        go.transform.SetParent(parent, false);
        go.transform.localPosition = pos;
        go.transform.localScale    = scale;
        go.GetComponent<Renderer>().sharedMaterial =
            new Material(Shader.Find("Standard")) { color = col };

        var c2 = go.GetComponent<Collider>();
        if (c2 != null) Object.Destroy(c2);
    }
}
