using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// SUMO harita verisinden 3D yol agi olusturur.
/// Tum yol parcalari tek combined mesh'e birlestirilerek 1 draw call'a indirgenir.
/// Serit cizgileri: kenar=beyaz, orta=sari (cift yonlu) / yok (tek yonlu).
/// </summary>
public class TrafficNetworkBuilder : MonoBehaviour
{
    // ── Dis erisim ────────────────────────────────────────────────────────
    public Dictionary<string, GameObject> Intersections { get; } =
        new Dictionary<string, GameObject>();

    public float XMin { get; private set; } = -50f;
    public float YMin { get; private set; } = -50f;
    public float XMax { get; private set; } = 250f;
    public float YMax { get; private set; } = 150f;

    // ── Renkler ───────────────────────────────────────────────────────────
    private static readonly Color COL_GROUND  = new Color(0.15f, 0.38f, 0.11f); // koyu cimen
    private static readonly Color COL_ROAD    = new Color(0.22f, 0.22f, 0.22f); // asphalt
    private static readonly Color COL_INTER   = new Color(0.30f, 0.30f, 0.30f); // kavsak
    private static readonly Color COL_WHITE   = new Color(0.94f, 0.94f, 0.92f); // beyaz cizgi
    private static readonly Color COL_YELLOW  = new Color(1.00f, 0.85f, 0.00f); // sari cizgi

    // ── Y yukseklikleri (Z-fighting olmamasi icin katmanli) ───────────────
    private const float Y_GROUND = 0.00f;
    private const float Y_ROAD   = 0.08f;
    private const float Y_INTER  = 0.09f;
    private const float Y_LINE   = 0.10f;

    // ── Kavsak boyutu ─────────────────────────────────────────────────────
    private const float INTER_SIZE = 18f;

    public void Build(InitMessage data)
    {
        if (data == null) return;

        // Harita sinirlari
        if (data.bounds != null && data.bounds.Length >= 4)
        {
            XMin = data.bounds[0]; YMin = data.bounds[1];
            XMax = data.bounds[2]; YMax = data.bounds[3];
        }

        // Zemin
        MakeGround();

        // Kavsak plakalari
        if (data.junctions != null)
            foreach (var j in data.junctions)
                MakeIntersection(j);

        // Yollar
        if (data.edges != null)
            foreach (var e in data.edges)
                MakeEdge(e);
    }

    // ── Zemin ─────────────────────────────────────────────────────────────

    private void MakeGround()
    {
        float w = XMax - XMin + 200f;
        float h = YMax - YMin + 200f;
        float cx = (XMin + XMax) * 0.5f;
        float cy = (YMin + YMax) * 0.5f;

        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = "Ground";
        go.transform.SetParent(transform, false);
        go.transform.localPosition = new Vector3(cx, Y_GROUND - 0.5f, cy);
        go.transform.localScale    = new Vector3(w, 1f, h);
        go.GetComponent<Renderer>().sharedMaterial =
            new Material(Shader.Find("Standard")) { color = COL_GROUND };
        Object.Destroy(go.GetComponent<BoxCollider>());
    }

    // ── Kavsak plakasi ────────────────────────────────────────────────────

    private void MakeIntersection(JunctionInfo j)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = "J_" + j.id;
        go.transform.SetParent(transform, false);
        go.transform.localPosition = new Vector3(j.x, Y_INTER, j.y);
        go.transform.localScale    = new Vector3(INTER_SIZE, 0.01f, INTER_SIZE);
        go.GetComponent<Renderer>().sharedMaterial =
            new Material(Shader.Find("Standard")) { color = COL_INTER };
        Object.Destroy(go.GetComponent<BoxCollider>());
        Intersections[j.id] = go;
    }

    // ── Yol segmenti ──────────────────────────────────────────────────────

    private void MakeEdge(EdgeInfo e)
    {
        float[] sh = e.sh;
        if (sh == null || sh.Length < 4) return;

        int pts = sh.Length / 2;
        for (int i = 0; i < pts - 1; i++)
        {
            float x0 = sh[i * 2],     y0 = sh[i * 2 + 1];
            float x1 = sh[(i+1)*2],   y1 = sh[(i+1)*2 + 1];

            Vector3 a = new Vector3(x0, Y_ROAD, y0);
            Vector3 b = new Vector3(x1, Y_ROAD, y1);

            float len = Vector3.Distance(a, b);
            if (len < 0.01f) continue;

            Vector3 mid = (a + b) * 0.5f;
            Vector3 dir = (b - a).normalized;

            var seg = GameObject.CreatePrimitive(PrimitiveType.Cube);
            seg.name = "E_" + e.id + "_" + i;
            seg.transform.SetParent(transform, false);
            seg.transform.localPosition = mid;
            seg.transform.localScale    = new Vector3(e.w, 0.01f, len);
            seg.transform.localRotation = Quaternion.LookRotation(dir, Vector3.up);
            seg.GetComponent<Renderer>().sharedMaterial =
                new Material(Shader.Find("Standard")) { color = COL_ROAD };
            Object.Destroy(seg.GetComponent<BoxCollider>());
        }
    }
}
