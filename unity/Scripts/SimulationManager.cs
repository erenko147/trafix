using System.Collections.Generic;
using UnityEngine;

[DefaultExecutionOrder(-10)]
public class SimulationManager : MonoBehaviour
{
    // ── Alt sistemler ─────────────────────────────────────────────────────
    private SumoWebSocketClient  _ws;
    private TrafficNetworkBuilder _network;
    private VehiclePool           _vehicles;
    private HUDController         _hud;

    private readonly Dictionary<string, TrafficLightController> _tls =
        new Dictionary<string, TrafficLightController>();

    // Frame tampon (thread guvenli)
    private FrameMessage _lastFrame;
    private bool         _newFrame;

    private bool _sceneBuilt;

    // ── Baslangic ─────────────────────────────────────────────────────────

    private void Awake()
    {
        // Yol agi (init mesaji gelince doldurulacak)
        var netGO = new GameObject("Network");
        netGO.transform.SetParent(transform, false);
        _network = netGO.AddComponent<TrafficNetworkBuilder>();

        // Arac havuzu
        var poolGO = new GameObject("VehiclePool");
        poolGO.transform.SetParent(transform, false);
        _vehicles = poolGO.AddComponent<VehiclePool>();

        // HUD
        var hudGO = new GameObject("HUD");
        hudGO.transform.SetParent(transform, false);
        _hud = hudGO.AddComponent<HUDController>();

        // WebSocket
        _ws = gameObject.AddComponent<SumoWebSocketClient>();
        _ws.OnInitReceived  += OnInit;
        _ws.OnFrameReceived += OnFrame;

        SetupLighting();
        SetupCamera(100f, 50f, 150f);   // varsayilan, init sonra guncellenir
    }

    // ── Init mesaji: sahne kur ────────────────────────────────────────────

    private void OnInit(InitMessage data)
    {
        if (_sceneBuilt) return;
        _sceneBuilt = true;

        // Yol agini olustur
        _network.Build(data);

        // Trafik isikli kavsaklara TLS controller ekle
        if (data.junctions != null)
        {
            foreach (var j in data.junctions)
            {
                if (j.tls != 1) continue;
                if (!_network.Intersections.TryGetValue(j.id, out var go)) continue;
                var ctrl = TrafficLightController.Build(go);
                _tls[j.id] = ctrl;
            }
        }

        // Kamerayı harita sınırlarına göre konumlandır
        float cx = (_network.XMin + _network.XMax) * 0.5f;
        float cy = (_network.YMin + _network.YMax) * 0.5f;
        float span = Mathf.Max(_network.XMax - _network.XMin,
                               _network.YMax - _network.YMin);
        SetupCamera(cx, cy, span * 0.75f);

        Debug.Log($"[SIM] Sahne hazir: {data.junctions?.Length ?? 0} kavsak, " +
                  $"{data.edges?.Length ?? 0} kenar, {_tls.Count} TLS");
    }

    // ── Frame mesaji: guncelle ────────────────────────────────────────────

    private void OnFrame(FrameMessage frame)
    {
        _lastFrame = frame;
        _newFrame  = true;
    }

    private void Update()
    {
        if (!_newFrame || _lastFrame == null) return;
        _newFrame = false;

        var f = _lastFrame;

        _vehicles.UpdateVehicles(f.v);

        if (f.tls != null)
            foreach (var t in f.tls)
                if (_tls.TryGetValue(t.i, out var ctrl))
                    ctrl.UpdatePhase(t.p, t.s);

        // f.v.Length: anlik araç sayisi (ActiveCount bir frame gec guncellenir)
        int vCount = f.v != null ? f.v.Length : 0;
        _hud.SetData(f.t, vCount, _ws.IsConnected, f.tls);
    }

    private void OnDestroy()
    {
        if (_ws != null)
        {
            _ws.OnInitReceived  -= OnInit;
            _ws.OnFrameReceived -= OnFrame;
        }
    }

    // ── Yardimcilar ───────────────────────────────────────────────────────

    private void SetupCamera(float cx, float cy, float height)
    {
        Camera cam = Camera.main;
        if (cam == null)
        {
            var go = new GameObject("MainCamera");
            go.tag = "MainCamera";   // Camera.main bunu arar
            cam    = go.AddComponent<Camera>();
            go.AddComponent<AudioListener>();
        }

        // CameraRig ekle (yoksa)
        if (cam.GetComponent<CameraRig>() == null)
            cam.gameObject.AddComponent<CameraRig>();

        // Harita uzerine bak
        cam.transform.position = new Vector3(cx, height, cy - height * 0.5f);
        cam.transform.LookAt(new Vector3(cx, 0f, cy));

        cam.backgroundColor = new Color(0.52f, 0.80f, 0.97f);
        cam.clearFlags      = CameraClearFlags.SolidColor;
        cam.farClipPlane    = 3000f;
    }

    private void SetupLighting()
    {
        var sun = GameObject.Find("Directional Light");
        if (sun == null) sun = new GameObject("Directional Light");
        var l = sun.GetComponent<Light>() ?? sun.AddComponent<Light>();
        l.type      = LightType.Directional;
        l.intensity = 1.3f;
        l.color     = new Color(1f, 0.96f, 0.86f);
        l.shadows   = LightShadows.Soft;
        sun.transform.rotation = Quaternion.Euler(48f, -35f, 0f);

        RenderSettings.ambientLight = new Color(0.38f, 0.42f, 0.52f);
        RenderSettings.fogColor     = new Color(0.60f, 0.75f, 0.90f);
        RenderSettings.fog          = false;
    }
}
