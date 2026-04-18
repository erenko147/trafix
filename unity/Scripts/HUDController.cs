using System.Collections.Generic;
using UnityEngine;

public class HUDController : MonoBehaviour
{
    private int    _step;
    private int    _vehicles;
    private bool   _connected;
    private float  _fps;
    private float  _fpsTimer;
    private int    _fpsCount;
    private bool   _show = true;
    private readonly List<string> _tlsLines = new List<string>();

    private GUIStyle _styleTitle;
    private GUIStyle _styleLabel;
    private bool     _stylesReady;

    public void SetData(int step, int vCount, bool conn, TLSData[] tlsArr)
    {
        _step      = step;
        _vehicles  = vCount;
        _connected = conn;

        _tlsLines.Clear();
        if (tlsArr == null) return;
        foreach (var t in tlsArr)
        {
            string name;
            switch (t.p)
            {
                case 0: name = "K/G Yesil"; break;
                case 2: name = "D/B Yesil"; break;
                case 1:
                case 3: name = "Sari";      break;
                default: name = "Faz " + t.p; break;
            }
            _tlsLines.Add(t.i + ": " + name);
        }
    }

    private void Update()
    {
        _fpsCount++;
        _fpsTimer += Time.unscaledDeltaTime;
        if (_fpsTimer >= 0.5f)
        {
            _fps      = _fpsCount / _fpsTimer;
            _fpsCount = 0;
            _fpsTimer = 0f;
        }
        try { if (Input.GetKeyDown(KeyCode.H)) _show = !_show; } catch { }
    }

    private void OnGUI()
    {
        if (!_show) return;
        BuildStyles();

        float w = 245f;
        float h = 190f + _tlsLines.Count * 20f;

        // Panel arkaplan
        GUI.color = new Color(0.04f, 0.04f, 0.10f, 0.88f);
        GUI.DrawTexture(new Rect(8, 8, w, h), Texture2D.whiteTexture);
        GUI.color = Color.white;

        GUILayout.BeginArea(new Rect(16, 16, w - 16, h - 16));

        GUILayout.Label("TraFix 3D Simulasyon", _styleTitle);
        GUILayout.Space(3);

        Row("Adim :",    _step.ToString());
        Row("Arac :",    _vehicles.ToString());
        Row("FPS  :",    ((int)_fps).ToString(),
            _fps >= 30 ? new Color(0.2f,1f,0.4f) : new Color(1f,0.5f,0.1f));
        Row("SUMO :",    _connected ? "Bagli" : "Bekleniyor...",
            _connected ? new Color(0.2f,1f,0.4f) : new Color(1f,0.3f,0.3f));

        if (_tlsLines.Count > 0)
        {
            GUILayout.Space(6);
            GUILayout.Label("Trafik Isiklari:", _styleTitle);
            foreach (var l in _tlsLines)
                GUILayout.Label(l, _styleLabel);
        }

        GUILayout.Space(8);
        GUILayout.Label("[H] gizle/goster | [WASD] hareket", _styleLabel);
        GUILayout.Label("[RMB+Sur] dondur | [Scroll] zoom", _styleLabel);
        GUILayout.Label("[Q/E] asagi/yukari | [Shift] hizli", _styleLabel);

        GUILayout.EndArea();
    }

    private void Row(string label, string val, Color? col = null)
    {
        GUILayout.BeginHorizontal();
        GUILayout.Label(label, _styleLabel, GUILayout.Width(55));
        var old = GUI.contentColor;
        GUI.contentColor = col ?? new Color(0.9f, 0.9f, 0.9f);
        GUILayout.Label(val, _styleLabel);
        GUI.contentColor = old;
        GUILayout.EndHorizontal();
    }

    private void BuildStyles()
    {
        if (_stylesReady) return;
        _stylesReady = true;

        _styleTitle = new GUIStyle(GUI.skin.label)
        {
            fontSize  = 13,
            fontStyle = FontStyle.Bold
        };
        _styleTitle.normal.textColor = new Color(0.4f, 0.85f, 1f);

        _styleLabel = new GUIStyle(GUI.skin.label) { fontSize = 11 };
        _styleLabel.normal.textColor = new Color(0.88f, 0.88f, 0.88f);
    }
}
