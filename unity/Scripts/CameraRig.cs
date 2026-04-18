using UnityEngine;

/// <summary>
/// Kamera kontrol: WASD/oklar=hareket, SagTus+Sur=dondur,
/// Scroll=zoom, Q/E=asagi/yukari, Shift=hizli.
/// Her iki Input System modunda calisir (Legacy + New Input System).
/// </summary>
[RequireComponent(typeof(Camera))]
public class CameraRig : MonoBehaviour
{
    [SerializeField] private float moveSpeed   = 60f;
    [SerializeField] private float scrollSpeed = 40f;
    [SerializeField] private float dragSens    = 0.15f;
    [SerializeField] private float minH        = 3f;
    [SerializeField] private float maxH        = 600f;

    private float _yaw;
    private float _pitch = 50f;
    private bool  _drag;
    private Vector3 _lastMouse;

    private void Start()
    {
        var e  = transform.eulerAngles;
        _yaw   = e.y;
        _pitch = Mathf.Clamp(e.x, 5f, 85f);
    }

    private void Update()
    {
        DragRotate();
        Move();
        Scroll();
    }

    // ── Sag-tus surukleme ile dondur ─────────────────────────────────────────

    private void DragRotate()
    {
        try
        {
            if (Input.GetMouseButtonDown(1)) { _drag = true;  _lastMouse = Input.mousePosition; }
            if (Input.GetMouseButtonUp(1))     _drag = false;
            if (!_drag) return;

            Vector3 delta = Input.mousePosition - _lastMouse;
            _lastMouse = Input.mousePosition;
            _yaw   += delta.x * dragSens;
            _pitch -= delta.y * dragSens;
            _pitch  = Mathf.Clamp(_pitch, 5f, 85f);
            transform.rotation = Quaternion.Euler(_pitch, _yaw, 0f);
        }
        catch { }
    }

    // ── WASD / Ok tuslar ile hareket ─────────────────────────────────────────

    private void Move()
    {
        try
        {
            float spd = moveSpeed
                * (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift) ? 3f : 1f)
                * Time.deltaTime;

            Vector3 dir = Vector3.zero;
            if (Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.UpArrow))    dir += transform.forward;
            if (Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.DownArrow))  dir -= transform.forward;
            if (Input.GetKey(KeyCode.A) || Input.GetKey(KeyCode.LeftArrow))  dir -= transform.right;
            if (Input.GetKey(KeyCode.D) || Input.GetKey(KeyCode.RightArrow)) dir += transform.right;
            if (Input.GetKey(KeyCode.Q))                                      dir -= Vector3.up;
            if (Input.GetKey(KeyCode.E))                                      dir += Vector3.up;

            if (dir.sqrMagnitude < 0.001f) return;
            Vector3 np = transform.position + dir.normalized * spd;
            np.y = Mathf.Clamp(np.y, minH, maxH);
            transform.position = np;
        }
        catch { }
    }

    // ── Scroll ile zoom ──────────────────────────────────────────────────────

    private void Scroll()
    {
        try
        {
            float s = Input.GetAxis("Mouse ScrollWheel");
            if (Mathf.Abs(s) < 0.001f) return;
            Vector3 np = transform.position + transform.forward * (s * scrollSpeed);
            np.y = Mathf.Clamp(np.y, minH, maxH);
            transform.position = np;
        }
        catch { }
    }
}
