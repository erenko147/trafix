using System;
using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class SumoWebSocketClient : MonoBehaviour
{
    [SerializeField] private string wsUrl = "ws://127.0.0.1:8765";

    public event Action<InitMessage>  OnInitReceived;
    public event Action<FrameMessage> OnFrameReceived;
    public bool IsConnected { get; private set; }

    private ClientWebSocket             _ws;
    private CancellationTokenSource    _cts;
    private readonly ConcurrentQueue<string> _queue = new ConcurrentQueue<string>();

    private void Start()
    {
        _cts = new CancellationTokenSource();
        ConnectLoop();
    }

    private async void ConnectLoop()
    {
        while (!_cts.IsCancellationRequested)
        {
            _ws = new ClientWebSocket();
            try
            {
                await _ws.ConnectAsync(new Uri(wsUrl), _cts.Token);
                IsConnected = true;
                await ReceiveLoop();
            }
            catch (Exception) { IsConnected = false; }

            try { await Task.Delay(2000, _cts.Token); }
            catch (OperationCanceledException) { break; }
        }
    }

    private async Task ReceiveLoop()
    {
        byte[]        buf = new byte[262144];   // 256 KB
        StringBuilder sb  = new StringBuilder();

        while (_ws.State == WebSocketState.Open && !_cts.IsCancellationRequested)
        {
            sb.Clear();
            WebSocketReceiveResult res;
            try
            {
                do
                {
                    res = await _ws.ReceiveAsync(new ArraySegment<byte>(buf), _cts.Token);
                    if (res.MessageType == WebSocketMessageType.Close) { IsConnected = false; return; }
                    sb.Append(Encoding.UTF8.GetString(buf, 0, res.Count));
                }
                while (!res.EndOfMessage);
            }
            catch (Exception) { IsConnected = false; return; }

            _queue.Enqueue(sb.ToString());
        }
        IsConnected = false;
    }

    private void Update()
    {
        while (_queue.TryDequeue(out string raw))
        {
            try { Dispatch(raw); }
            catch (Exception e) { Debug.LogWarning("[WS] Parse hatasi: " + e.Message); }
        }
    }

    private void Dispatch(string raw)
    {
        // Bridge sends plain JSON with "type":"init" or "type":"frame"
        if (raw.Contains("\"type\":\"init\""))
        {
            var init = JsonUtility.FromJson<InitMessage>(raw);
            if (init != null) OnInitReceived?.Invoke(init);
            return;
        }

        var frame = JsonUtility.FromJson<FrameMessage>(raw);
        if (frame != null) OnFrameReceived?.Invoke(frame);
    }

    private void OnDestroy()
    {
        _cts?.Cancel();
        _ws?.Abort();
    }
}
