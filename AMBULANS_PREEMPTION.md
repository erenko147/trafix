# Acil Araç (Ambulans) Önceliği — Devir Dokümanı

> Bu dosya, ambulans önceliği çalışmasının **tam bağlamını** taşır. Kodla
> birlikte repo'da yaşar; herhangi bir geliştirici veya yeni bir Claude
> Code oturumu bunu okuyup kaldığı yerden devam edebilir.
> Branch: `Arda` · Repo: github.com/erenko147/trafix

## 1. Amaç

SUMO tabanlı TraFix trafik sinyal sistemine acil araç (ambulans)
önceliği eklendi. Ambulans bir kavşağa yaklaşınca o yön yeşile alınır,
diğer yönler kırmızı; araç geçtikten sonra AI kontrolü geri devralır.

## 2. Eren'in mimari kararları (önemli)

- **AI eğitilmedi.** Acil durum AI/Governor'a koyulmadı; SUMO/TraCI
  katmanında ele alındı.
- **Governor'a dokunulmadı.** Governor sadece AI faz seçimini kısıtlar
  (6 preset faz). "Ambulans yönünün tüm çıkışları yeşil" diye bir faz
  yok → Governor bu işi ifade edemez. Yanlış katman.
- **Ayrı modül istendi.** Preemption mantığı `run_sumo_live.py`'dan
  çıkarılıp bağımsız modüle taşındı (sim döngüsünden ayrık).
- **İki metrik istendi:** (a) acil araç kavşaktan kaç adımda geçti,
  (b) kaç araç ne kadar bekledi.

## 3. Yapılan değişiklikler

| Dosya | Durum | İçerik |
|---|---|---|
| `sumo/emergency_preemption.py` | YENİ | `EmergencyPreemptionController` — tüm state machine + metrik toplama |
| `sumo/run_sumo_live.py` | değişti | Gömülü mantık silindi; her adım `preempt.update()` çağrılıyor; tamamlanan metrikler backend'e POST |
| `backend/main.py` | değişti | `POST /emergency_event` + `GET /emergency_metrics` |
| `main.py` | değişti | `/emergency` route'u (ayrı ekran) |
| `frontend/emergency.html` | YENİ | Ayrı acil araç metrik ekranı |
| `frontend/dashboard.html` | değişti | Sayaç bug fix + harita SUMO grid'ine göre + △ Acil Araç sekmesi |
| `sumo/emergency_test.rou.xml` | YENİ | 5 ambulanslı demo senaryosu |
| `sumo/demo.sumocfg` | değişti | route-files'a emergency_test.rou.xml eklendi |

## 4. Modül mimarisi (`emergency_preemption.py`)

`EmergencyPreemptionController` 3 katman:

1. **Algılama** (`_closest_emergency`, `_emergency_on_edge`)
   - Şu an: SUMO/TraCI (`getLastStepVehicleIDs` + `getTypeID=="emergency"`)
   - Mesafe-tabanlı öncelik: `lane_len - lane_pos` en küçük = en yakın
2. **Karar** (state machine) — **taşınabilir çekirdek, değişmez**
   - `None → yellow → allred → green → return_yellow → None`
   - `green` aşamasında CLEAR / NEXT (ikinci ambulans) / TIMEOUT
   - Kritik: bitişte `setProgram(tls,"0")` ile orijinal program geri
     yüklenir, yoksa kavşak donar
3. **Aktüatör** (`_set_state`, `_restore_program`)
   - Şu an: `setRedYellowGreenState` / `setProgram`

Metrikler: her olay için `transit_steps`, `vehicles_waited`,
`total_wait_steps`, `avg_wait_steps`, `result` (cleared/timeout).
İnsan-okur etiketler: `junction_name` (K1–K5), `approach_dir`
(Kuzey/Güney/Doğu/Batı).

## 5. Sonraki hedef: CARLA + YOLO

Canlı SUMO'da kurulmayacak. Hedef: CARLA simülasyonu + YOLO ile
ambulans tespiti. Taşıma planı:

```
SUMO TraCI                  →  CARLA Python API
getLastStepVehicleIDs/...   →  kamera sensör frame + YOLO "ambulance" + ROI eşleme
setRedYellowGreenState      →  traffic_light.set_state / freeze(True)
setProgram("0")             →  traffic_light.freeze(False)
```

**Sadece ① Algılama ve ③ Aktüatör katmanları değişir.** ② Karar
(state machine) ve metrik mantığı aynen kalır. `run_sumo_live.py`'a
ve Governor'a dokunulmaz.

## 6. Test

```
python baslat.py
```
- Dashboard: http://127.0.0.1:8000/  (harita SUMO grid'iyle aynı)
- Acil ekran: http://127.0.0.1:8000/emergency  (△ sekmesi)
- Demo senaryo adımları: 30, 90, 160 (2 ambulans aynı anda), 230
- Log: `logs/decisions_live.log` → `[EMERGENCY ...]`, `[EMERGENCY METRIC]`

## 7. Kavşak topolojisi (map.net.xml — birebir)

```
K1(J0) ───── K3(J2) ───── K5(J4)     üst sıra (SUMO y=380)
  │            │
K2(J1) ───── K4(J3)                  alt sıra (SUMO y=160)
```
Bağlantılar: E0:K1-K2, E1:K1-K3, E2:K3-K4, E3:K2-K4, E4:K3-K5
Sıralama: K1=J0, K2=J1, K3=J2, K4=J3, K5=J4 (sorted tls_ids index+1)

## 8. Bilinen sınırlar / kararlar

- Kalıcı sıkışan ambulans: `MAX_EMERGENCY_GREEN` (60 adım) sonra
  timeout → AI geri alır (kasıtlı).
- Hizmet sürerken daha yakın yeni ambulans için kesilmez (kasıtlı).
- yellow/allred aşamalarında yeniden tespit yapılmaz (kendini düzeltir).
