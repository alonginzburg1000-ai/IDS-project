# IDS Runtime Web Dashboard

מערכת runtime עבור IDS דו-שלבי:

1. מודל בינארי: `normal` / `attack`
2. מודל רב-מחלקתי: `dos`, `probe`, `r2l`, `u2r`, שמופעל רק אם המודל הבינארי החזיר `attack`

השכבה הזו מרחיבה רק את ה-runtime וה-UI. היא לא משנה מודלים מאומנים, לא משנה artifacts, ולא משנה קוד אימון.

## מבנה תיקיות

```text
agent/
  __init__.py
  agent.py
server/
  __init__.py
  app.py
  config.py
  inference.py
  model_loader.py
  preprocessing.py
  storage.py
  static/
    dashboard.js
    style.css
  templates/
    dashboard.html
artifacts/
  binary_model_weights_best.npz
  multiclass_model_weights_best.npz
  binary_preprocess.npz
  multiclass_preprocess.npz
  multiclass_label_map.json
config.json
requirements.txt
README.md
```

## מה רץ במערכת

`server/app.py` מרים Flask Web Server עם Dashboard ו-Agent פנימי שניתן להפעיל ולעצור מתוך ה-Dashboard.

כשהשרת עולה, ההסנפה מתחילה במצב עצור. לחיצה על "התחל הסנפה" מפעילה Agent כ-background thread בתוך תהליך Flask. לחיצה על "עצור הסנפה" מסמנת ל-thread לעצור דרך `threading.Event`. יש הגנה מפני יצירת כמה threads במקביל.

ה-Agent משתמש ב-Scapy, לוכד תעבורה חיה, ומעביר לשרת raw network fields בלבד. הוא לא שולח raw packets ולא מייצר בעצמו את וקטור 122 הפיצ'רים.

Flask הוא מקור האמת ל-preprocessing ול-inference:

```text
raw packet fields
-> validate
-> derive service
-> derive flag
-> calculate land
-> fill missing features with defaults
-> one-hot alignment to feature_names
-> normalization from saved preprocess artifacts
-> binary inference
-> optional multiclass inference
-> store in memory
-> update dashboard APIs
```

## Endpoints

```text
GET  /
GET  /dashboard
GET  /health
POST /predict
GET  /api/traffic
GET  /api/suspicious
GET  /api/stats/attack-types
GET  /api/sniffer/status
POST /api/sniffer/start
POST /api/sniffer/stop
```

`/dashboard` הוא המסך הראשי.

ה-Dashboard כולל:

```text
כל התעבורה
תעבורה חשודה
סיווג תעבורה חשודה
כפתורי התחל הסנפה / עצור הסנפה
סטטוס הסנפה פעילה / הסנפה עצורה
```

העדכון החי נעשה ב-JavaScript polling כל שנייה. גרף העוגה משתמש ב-Chart.js דרך CDN.

## Artifacts

ה-runtime רק טוען את הקבצים האלה:

```text
artifacts/binary_model_weights_best.npz
artifacts/multiclass_model_weights_best.npz
artifacts/binary_preprocess.npz
artifacts/multiclass_preprocess.npz
artifacts/multiclass_label_map.json
```

הוא לא משנה אותם.

## NSL-KDD Approximation

המודלים אומנו על NSL-KDD connection-level records, אבל Agent חי רואה packet בודד בכל פעם.

בגרסה הזו מחושבים רק פיצ'רים שניתן להפיק מפקטה אחת:

```text
protocol_type -> protocol
service       -> derived from dst_port/src_port
flag          -> approximated from TCP flags
land          -> src_ip == dst_ip and src_port == dst_port
src_bytes     -> approximated from payload_len
wrong_fragment
urgent
```

פיצ'רים שדורשים היסטוריית חיבורים או חלון host נשארים `0` / default:

```text
duration
dst_bytes
count
srv_count
same_srv_rate
diff_srv_rate
dst_host_count
dst_host_srv_count
dst_host_same_srv_rate
dst_host_diff_srv_rate
dst_host_same_src_port_rate
dst_host_srv_diff_host_rate
dst_host_serror_rate
dst_host_srv_serror_rate
dst_host_rerror_rate
dst_host_srv_rerror_rate
```

זו התאמה מקורבת ביחס ל-training distribution של NSL-KDD. גרסה עתידית יכולה להוסיף window/session state בצד Flask כדי לחשב את הפיצ'רים ההיסטוריים בצורה נאמנה יותר.

## config.json

```json
{
  "flask_host": "127.0.0.1",
  "flask_port": 5000,
  "binary_threshold": 0.55,
  "scapy_interface": null,
  "packet_limit": 0,
  "log_level": "INFO",
  "artifacts_path": "artifacts",
  "agent_server_url": "http://127.0.0.1:5000/predict",
  "agent_enabled": true,
  "request_timeout_seconds": 3.0,
  "logs_path": "logs",
  "traffic_store_limit": 1000
}
```

`packet_limit: 0` אומר שה-Agent רץ ללא הגבלת packets.

`traffic_store_limit` קובע כמה רשומות נשמרות בזיכרון עבור ה-Dashboard.

## התקנה ב-Windows

מתוך `C:\project\ids`:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Scapy על Windows בדרך כלל דורש Npcap והרצה של PowerShell כ-Administrator.

בדיקת interfaces:

```powershell
python -c "from scapy.all import get_if_list; print('\n'.join(get_if_list()))"
```

אם צריך, אפשר לשים interface ספציפי ב-`config.json` תחת `scapy_interface`.

## הרצה

הרצה רגילה, כולל Dashboard ו-Agent פנימי:

```powershell
cd C:\project\ids
.\venv\Scripts\Activate.ps1
python -m server.app --config config.json
```

השרת יעלה עם ההסנפה במצב עצור. פתח את ה-Dashboard ולחץ על "התחל הסנפה" כדי להתחיל לכידת תעבורה חיה. לחץ "עצור הסנפה" כדי לעצור.

פתח בדפדפן:

```text
http://127.0.0.1:5000/dashboard
```

או:

```text
http://127.0.0.1:5000
```

הכתובת `/` מפנה ל-`/dashboard`.

בדיקת health:

```text
http://127.0.0.1:5000/health
```

## הרצה בלי אפשרות להפעיל Agent

לבדיקת UI/API בלי Scapy:

```powershell
python -m server.app --config config.json --no-agent
```

אחר כך אפשר לשלוח packet ידני:

```powershell
$body = @{
  timestamp = 1712500000.12
  src_ip = "192.168.1.10"
  dst_ip = "192.168.1.20"
  src_port = 51544
  dst_port = 80
  protocol = "tcp"
  packet_len = 512
  tcp_flags = "PA"
  ttl = 64
  ip_len = 512
  payload_len = 460
  wrong_fragment = 0
  urgent = 0
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -Body $body -ContentType "application/json"
```

הרשומה תופיע ב-Dashboard בלי רענון ידני.

## Logs

ברירת המחדל:

```text
logs/server.log
logs/agent.log
logs/attacks.log
```

`server.log` מכיל החלטות מודל, זמני תגובה, ושגיאות.

`agent.log` מכיל סטטוס Agent ושגיאות Scapy.

`attacks.log` מכיל JSON line לכל רשומה שסווגה כ-attack.
