"""Run: python test_key.py   -> tells you exactly whether Google accepts your key."""
import os, sys, requests
try:
    import tomllib
    key = tomllib.load(open(".streamlit/secrets.toml", "rb")).get("GEMINI_API_KEY", "")
except Exception:
    key = ""
key = (key or os.getenv("GEMINI_API_KEY", "")).strip()
if not key or key.startswith("PASTE"):
    sys.exit("No key found. Put it in .streamlit/secrets.toml")
print(f"Key found, length {len(key)}, starts with {key[:4]}...")
r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                 headers={"x-goog-api-key": key}, timeout=30)
print("Status:", r.status_code)
if r.ok:
    names = [m["name"] for m in r.json().get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
    print("KEY WORKS. Models you can use:")
    for n in names[:15]: print("  ", n.replace("models/", ""))
else:
    print(r.text[:500])
