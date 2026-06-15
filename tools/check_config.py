import json, sys
path = sys.argv[1]
c = json.load(open(path))
m = c.get("phoneme_id_map", {})
print("phoneme_id_map entries:", len(m))
print("sample:", list(m.items())[:6])
print("num_symbols:", c.get("num_symbols"))
print("audio:", c.get("audio"))
print("espeak:", c.get("espeak"))
