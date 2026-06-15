from piper_phonemize import (
    phonemize_espeak, phoneme_ids_espeak,
    get_espeak_map, get_max_phonemes
)
m = get_espeak_map()
print("map size:", len(m))
print("max_phonemes:", get_max_phonemes())
r = phonemize_espeak("Bugün hava çok güzel.", "tr")
print("phonemes:", r)
ids = phoneme_ids_espeak(r[0])
print("ids[:10]:", ids[:10], "len:", len(ids))
print("BOS=1 PAD=0 EOS=2:", ids[0], ids[1], ids[-1])
print("OK")
