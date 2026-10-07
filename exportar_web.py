# -*- coding: utf-8 -*-
"""Vuelve a incrustar el modelo de Cerebro dentro de index.html.

Úsalo cada vez que reentrenes el modelo:

    python3 exportar_web.py                       # usa cerebro-main/checkpoints/best.npz
    python3 exportar_web.py ruta/a/otro.npz       # o un checkpoint concreto

Solo necesita numpy. Después sube index.html a GitHub.
"""
import base64, json, re, sys
import numpy as np

CKPT = sys.argv[1] if len(sys.argv) > 1 else "cerebro-main/checkpoints/best.npz"
HTML = "index.html"

z = np.load(CKPT, allow_pickle=False)
meta = json.loads(str(z["meta"]))

tensores, trozos, pos = [], [], 0
for k in z.files:
    if k.startswith("p__"):
        a = np.ascontiguousarray(z[k], dtype="<f4")
        tensores.append({"nombre": k[3:], "forma": list(a.shape), "desde": pos, "n": int(a.size)})
        trozos.append(a.reshape(-1))
        pos += a.size

info = {
    "config": meta["config"],
    "fusiones": meta["tokenizador"]["fusiones"],
    "n_parametros": meta.get("n_parametros"),
    "val_loss": meta.get("val_loss"),
    "paso": meta.get("paso"),
    "perfil": meta.get("perfil"),
    "tensores": tensores,
}
texto_json = json.dumps(info, ensure_ascii=False, separators=(",", ":"))
b64 = base64.b64encode(np.concatenate(trozos).tobytes()).decode()

html = open(HTML, encoding="utf-8").read()
html, n1 = re.subn(r'(<script id="mj" type="application/json">).*?(</script>)',
                   lambda m: m.group(1) + texto_json + m.group(2), html, count=1, flags=re.S)
html, n2 = re.subn(r'(<script id="mb" type="application/octet-stream">).*?(</script>)',
                   lambda m: m.group(1) + b64 + m.group(2), html, count=1, flags=re.S)
if not (n1 and n2):
    sys.exit("No encontré las etiquetas del modelo en index.html.")
open(HTML, "w", encoding="utf-8").write(html)
print(f"OK: {pos} pesos incrustados en {HTML} ({len(html)/1e6:.2f} MB)")
