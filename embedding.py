import re
import numpy as np
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────
# 👇 ВВЕДИТЕ СВОЙ ТЕКСТ ЗДЕСЬ — слова, предложение, что угодно
TEXT = """
кошка собака волк медведь
яблоко хлеб суп банан
машина поезд самолёт велосипед
"""
# ─────────────────────────────────────────────────────────────

# Разбивка текста: нижний регистр, убираем пунктуацию, дубликаты
words = list(dict.fromkeys(
    w.lower() for w in re.split(r'[\s,\.!?;:\-—]+', TEXT) if len(w) > 1
))

if len(words) < 3:
    raise ValueError("Введите хотя бы 3 слова.")

COLORS = ['#46f0f0','#3cb44b','#f58231','#e6194b','#4363d8','#f032e6',
          '#fabebe','#911eb4','#008080','#e6beff','#9a6324','#fffac8']

# 1. Эмбеддинги
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
raw_embeddings = model.encode(words)

# 2. PCA до 3D — честно, без искажения расстояний
n_components = min(3, len(words))
pca = PCA(n_components=n_components)
embeddings_3d = pca.fit_transform(raw_embeddings)

# Если слов меньше 3 — дополняем нулями
if embeddings_3d.shape[1] < 3:
    pad = np.zeros((len(words), 3 - embeddings_3d.shape[1]))
    embeddings_3d = np.hstack([embeddings_3d, pad])

var = pca.explained_variance_ratio_
max_val = np.max(np.abs(embeddings_3d)) * 1.2

# 3. Косинусное сходство по оригинальным векторам (до PCA)
norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
normed = raw_embeddings / norms

def top_similar(idx, n=3):
    n = min(n, len(words) - 1)
    sims = normed @ normed[idx]
    top = np.argsort(sims)[::-1][1:n+1]
    return [(words[j], round(float(sims[j]), 3)) for j in top]

# 4. Система координат
fig = go.Figure()
ax_c = 'yellow'
bases = np.eye(3)
ticks = np.setdiff1d(np.arange(-int(max_val), int(max_val) + 1), [0])
zeros = np.zeros_like(ticks)

for i, label in enumerate(['x', 'y', 'z']):
    v = bases[i] * max_val
    fig.add_trace(go.Scatter3d(
        x=[-v[0], v[0]], y=[-v[1], v[1]], z=[-v[2], v[2]],
        mode='lines', line=dict(color=ax_c, width=3),
        showlegend=False, hoverinfo='none'))
    fig.add_trace(go.Cone(
        x=[v[0]], y=[v[1]], z=[v[2]],
        u=[bases[i][0]], v=[bases[i][1]], w=[bases[i][2]],
        sizemode='absolute', sizeref=0.5,
        colorscale=[[0, ax_c], [1, ax_c]],
        showscale=False, hoverinfo='none'))
    t_pos = [ticks if j == i else zeros for j in range(3)]
    fig.add_trace(go.Scatter3d(
        x=t_pos[0], y=t_pos[1], z=t_pos[2],
        mode='markers', marker=dict(color=ax_c, size=4, symbol='cross'),
        showlegend=False, hoverinfo='none'))

fig.add_trace(go.Scatter3d(
    x=[max_val, 0, 0, 0], y=[0, max_val, 0, 0], z=[0, 0, max_val, 0],
    mode='text', text=['x', 'y', 'z', 'О(0,0,0)'],
    textposition=['top right']*3 + ['bottom right'],
    textfont=dict(color=[ax_c]*3 + ['white'], size=[18]*3 + [14]),
    showlegend=False, hoverinfo='none'))

# 5. Векторы слов
for i, (w, (x, y, z)) in enumerate(zip(words, embeddings_3d)):
    c = COLORS[i % len(COLORS)]
    n = np.linalg.norm([x, y, z])
    u, vd, wd = (x/n, y/n, z/n) if n > 0 else (0, 0, 0)

    similar = top_similar(i)
    hover = (f"<b>{w}</b><br>"
             f"<br>Косинусное сходство<br>(угол между векторами):<br>" +
             "<br>".join(f"  ↔ {sw}: {sc:.3f}" for sw, sc in similar))

    fig.add_trace(go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode='lines', line=dict(color=c, width=7),
        showlegend=False, hoverinfo='none'))
    fig.add_trace(go.Cone(
        x=[x], y=[y], z=[z], u=[u], v=[vd], w=[wd],
        sizemode='absolute', sizeref=0.12,
        colorscale=[[0, c], [1, c]],
        showscale=False, hoverinfo='none'))
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode='text+markers',
        text=[w],
        marker=dict(color=c, size=6),
        textposition='top center',
        textfont=dict(color=c, size=14),
        hovertext=hover, hoverinfo='text',
        showlegend=False))

# 6. Заголовок
title = (f"Визуализация эмбеддингов  —  {len(words)} слов в 3D-пространстве<br>"
         f"<sup>⚠️ Это упрощённая проекция: показано {var.sum()*100:.0f}% от реальной информации  "
         f"·  Наведи на слово, чтобы увидеть ближайших соседей</sup>")

fig.update_layout(
    paper_bgcolor='black',
    margin=dict(l=0, r=0, b=0, t=60),
    title=dict(text=title, font=dict(color='white', size=13), x=0.5),
    scene=dict(
        bgcolor='black',
        xaxis=dict(visible=False, range=[-max_val, max_val]),
        yaxis=dict(visible=False, range=[-max_val, max_val]),
        zaxis=dict(visible=False, range=[-max_val, max_val]),
        camera=dict(center=dict(x=0, y=0, z=0),
                    projection=dict(type='orthographic')))
)

fig.show()