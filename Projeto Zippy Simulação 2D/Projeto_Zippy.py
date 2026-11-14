#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulador do robô móvel Zippy, com navegação, LiDAR e planejamento A*."""
import math
import heapq
import os
import random
from collections import deque

import numpy as np
import pygame

# 1. CONFIGURAÇÕES GERAIS

IMG_FILE = "assets/Planta_Baixa_Modern_House.jpeg"
IMG_W, IMG_H = 1155, 649          
CELL = 6                          
MAPW, MAPH = 1158, 654            
GC, GR = MAPW // CELL, MAPH // CELL
PX_PER_M = 100.0                  

PANEL_W = 345                     
BOTTOM_H = 92                     

ROBOT_R = 0.12                    
ROBOT_HEIGHT = 0.12               
HEIGHT_MARGIN = 0.03              
WHEELBASE = 0.28                  
MAX_WHEEL = 1.0                   
V_MAX, W_MAX = 0.75, 2.4          
MAX_ACC = 2.2                     
SLIP = (0.998, 1.002)             

LIDAR_RANGE, LIDAR_MIN = 5.0, 0.10
LIDAR_NOISE, LIDAR_DROP = 0.01, 0.015

KP_ANG, KP_DIST = 2.8, 1.1
TOL_WP, TOL_FINAL = 0.10, 0.12
LOOKAHEAD = 0.10                  
D_CRITICA, D_SAIDA = 0.26, 0.42   
INFLATE_CELLS = 3.0              

# 2. A PLANTA: paredes, móveis e portas (coordenadas em pixels da imagem)

PAREDES = [
    
    ("parede superior", 0, 0, MAPW, 4),
    ("parede esquerda", 0, 240, 5, MAPH),
    ("parede direita", 1150, 0, MAPW, MAPH),
    ("parede inferior esquerda", 0, 642, 510, MAPH),
    ("parede inferior direita", 670, 642, MAPW, MAPH),
    ("volume sem piso superior", 0, 0, 671, 65),
    ("volume sem piso lateral", 0, 65, 84, 245),

    ("parede cozinha vertical", 80, 65, 89, 245),
    ("parede cozinha horizontal", 84, 65, 671, 72),

    ("divisória central superior", 665, 0, 675, 176),
    ("divisória central entre portas", 665, 268, 675, 465),
    ("divisória central inferior", 665, 550, 675, MAPH),

    ("parede quarto banheiro", 670, 410, MAPW, 420),
]

MOVEIS = [
    
    ("ilha com fogão", 245, 147, 508, 227),
    ("banquetas da ilha", 254, 227, 365, 263),
    ("aparador cozinha", 568, 165, 662, 182),
    ("aparador corredor", 600, 290, 662, 456),

    ("sofá curvo", 120, 320, 394, 435),
    ("poltrona esquerda", 15, 460, 84, 532),
    ("poltrona direita", 389, 460, 458, 532),
    ("mesa de centro", 162, 463, 324, 538),
    ("rack de TV", 0, 601, 376, 649),
    ("móvel baixo inferior", 400, 602, 510, 649),

    ("criado-mudo", 680, 308, 738, 411),
    ("cama", 766, 88, 1062, 410),
    ("escrivaninha", 1050, 0, 1155, 255),

    ("gabinete da pia", 700, 422, 887, 480),
    ("vaso sanitário", 960, 425, 1012, 506),
    ("box", 1060, 450, 1155, 582),
    ("armário banheiro", 691, 590, 751, 647),
    ("planta banheiro", 1090, 570, 1155, 649),
]

MOVEIS_PASSAVEIS = [
    ("banquetas da ilha", 254, 227, 365, 263, 0.38, 0,
     ((260, 229), (302, 229), (318, 229), (359, 229),
      (260, 255), (302, 255), (318, 255), (359, 255))),
    ("mesa de centro", 162, 463, 324, 538, 0.42, 0,
     ((171, 472), (315, 472), (171, 529), (315, 529))),
    ("cama", 766, 88, 1045, 410, 0.20, 0,
     ((785, 110), (1043, 110), (785, 389), (1043, 389))),
]

PORTAS = [
    ((570, 174), (670, 174)), ((670, 177), (670, 268)),
    ((568, 463), (670, 463)), ((670, 465), (670, 550)),
    ((510, 550), (510, 642)), ((510, 642), (665, 642)),
]

COMODOS = {   
    "Cozinha": (520, 245), "Jantar": (475, 355), "Sala": (525, 520),
    "Quarto": (900, 205), "Banheiro": (870, 565), "Entrada": (600, 610),
}
START_PX = (520, 345)
TOUR = ["Cozinha", "Jantar", "Sala", "Banheiro", "Quarto", "Entrada"]

# 3. UTILIDADES DE COORDENADAS (mundo em metros, y para CIMA)  <->  planta (px)

def px2m(px, py):
    return px / PX_PER_M, (IMG_H - py) / PX_PER_M

def m2px(x, y):
    return x * PX_PER_M, IMG_H - y * PX_PER_M

def m2cell(x, y):
    px, py = m2px(x, y)
    return int(px // CELL), int(py // CELL)

def cell2m(cx, cy):
    return px2m((cx + 0.5) * CELL, (cy + 0.5) * CELL)

# 4. FUNÇÕES DA MATÉRIA (Aulas 2-5)  - usadas de verdade pelos nós abaixo

def normalizar_angulo(a):
    """Menor diferença angular em [-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi

def converter_cmd_vel(v, omega, L=WHEELBASE, max_wheel_speed=MAX_WHEEL):
    """AC-2 Ex.1: (v, omega) -> (v_e, v_d) com saturação proporcional."""
    v_d = v + omega * L / 2.0
    v_e = v - omega * L / 2.0
    m = max(abs(v_e), abs(v_d))
    if m > max_wheel_speed:                     
        k = max_wheel_speed / m
        v_e, v_d = v_e * k, v_d * k
    return v_e, v_d

def processar_scan(leituras):
    """AC-2 Ex.2: filtra o LiDAR (descarta 0.0 e inf) e devolve o menor valor
    válido de 3 setores. Índice k = graus anti-horário a partir da frente."""
    l = np.asarray(leituras, dtype=float)
    ok = np.isfinite(l) & (l >= 0.1) & (l <= 5.0)

    def menor(idx):
        v = l[idx][ok[idx]]
        return float(v.min()) if v.size else LIDAR_RANGE

    frente = np.r_[345:360, 0:16]
    esq = np.arange(45, 136)
    dire = np.arange(225, 316)
    return {"frente": menor(frente), "esquerda": menor(esq), "direita": menor(dire)}

def controle_reativo(d, v_livre=0.5, d_critica=D_CRITICA):
    """AC-2 Ex.3: Braitenberg + trava de segurança. Retorna (v, omega).
    omega > 0 = girar para a ESQUERDA (convenção ROS)."""
    if d["frente"] < d_critica:
        return 0.0, (1.0 if d["esquerda"] >= d["direita"] else -1.0)
    return v_livre, float(np.clip(0.8 * (d["esquerda"] - d["direita"]), -1, 1))

def calcular_orientacao_alvo(x, y, theta, x_alvo, y_alvo, Kp=KP_ANG):
    """AC-2 Ex.4: controle P de orientação (Go-to-Goal). Retorna (omega, e_theta)."""
    theta_alvo = math.atan2(y_alvo - y, x_alvo - x)
    e = normalizar_angulo(theta_alvo - theta)
    return Kp * e, e

# 5. MUNDO (obstáculos), ROBÔ (física) E LIDAR

def pool(a):
    """Reduz um array (MAPH, MAPW) para a grade de células (qualquer px ocupado)."""
    return a.reshape(GR, CELL, GC, CELL).any(axis=(1, 3))

def shift(g, dy, dx):
    out = np.zeros_like(g)
    H, W = g.shape
    ys, yd = (slice(dy, H), slice(0, H - dy)) if dy >= 0 else (slice(0, H + dy), slice(-dy, H))
    xs, xd = (slice(dx, W), slice(0, W - dx)) if dx >= 0 else (slice(0, W + dx), slice(-dx, W))
    out[ys, xs] = g[yd, xd]
    return out

def dilate(g, r):
    """Dilatação circular (raio r em células) - inflação de obstáculos."""
    out = g.copy()
    R = int(math.ceil(r))
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            if (dx or dy) and dx * dx + dy * dy <= r * r + 1e-9:
                out |= shift(g, dy, dx)
    return out

class World:
    def __init__(self):
        self.wall_static = np.zeros((MAPH, MAPW), bool)
        for _, x0, y0, x1, y1 in PAREDES:
            self.wall_static[y0:y1, x0:x1] = True

        self.furniture_static = np.zeros((MAPH, MAPW), bool)
        for _, x0, y0, x1, y1 in MOVEIS:
            self.furniture_static[y0:y1, x0:x1] = True

        passage_mask = np.zeros((MAPH, MAPW), bool)
        support_mask = np.zeros((MAPH, MAPW), bool)
        for (_, x0, y0, x1, y1, clearance, margin, supports) in MOVEIS_PASSAVEIS:
            if clearance < ROBOT_HEIGHT + HEIGHT_MARGIN:
                continue
            ix0, iy0 = x0 + margin, y0 + margin
            ix1, iy1 = x1 - margin, y1 - margin
            if ix0 < ix1 and iy0 < iy1:
                passage_mask[iy0:iy1, ix0:ix1] = True
            for sx, sy in supports:
                radius = max(3, int(round(0.025 * PX_PER_M)))
                yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
                disk = xx * xx + yy * yy <= radius * radius
                ax0, ay0 = sx - radius, sy - radius
                ax1, ay1 = sx + radius + 1, sy + radius + 1
                cx0, cy0 = max(0, ax0), max(0, ay0)
                cx1, cy1 = min(MAPW, ax1), min(MAPH, ay1)
                if cx0 < cx1 and cy0 < cy1:
                    dx0, dy0 = cx0 - ax0, cy0 - ay0
                    support_mask[cy0:cy1, cx0:cx1] |= disk[dy0:dy0 + cy1 - cy0,
                                                            dx0:dx0 + cx1 - cx0]

        self.static = (self.wall_static |
                       (self.furniture_static & ~passage_mask) |
                       support_mask)
        self.boxes = []                       
        self.occ = self.static.copy()
        self.static_cells = pool(self.static)

        self.free_cells = ~self.static_cells
        self._disk_cache = {}

    def rebuild(self):
        self.occ = self.static.copy()
        for r in self.boxes:
            x0, y0 = max(0, r.left), max(0, r.top)
            x1, y1 = min(MAPW, r.right), min(MAPH, r.bottom)
            if x0 < x1 and y0 < y1:
                self.occ[y0:y1, x0:x1] = True

    def toggle_box(self, px, py, robot_xy, size=38):
        for r in self.boxes:
            if r.collidepoint(px, py):
                self.boxes.remove(r)
                self.rebuild()
                return "removida"
        r = pygame.Rect(0, 0, size, size)
        r.center = (int(px), int(py))
        x0, y0 = max(0, r.left), max(0, r.top)
        x1, y1 = min(MAPW, r.right), min(MAPH, r.bottom)
        if x0 >= x1 or y0 >= y1 or self.static[y0:y1, x0:x1].any():
            return "ocupado"
        rx, ry = m2px(*robot_xy)
        if math.hypot(rx - px, ry - py) < 45:
            return "perto"
        self.boxes.append(r)
        self.rebuild()
        return "criada"

    def is_occ(self, px, py):
        if px < 0 or py < 0 or px >= MAPW or py >= MAPH:
            return True
        return bool(self.occ[int(py), int(px)])

    def _disk_mask(self, radius_px):
        mask = self._disk_cache.get(radius_px)
        if mask is None:
            yy, xx = np.ogrid[-radius_px:radius_px + 1,
                              -radius_px:radius_px + 1]
            mask = (xx * xx + yy * yy) <= radius_px * radius_px
            self._disk_cache[radius_px] = mask
        return mask

    def overlap(self, x, y, r):
        """Número de pixels ocupados dentro do disco físico do robô."""
        px, py = m2px(x, y)
        cx, cy = int(round(px)), int(round(py))
        rp = int(math.ceil(r * PX_PER_M))
        x0, x1, y0, y1 = cx - rp, cx + rp + 1, cy - rp, cy + rp + 1
        if x0 < 0 or y0 < 0 or x1 > MAPW or y1 > MAPH:
            return 1
        return int(np.count_nonzero(self.occ[y0:y1, x0:x1] & self._disk_mask(rp)))

    def circle_free(self, x, y, r):
        return self.overlap(x, y, r) == 0

    def motion_free(self, x0, y0, x1, y1, r):
        """Verifica o volume varrido, não apenas a pose final (sem atravessar paredes)."""
        distance_px = math.hypot(x1 - x0, y1 - y0) * PX_PER_M
        steps = max(1, int(math.ceil(distance_px / max(1.0, CELL * 0.5))))
        start_clear = self.circle_free(x0, y0, r)
        previous_overlap = self.overlap(x0, y0, r)
        for i in range(1, steps + 1):
            t = i / steps
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            if start_clear:
                if not self.circle_free(x, y, r):
                    return False
            else:

                current_overlap = self.overlap(x, y, r)
                if current_overlap > previous_overlap:
                    return False
                previous_overlap = current_overlap
        return True

def lidar_scan(world, x, y, th, noise=True):
    """Aula 03: 360 raios por raycasting (amostragem ao longo do raio, vetorizado).
    Retorna array de 360 distâncias: inf = sem retorno, 0.0 = falha do sensor."""
    ang = th + np.radians(np.arange(360))
    steps = np.arange(LIDAR_MIN, LIDAR_RANGE, 0.03)
    xs = x + np.cos(ang)[:, None] * steps[None, :]
    ys = y + np.sin(ang)[:, None] * steps[None, :]
    px = (xs * PX_PER_M).astype(int)
    py = (IMG_H - ys * PX_PER_M).astype(int)
    inb = (px >= 0) & (px < MAPW) & (py >= 0) & (py < MAPH)
    hit = np.ones(px.shape, bool)
    hit[inb] = world.occ[py[inb], px[inb]]
    anyhit = hit.any(1)
    first = hit.argmax(1)
    r = np.where(anyhit, steps[first], np.inf)
    if noise:
        fin = np.isfinite(r)
        r[fin] += np.random.normal(0, LIDAR_NOISE, fin.sum())
        r[np.random.rand(360) < LIDAR_DROP] = 0.0
    return r

class Robot:
    """Física 'real' do robô (Aula 02): rodas -> v, omega -> integração da pose."""

    def __init__(self, x, y, th):
        self.x, self.y, self.th = x, y, th
        self.tl = self.tr = 0.0        
        self.vl = self.vr = 0.0        
        self.v = self.w = 0.0
        self.collisions = 0
        self.dist = 0.0
        self._colliding = False

    def step(self, dt, world, noise=True):
        a = MAX_ACC * dt
        self.vl += float(np.clip(self.tl - self.vl, -a, a))
        self.vr += float(np.clip(self.tr - self.vr, -a, a))
        sl, sr = SLIP if noise else (1.0, 1.0)
        vl, vr = self.vl * sl, self.vr * sr
        v, w = (vr + vl) / 2.0, (vr - vl) / WHEELBASE
        th2 = self.th + w * dt
        x2 = self.x + v * math.cos(self.th + w * dt / 2) * dt
        y2 = self.y + v * math.sin(self.th + w * dt / 2) * dt
        free_new = world.motion_free(self.x, self.y, x2, y2, ROBOT_R)
        escaping = (not free_new and self._colliding and
                    world.overlap(x2, y2, ROBOT_R) <= world.overlap(self.x, self.y, ROBOT_R))
        if free_new or escaping:
            self.dist += math.hypot(x2 - self.x, y2 - self.y)
            self.x, self.y, self.th = x2, y2, normalizar_angulo(th2)
            self.v, self.w = v, w
            self._colliding = not free_new
        else:                                    
            self.th = normalizar_angulo(th2)
            self.v, self.w = 0.0, w
            if not self._colliding:
                self.collisions += 1
            self._colliding = True

# 6. PLANEJAMENTO (IA): A* + suavização por linha de visada

def nearest_free(blocked, cell, maxr=40):
    H, W = blocked.shape
    cx, cy = cell
    cx, cy = min(max(cx, 0), W - 1), min(max(cy, 0), H - 1)
    if not blocked[cy, cx]:
        return (cx, cy)
    seen = {(cx, cy)}
    q = deque([(cx, cy, 0)])
    while q:
        x, y, d = q.popleft()
        if d > maxr:
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in seen:
                if not blocked[ny, nx]:
                    return (nx, ny)
                seen.add((nx, ny))
                q.append((nx, ny, d + 1))
    return None

def astar(blocked, near, start, goal):
    """A* em grade 8-conectada. 'near' = células perto de parede (custo extra:
    o caminho tende a passar pelo meio dos cômodos e portas)."""
    H, W = blocked.shape
    blk = blocked.ravel().tolist()
    pen = (near.ravel() * 2.5).tolist()
    s, g = start[1] * W + start[0], goal[1] * W + goal[0]
    gx, gy = goal
    INF = 1e18
    gs = {s: 0.0}
    parent = {s: -1}
    openq = [(0.0, s)]
    closed = set()
    D = 1.4142
    nbrs = ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
            (1, 1, D), (1, -1, D), (-1, 1, D), (-1, -1, D))
    while openq:
        _, cur = heapq.heappop(openq)
        if cur in closed:
            continue
        if cur == g:
            path = []
            while cur != -1:
                path.append((cur % W, cur // W))
                cur = parent[cur]
            return path[::-1]
        closed.add(cur)
        cx, cy = cur % W, cur // W
        for dx, dy, c in nbrs:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or ny < 0 or nx >= W or ny >= H:
                continue
            ni = ny * W + nx
            if blk[ni] or ni in closed:
                continue
            if dx and dy and (blk[cy * W + nx] or blk[ny * W + cx]):
                continue                           
            ng = gs[cur] + c + pen[ni]
            if ng < gs.get(ni, INF):
                gs[ni] = ng
                parent[ni] = cur
                ddx, ddy = abs(nx - gx), abs(ny - gy)
                h = (ddx + ddy) + (D - 2) * min(ddx, ddy)
                heapq.heappush(openq, (ng + 1.1 * h, ni))
    return None

def line_free(blocked, a, b):
    """Checa a linha central em superamostragem; bloqueia diagonais que raspem quinas.

    A grade recebida já está inflada pelo raio físico do Zippy e pela folga
    de segurança. Amostrar a cada 1/4 de célula evita saltar pixels bloqueados.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = max(1, int(math.ceil(max(abs(dx), abs(dy)) * 4)))
    prev = None
    h, w = blocked.shape
    for i in range(n + 1):
        t = i / n
        x = int(round(a[0] + dx * t))
        y = int(round(a[1] + dy * t))
        if not (0 <= x < w and 0 <= y < h) or blocked[y, x]:
            return False
        if prev is not None:
            px, py = prev
            
            if x != px and y != py:
                if blocked[py, x] or blocked[y, px]:
                    return False
        prev = (x, y)
    return True

def smooth_path(cells, blocked):
    out, i = [cells[0]], 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not line_free(blocked, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out

# 7. ARQUITETURA ESTILO ROS 2: barramento de tópicos + nós

class Bus:
    """Mini publish/subscribe (conceito de tópicos do ROS 2)."""

    def __init__(self):
        self.subs, self.last, self.cnt, self.rate, self.stamp = {}, {}, {}, {}, {}
        self.t0 = 0.0
        self.now = 0.0

    def sub(self, topic, cb):
        self.subs.setdefault(topic, []).append(cb)

    def pub(self, topic, msg):
        self.last[topic] = msg
        self.cnt[topic] = self.cnt.get(topic, 0) + 1
        self.stamp[topic] = self.now
        for cb in self.subs.get(topic, []):
            cb(msg)

    def tick(self, now):
        self.now = now
        if now - self.t0 >= 1.0:
            dt = now - self.t0
            self.rate = {k: v / dt for k, v in self.cnt.items()}
            self.cnt = {k: 0 for k in self.cnt}
            self.t0 = now

class Params:
    def __init__(self):
        self.use_odom = False        
        self.noise = True
        self.mode = "OCIOSO"         
        self.manual = (0.0, 0.0)

class Node:
    period = 0.1

    def __init__(self, sim):
        self.sim, self.bus, self.acc = sim, sim.bus, 0.0

    def tick(self):
        pass

class MotorDriverNode(Node):
    """/cmd_vel -> rodas (usa converter_cmd_vel da AC-2)."""
    period = 1 / 60

    def __init__(self, sim):
        super().__init__(sim)
        self.cmd, self.age = (0.0, 0.0), 0.0
        self.bus.sub("/cmd_vel", self.on_cmd)

    def on_cmd(self, m):
        self.cmd, self.age = m, 0.0

    def tick(self):
        self.age += self.period
        v, w = self.cmd if self.age < 0.5 else (0.0, 0.0)     
        ve, vd = converter_cmd_vel(v, w)
        self.sim.robot.tl, self.sim.robot.tr = ve, vd
        self.bus.pub("/wheel_cmd", (ve, vd))

class EncoderOdomNode(Node):
    """Odometria por integração das rodas (Aula 02) -> /odom (acumula deriva)."""
    period = 1 / 30

    def __init__(self, sim):
        super().__init__(sim)
        self.x, self.y, self.th = sim.robot.x, sim.robot.y, sim.robot.th

    def tick(self):
        r = self.sim.robot
        n = 0.003 if self.sim.p.noise else 0.0
        vl, vr = r.vl + random.gauss(0, n), r.vr + random.gauss(0, n)
        v, w = (vr + vl) / 2.0, (vr - vl) / WHEELBASE
        self.th = normalizar_angulo(self.th + w * self.period)
        self.x += v * math.cos(self.th) * self.period
        self.y += v * math.sin(self.th) * self.period
        self.bus.pub("/odom", (self.x, self.y, self.th, v, w))

    def relocalize(self):
        r = self.sim.robot
        self.x, self.y, self.th = r.x, r.y, r.th

class GroundTruthNode(Node):
    period = 1 / 30

    def tick(self):
        r = self.sim.robot
        self.bus.pub("/ground_truth", (r.x, r.y, r.th))

class LidarNode(Node):
    period = 0.1

    def tick(self):
        r = self.sim.robot
        self.bus.pub("/scan", lidar_scan(self.sim.world, r.x, r.y, r.th, self.sim.p.noise))

class PerceptionNode(Node):
    """/scan -> /scan_sectors (filtro + setores, AC-2 Ex.2)."""

    def __init__(self, sim):
        super().__init__(sim)
        self.bus.sub("/scan", lambda s: self.bus.pub("/scan_sectors", processar_scan(s)))

class MapperNode(Node):
    """Mapeamento por grade de ocupação (log-odds) a partir do /scan."""
    period = 0.5

    def __init__(self, sim):
        super().__init__(sim)
        self.L = np.zeros((GR, GC), np.float32)
        self.seen = np.zeros((GR, GC), bool)
        self.coverage = 0.0
        self.bus.sub("/scan", self.on_scan)

    def on_scan(self, scan):
        x, y, th = self.sim.est_pose()
        n = len(scan)
        ang = th + np.radians(np.arange(n))
        rr = np.where(np.isfinite(scan), scan, LIDAR_RANGE)
        valid = (scan >= LIDAR_MIN)               
        hit = np.isfinite(scan) & valid
        d = np.arange(0.06, LIDAR_RANGE, CELL / PX_PER_M * 0.9)
        m = (d[None, :] < (rr[:, None] - 0.05)) & valid[:, None]
        xs = x + np.cos(ang)[:, None] * d[None, :]
        ys = y + np.sin(ang)[:, None] * d[None, :]
        cx = (xs * PX_PER_M // CELL).astype(int)
        cy = ((IMG_H - ys * PX_PER_M) // CELL).astype(int)
        ok = m & (cx >= 0) & (cx < GC) & (cy >= 0) & (cy < GR)
        np.add.at(self.L, (cy[ok], cx[ok]), -0.45)
        self.seen[cy[ok], cx[ok]] = True
        hx = x + np.cos(ang[hit]) * rr[hit]
        hy = y + np.sin(ang[hit]) * rr[hit]
        hcx = (hx * PX_PER_M // CELL).astype(int)
        hcy = ((IMG_H - hy * PX_PER_M) // CELL).astype(int)
        k = (hcx >= 0) & (hcx < GC) & (hcy >= 0) & (hcy < GR)
        np.add.at(self.L, (hcy[k], hcx[k]), 1.0)
        self.seen[hcy[k], hcx[k]] = True
        np.clip(self.L, -3.0, 3.0, out=self.L)

    def occupied(self):
        return self.L > 0.8

    def free(self):
        return self.seen & (self.L < -0.4)

    def tick(self):
        w = self.sim.world
        tot = w.free_cells.sum()
        self.coverage = float((self.seen & w.free_cells).sum()) / max(1, tot)
        self.bus.pub("/map", {"coverage": self.coverage})

    def frontier(self, start_cell, blocked, blacklist):
        """Fronteira = célula livre conhecida vizinha de região desconhecida.
        BFS a partir do robô -> fronteira mais próxima e alcançável."""
        free = self.free() & ~blocked
        unk = ~self.seen
        near_unk = np.zeros_like(unk)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            near_unk |= shift(unk, dy, dx)
        front = free & near_unk
        s = nearest_free(~free, start_cell, 6)
        if s is None:
            return None
        seen = {s}
        q = deque([s])
        while q:
            x, y = q.popleft()
            if front[y, x] and all(math.hypot(x - bx, y - by) > 7 for bx, by in blacklist):
                return (x, y)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < GC and 0 <= ny < GR and (nx, ny) not in seen and free[ny, nx]:
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return None

class PlannerNode(Node):
    """/goal -> A* -> /path. Replaneja sozinho se o caminho for bloqueado."""
    period = 0.5

    def __init__(self, sim):
        super().__init__(sim)
        w = sim.world
        self.static_blocked = dilate(w.static_cells, INFLATE_CELLS)
        self.static_near = dilate(w.static_cells, 1.5)
        self.goal = None
        self.waypoints = []                 
        self.dense = []
        self.map_kind = "planta"
        self.new_obs = np.zeros((GR, GC), bool)
        self.bus.sub("/goal", self.on_goal)
        self.bus.sub("/goal_reached", lambda m: self.clear())
        self.bus.sub("/replan_request", lambda m: self.plan())

    def clear(self):
        self.goal, self.waypoints = None, []

    def blocked_grid(self):
        mp = self.sim.mapper
        if self.map_kind == "aprendido":
            return dilate(mp.occupied(), INFLATE_CELLS) | ~mp.seen
        self.new_obs = mp.occupied() & ~self.static_near
        if self.new_obs.any():
            return self.static_blocked | dilate(self.new_obs, INFLATE_CELLS)
        return self.static_blocked

    def on_goal(self, m):
        self.goal = m
        self.map_kind = m.get("map", "planta")
        self.plan()

    def plan(self):
        if not self.goal:
            return
        blocked = self.blocked_grid()
        near = dilate(blocked, 2.5) & ~blocked
        x, y, _ = self.sim.est_pose()
        s = nearest_free(blocked, m2cell(x, y))
        gx, gy = self.goal["goal"]
        g = nearest_free(blocked, m2cell(gx, gy))
        cells = astar(blocked, near, s, g) if (s and g) else None
        if not cells:
            self.waypoints, self.dense = [], []
            self.bus.pub("/path", {"pts": [], "name": self.goal.get("name", "")})
            return
        self.dense = cells
        self.waypoints = smooth_path(cells, blocked)
        pts = [cell2m(*c) for c in self.waypoints]
        length = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                     for i in range(len(pts) - 1))
        self.bus.pub("/path", {"pts": pts, "name": self.goal.get("name", ""), "length": length})

    def tick(self):
        if self.goal and not self.waypoints:       
            self.retry = getattr(self, "retry", 0) + 1
            if self.retry % 2 == 0:
                self.plan()
            return
        if not self.goal:
            return
        blocked = self.blocked_grid()
        for i in range(len(self.waypoints) - 1):
            if not line_free(blocked, self.waypoints[i], self.waypoints[i + 1]):
                self.sim.log("Planner: caminho bloqueado -> replanejando (A*)")
                self.plan()
                return

class BrainNode(Node):
    """FSM + controlador (AC-2 Ex.3/4/5) -> /cmd_vel."""
    period = 1 / 30

    def __init__(self, sim):
        super().__init__(sim)
        self.estado, self.path, self.idx = "PARADO", [], 1
        self.goal_name, self.cool, self.dir = "", 0.0, 1
        self.sec = {"frente": 5.0, "esquerda": 5.0, "direita": 5.0}
        self.hist = deque(maxlen=90)
        self.bus.sub("/scan_sectors", lambda s: setattr(self, "sec", s))
        self.bus.sub("/path", self.on_path)

    def on_path(self, m):
        self.goal_name = m.get("name", "")
        if not m["pts"]:
            if self.estado != "SEM_ROTA":
                self.sim.log(f"Sem rota para {self.goal_name or 'o destino'}! (vou tentar de novo)")
            self.path, self.estado = [], "SEM_ROTA"
            return
        self.path, self.idx = m["pts"], 1
        if self.estado != "DESVIAR_OBSTACULO":
            self.estado = "IR_PARA_ALVO"
        self.hist.clear()

    def start_goal(self):
        self.estado = "IR_PARA_ALVO"

    def step_fsm(self, x, y, th):
        """Retorna (v, omega) e atualiza o estado (AC-2 Ex.5)."""
        sec, v, w = self.sec, 0.0, 0.0
        if self.cool > 0:
            self.cool -= self.period
        if self.estado == "DESVIAR_OBSTACULO":
            v, w = 0.0, 1.0 * self.dir               
            if sec["frente"] > D_SAIDA:
                self.estado, self.cool = "IR_PARA_ALVO", 1.2
                self.sim.log("Frente livre -> replanejando a rota")
                self.bus.pub("/replan_request", True)
        elif self.estado == "IR_PARA_ALVO" and self.path:
            gx, gy = self.path[-1]
            while self.idx < len(self.path) - 1 and \
                    math.hypot(self.path[self.idx][0] - x, self.path[self.idx][1] - y) < TOL_WP:
                self.idx += 1
            if math.hypot(gx - x, gy - y) < TOL_FINAL:
                self.estado = "OBJETIVO_ALCANÇADO"
                self.sim.log(f"Chegamos: {self.goal_name}")
                self.bus.pub("/goal_reached", self.goal_name)
                return 0.0, 0.0
            if self.cool <= 0 and sec["frente"] < D_CRITICA:
                self.estado = "DESVIAR_OBSTACULO"
                self.dir = 1 if sec["esquerda"] >= sec["direita"] else -1
                self.sim.log(f"Obstáculo a {sec['frente']:.2f} m! Freio + giro")
                return 0.0, 1.0 * self.dir

            ax, ay = self.path[self.idx - 1]
            bx, by = self.path[self.idx]
            seg = math.hypot(bx - ax, by - ay)
            if seg > 1e-6:
                ux, uy = (bx - ax) / seg, (by - ay) / seg
                t = min(max((x - ax) * ux + (y - ay) * uy, 0.0), seg)
                tc = min(seg, t + LOOKAHEAD)
                tx, ty = ax + ux * tc, ay + uy * tc
            else:
                tx, ty = bx, by
            w, e = calcular_orientacao_alvo(x, y, th, tx, ty)
            w = float(np.clip(w, -W_MAX, W_MAX))
            dB = math.hypot(bx - x, by - y)
            
            alinhamento = max(0.0, math.cos(e))
            v = float(np.clip(KP_DIST * dB, 0.06, 0.38)) * alinhamento ** 4
            if self.idx < len(self.path) - 1 and dB < 0.45 and seg > 1e-6:     
                cx_, cy_ = self.path[self.idx + 1]
                s2 = math.hypot(cx_ - bx, cy_ - by)
                if s2 > 1e-6:
                    ct = ((bx - ax) * (cx_ - bx) + (by - ay) * (cy_ - by)) / (seg * s2)
                    turn = math.acos(max(-1.0, min(1.0, ct)))
                    v *= max(0.3, 1.0 - turn / math.pi)
            v *= float(np.clip((sec["frente"] - 0.22) / 0.6, 0.25, 1.0))   
            
            self.hist.append((x, y))
            if len(self.hist) == self.hist.maxlen and \
                    math.hypot(x - self.hist[0][0], y - self.hist[0][1]) < 0.04:
                self.hist.clear()
                self.sim.log("Robô parado há 3 s -> replanejando")
                self.bus.pub("/replan_request", True)
        return v, w

    def tick(self):
        p = self.sim.p
        if p.mode == "MANUAL":
            self.estado = "MANUAL"
            self.bus.pub("/cmd_vel", p.manual)
        else:
            if self.estado == "MANUAL":
                self.estado = "PARADO"
            x, y, th = self.sim.est_pose()
            self.bus.pub("/cmd_vel", self.step_fsm(x, y, th))
        self.bus.pub("/zippy/status", self.estado)

class MissionNode(Node):
    """Quem decide os destinos: clique, patrulha, exploração ou limpeza total."""
    period = 0.4
    CLEAN_STEP = 4        
    BRUSH_R = 2.5         

    def __init__(self, sim):
        super().__init__(sim)
        self.p = sim.p
        self.tour_i, self.wait, self.blacklist, self.cur_frontier = 0, 0.0, [], None
        self.done_explore = False
        
        self.clean = np.zeros((GR, GC), bool)      
        self.skip = set()                          
        self.clean_current = None
        self.region = np.zeros((GR, GC), bool)     
        self.region_total = 1
        self.clean_pct = 0.0
        R = int(math.ceil(self.BRUSH_R))
        offs = [(dx, dy) for dy in range(-R, R + 1) for dx in range(-R, R + 1)
                if dx * dx + dy * dy <= self.BRUSH_R ** 2]
        self._dx = np.array([o[0] for o in offs])
        self._dy = np.array([o[1] for o in offs])

    def mark_clean(self):
        """Chamado a cada passo da simulação: a escova limpa as células sob o robô."""
        r = self.sim.robot
        cx, cy = m2cell(r.x, r.y)
        xs = np.clip(cx + self._dx, 0, GC - 1)
        ys = np.clip(cy + self._dy, 0, GR - 1)
        self.clean[ys, xs] = True

    def _flood(self, blocked, s):
        """Todas as células alcançáveis a partir de s (8-conectado)."""
        reach = np.zeros((GR, GC), bool)
        reach[s[1], s[0]] = True
        q = deque([s])
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < GC and 0 <= ny < GR and not reach[ny, nx] and not blocked[ny, nx]:
                    reach[ny, nx] = True
                    q.append((nx, ny))
        return reach

    def start_total_clean(self):
        self.sim.stop()
        self.p.mode = "LIMPEZA_TOTAL"
        self.period = 0.1                        
        pl = self.sim.planner
        pl.map_kind = "planta"
        blocked = pl.blocked_grid()
        x, y, _ = self.sim.est_pose()
        s = nearest_free(blocked, m2cell(x, y))
        self.clean[:] = False
        self.skip = set()
        self.clean_current = None
        if s is None:
            self.sim.log("Limpeza: robô sem espaço livre")
            self.p.mode = "OCIOSO"
            return
        self.region = self._flood(blocked, s)
        self.region_total = max(1, int(self.region.sum()))
        self.clean_pct = 0.0
        self.sim.brain.estado = "PARADO"
        self.mark_clean()
        self.sim.log(f"Limpeza total iniciada: {self.region_total} células a limpar")
        self._next_clean_target()

    def _finish_clean(self):
        self.clean_current = None
        self.p.mode = "OCIOSO"
        self.sim.brain.estado = "PARADO"
        self.sim.planner.clear()
        self.sim.brain.path = []
        self.sim.log(f"Limpeza total concluída: {self.clean_pct * 100:.0f}% da área alcançável")

    def _next_clean_target(self):
        """Vai ao ponto sujo ALCANÇÁVEL mais próximo (BFS a partir do robô)."""
        if self.p.mode != "LIMPEZA_TOTAL":
            return
        pl = self.sim.planner
        pl.map_kind = "planta"
        blocked = pl.blocked_grid()
        x, y, _ = self.sim.est_pose()
        s = nearest_free(blocked, m2cell(x, y))
        if s is None:
            self._finish_clean()
            return
        st = self.CLEAN_STEP
        seen = {s}
        q = deque([s])
        found = None
        while q:
            cx, cy = q.popleft()
            if (cx % st == 0 and cy % st == 0 and not self.clean[cy, cx]
                    and (cx, cy) not in self.skip):
                found = (cx, cy)
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nx, ny = cx + dx, cy + dy
                if (0 <= nx < GC and 0 <= ny < GR and (nx, ny) not in seen
                        and not blocked[ny, nx]):
                    seen.add((nx, ny))
                    q.append((nx, ny))
        if found is None:
            self._finish_clean()
            return
        self.clean_current = found
        self.bus.pub("/goal", {"goal": cell2m(*found), "name": "limpeza", "map": "planta"})
        self.sim.brain.start_goal()

    def send_goal(self, name, pos_m, kind="planta"):
        self.bus.pub("/goal", {"goal": pos_m, "name": name, "map": kind})
        self.sim.brain.start_goal()
        self.sim.log(f"Missão: ir para {name}")

    def goto_px(self, name, px, py):
        self.send_goal(name, px2m(px, py))

    def tick(self):
        p, br = self.sim.p, self.sim.brain
        idle = br.estado in ("OBJETIVO_ALCANÇADO", "SEM_ROTA", "PARADO")
        if p.mode != "LIMPEZA_TOTAL":
            self.period = 0.4

        if p.mode == "PATRULHA" and idle:
            self.wait += self.period
            if self.wait >= 1.5:
                self.wait = 0.0
                n = TOUR[self.tour_i % len(TOUR)]
                self.tour_i += 1
                self.goto_px(n, *COMODOS[n])

        elif p.mode == "LIMPEZA_TOTAL":
            self.clean_pct = float((self.clean & self.region).sum()) / self.region_total
            if idle:
                if self.clean_current is not None:
                    
                    self.skip.add(self.clean_current)
                    if br.estado == "SEM_ROTA":
                        self.sim.log("Ponto sem rota: pulando")
                    self.clean_current = None
                self._next_clean_target()

        elif p.mode == "EXPLORAR" and idle:
            if br.estado == "SEM_ROTA" and self.cur_frontier:
                self.blacklist.append(self.cur_frontier)
            elif br.estado == "OBJETIVO_ALCANÇADO" and self.cur_frontier:
                self.blacklist.append(self.cur_frontier)
            self.cur_frontier = None
            mp = self.sim.mapper
            x, y, _ = self.sim.est_pose()
            blocked = dilate(mp.occupied(), INFLATE_CELLS) | ~mp.seen
            f = mp.frontier(m2cell(x, y), blocked, self.blacklist)
            if f is None:
                p.mode = "OCIOSO"
                br.estado = "PARADO"
                self.done_explore = True
                self.sim.log(f"Exploração concluída! Cobertura {mp.coverage * 100:.0f}%")
            else:
                self.cur_frontier = f
                self.send_goal("fronteira", cell2m(*f), "aprendido")

# 8. SIMULAÇÃO (agrupa mundo, robô e nós)

class Sim:
    DT = 1 / 60

    def __init__(self):
        self.world = World()
        self.p = Params()
        self.logs = deque(maxlen=6)
        self.reset()

    def log(self, msg):
        self.logs.append(f"[{self.t:6.1f}s] {msg}")

    def reset(self, keep_boxes=False):
        if not keep_boxes:
            self.world.boxes = []
            self.world.rebuild()
        self.t = 0.0
        self.bus = Bus()
        x, y = px2m(*START_PX)
        self.robot = Robot(x, y, 0.0)
        self.p.mode = "OCIOSO"
        self.trail, self.odom_trail = [], []
        self.motor = MotorDriverNode(self)
        self.odom = EncoderOdomNode(self)
        self.gt = GroundTruthNode(self)
        self.lidar = LidarNode(self)
        self.perc = PerceptionNode(self)
        self.mapper = MapperNode(self)
        self.planner = PlannerNode(self)
        self.brain = BrainNode(self)
        self.mission = MissionNode(self)
        self.nodes = [self.odom, self.gt, self.lidar, self.mapper, self.planner,
                      self.brain, self.mission, self.motor]
        self.logs.clear()
        self.log("Zippy pronto. Use 1-6 para cômodos ou 7 para limpeza total.")

    def est_pose(self):
        if self.p.use_odom:
            o = self.odom
            return o.x, o.y, o.th
        r = self.robot
        return r.x, r.y, r.th

    def odom_error(self):
        r, o = self.robot, self.odom
        return math.hypot(r.x - o.x, r.y - o.y), math.degrees(normalizar_angulo(r.th - o.th))

    def step(self, dt=None):
        dt = dt or self.DT
        self.t += dt
        self.bus.tick(self.t)
        self.robot.step(dt, self.world, self.p.noise)
        if self.p.mode == "LIMPEZA_TOTAL":
            self.mission.mark_clean()
        for n in self.nodes:
            n.acc += dt
            if n.acc >= n.period - 1e-9:
                n.acc -= n.period
                n.tick()
        if not self.trail or math.hypot(self.robot.x - self.trail[-1][0],
                                        self.robot.y - self.trail[-1][1]) > 0.05:
            self.trail.append((self.robot.x, self.robot.y))
            self.odom_trail.append((self.odom.x, self.odom.y))
            if len(self.trail) > 1500:
                self.trail.pop(0)
                self.odom_trail.pop(0)

    def go_to_room(self, name):
        self.p.mode = "IR"
        self.mission.goto_px(name, *COMODOS[name])

    def go_to_px(self, px, py):
        self.p.mode = "IR"
        self.mission.goto_px("ponto clicado", px, py)

    def stop(self):
        self.p.mode = "OCIOSO"
        self.brain.estado = "PARADO"
        self.planner.clear()
        self.brain.path = []

# 9. INTERFACE (desenho + teclado/mouse)

COR_ESTADO = {
    "PARADO": (150, 150, 160), "IR_PARA_ALVO": (60, 200, 120),
    "DESVIAR_OBSTACULO": (255, 170, 40), "OBJETIVO_ALCANÇADO": (90, 160, 255),
    "SEM_ROTA": (240, 80, 80), "MANUAL": (200, 120, 255),
}
HELP = [
    "Clique: ir até lá | 1-6: cômodos | 7: limpeza total | P: patrulha | E: explorar | TAB: manual (setas)",
    "Botão dir.: caixa surpresa | V: vista (planta/mapa aprendido/grade) | L: lidar | B: barreiras | O: odometria | U: guiar pela odom.",
    "Z: relocalizar odom. | N: ruído | ESPAÇO: pausa | [ ]: velocidade | S: parar | R: reiniciar | ESC: sair",
]

class App:
    def __init__(self, scale=None):
        pygame.display.init()
        pygame.font.init()
        info = pygame.display.Info()
        if scale is None:
            sw, sh = info.current_w or 1600, info.current_h or 900
            scale = min(0.85, (sw - 40 - PANEL_W) / IMG_W, (sh - 110 - BOTTOM_H) / IMG_H)
        self.S = max(0.45, scale)
        self.iw, self.ih = int(IMG_W * self.S), int(IMG_H * self.S)
        self.W, self.H = self.iw + PANEL_W, self.ih + BOTTOM_H
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Zippy na Casa - Simulador de Robô Móvel Inteligente (IMR)")
        self.clock = pygame.time.Clock()
        fnt = "dejavusansmono,consolas,menlo,couriernew,monospace"
        self.f = pygame.font.SysFont(fnt, 13)
        self.fb = pygame.font.SysFont(fnt, 15, bold=True)
        self.fs = pygame.font.SysFont(fnt, 11)
        self.ft = pygame.font.SysFont(fnt, 19, bold=True)
        self.sim = Sim()
        self.bg = None
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), IMG_FILE)
        if os.path.exists(path):
            img = pygame.image.load(path).convert()
            self.bg = pygame.transform.smoothscale(img, (self.iw, self.ih))
        self.view = 0               
        self.show_lidar, self.show_barriers, self.show_odom = True, False, True
        self.paused, self.speed = False, 1
        self.map_surf, self.map_t = None, -9.0
        self.mouse_world = None

    def p2s(self, px, py):
        return int(px * self.S), int(py * self.S)

    def m2s(self, x, y):
        return self.p2s(*m2px(x, y))

    def s2p(self, sx, sy):
        return sx / self.S, sy / self.S

    def learned_surface(self):
        sim = self.sim
        if sim.t - self.map_t < 0.25 and self.map_surf is not None:
            return self.map_surf
        self.map_t = sim.t
        mp = sim.mapper
        img = np.empty((GR, GC, 3), np.uint8)
        img[:] = (95, 100, 112)                                   
        img[mp.free()] = (236, 234, 224)                          
        occ = mp.occupied()
        img[occ] = (24, 24, 30)                                   
        novos = occ & ~sim.planner.static_near
        img[novos] = (255, 140, 0)                                
        self.map_surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        return self.map_surf

    def draw_world(self):
        sc, sim = self.screen, self.sim
        if self.view == 1:
            m = pygame.transform.scale(self.learned_surface(), (self.iw, self.ih))
            sc.blit(m, (0, 0))
        else:
            if self.bg:
                sc.blit(self.bg, (0, 0))
            else:
                sc.fill((226, 205, 180), (0, 0, self.iw, self.ih))
                for _, x0, y0, x1, y1 in PAREDES + MOVEIS:
                    pygame.draw.rect(sc, (90, 70, 60), (*self.p2s(x0, y0), int((x1 - x0) * self.S),
                                                        int((y1 - y0) * self.S)))
        
        if self.view != 1 and sim.mission.clean.any():
            sc.blit(pygame.transform.scale(
                self._grid_surface(sim.mission.clean, (0, 200, 255, 70)),
                (self.iw, self.ih)), (0, 0))
        if self.view == 2:
            ov = pygame.Surface((self.iw, self.ih), pygame.SRCALPHA)
            ib = pygame.transform.scale(self._grid_surface(sim.planner.static_blocked, (255, 60, 60, 110)),
                                        (self.iw, self.ih))
            ov.blit(ib, (0, 0))
            if sim.planner.new_obs.any():
                nb = dilate(sim.planner.new_obs, INFLATE_CELLS)
                ov.blit(pygame.transform.scale(self._grid_surface(nb, (255, 160, 0, 150)),
                                               (self.iw, self.ih)), (0, 0))
            sc.blit(ov, (0, 0))
        if self.show_barriers:
            ov = pygame.Surface((self.iw, self.ih), pygame.SRCALPHA)
            blocked = pool(sim.world.occ)
            ib = pygame.transform.scale(
                self._grid_surface(blocked, (255, 45, 45, 85)), (self.iw, self.ih))
            ov.blit(ib, (0, 0))
            sc.blit(ov, (0, 0))

        for r in sim.world.boxes:
            rr = pygame.Rect(*self.p2s(r.left, r.top), int(r.w * self.S), int(r.h * self.S))
            pygame.draw.rect(sc, (210, 120, 40), rr)
            pygame.draw.rect(sc, (255, 220, 160), rr, 2)
            sc.blit(self.fs.render("caixa", True, (40, 20, 0)), (rr.x + 3, rr.centery - 6))
        
        for n, (px, py) in COMODOS.items():
            pos = self.p2s(px, py)
            pygame.draw.circle(sc, (255, 255, 255), pos, 4, 1)
            t = self.fs.render(n, True, (255, 255, 255), (30, 30, 40))
            sc.blit(t, (pos[0] - t.get_width() // 2, pos[1] + 6))
        
        if len(sim.trail) > 1:
            pygame.draw.lines(sc, (80, 230, 120), False, [self.m2s(*p) for p in sim.trail], 2)
        if self.show_odom and len(sim.odom_trail) > 1:
            pygame.draw.lines(sc, (255, 130, 30), False, [self.m2s(*p) for p in sim.odom_trail], 1)
        
        pts = sim.brain.path
        if len(pts) > 1:
            sp = [self.m2s(*p) for p in pts]
            pygame.draw.lines(sc, (0, 220, 255), False, sp, 3)
            for i, q in enumerate(sp):
                pygame.draw.circle(sc, (0, 255, 255) if i >= sim.brain.idx else (0, 120, 140), q, 4)
            pygame.draw.circle(sc, (255, 215, 0), sp[-1], 9, 3)
        
        r = sim.robot
        rp = self.m2s(r.x, r.y)
        scan = sim.bus.last.get("/scan")
        if self.show_lidar and scan is not None:
            for k in range(0, 360, 2):
                d = scan[k]
                if d == 0.0:
                    continue
                dd = LIDAR_RANGE if not np.isfinite(d) else d
                a = r.th + math.radians(k)
                e = self.m2s(r.x + dd * math.cos(a), r.y + dd * math.sin(a))
                hit = np.isfinite(d) and d < LIDAR_RANGE
                pygame.draw.line(sc, (255, 210, 60) if hit else (60, 200, 120), rp, e, 1)
                if hit:
                    pygame.draw.circle(sc, (255, 80, 60), e, 2)
        
        if self.show_odom:
            o = sim.odom
            op = self.m2s(o.x, o.y)
            pygame.draw.circle(sc, (255, 130, 30), op, max(4, int(ROBOT_R * PX_PER_M * self.S)), 2)
            pygame.draw.line(sc, (255, 130, 30), op, (op[0] + 16 * math.cos(o.th) * self.S * 1.2,
                                                       op[1] - 16 * math.sin(o.th) * self.S * 1.2), 2)
        rad = max(5, int(ROBOT_R * PX_PER_M * self.S))
        pygame.draw.circle(sc, (0, 170, 255), rp, rad)
        pygame.draw.circle(sc, (255, 255, 255), rp, rad, 2)
        fx, fy = rp[0] + (rad + 6) * math.cos(r.th), rp[1] - (rad + 6) * math.sin(r.th)
        pygame.draw.line(sc, (255, 60, 60), rp, (fx, fy), 3)
        
        est = sim.brain.estado
        t = self.fs.render(est, True, (0, 0, 0), COR_ESTADO.get(est, (200, 200, 200)))
        sc.blit(t, (rp[0] - t.get_width() // 2, rp[1] - rad - 18))
        if self.view == 1:
            sc.blit(self.fb.render("MAPA APRENDIDO (grade de ocupação)", True, (255, 255, 255), (30, 30, 40)), (8, 8))
        elif self.view == 2:
            sc.blit(self.fb.render("GRADE INFLADA p/ planejamento A*", True, (255, 255, 255), (30, 30, 40)), (8, 8))

    def _grid_surface(self, grid, rgba):
        arr = np.zeros((GR, GC, 4), np.uint8)
        arr[grid] = rgba
        s = pygame.Surface((GC, GR), pygame.SRCALPHA)
        pygame.surfarray.pixels3d(s)[:] = arr[..., :3].swapaxes(0, 1)
        pygame.surfarray.pixels_alpha(s)[:] = arr[..., 3].swapaxes(0, 1)
        return s

    def text(self, s, x, y, col=(225, 225, 230), font=None):
        self.screen.blit((font or self.f).render(s, True, col), (x, y))

    def draw_panel(self):
        sc, sim = self.screen, self.sim
        x0 = self.iw
        pygame.draw.rect(sc, (24, 26, 32), (x0, 0, PANEL_W, self.H))
        y = 8
        self.text("ZIPPY  ·  Robô móvel inteligente", x0 + 12, y, (255, 255, 255), self.ft)
        y += 30
        est = sim.brain.estado
        pygame.draw.rect(sc, COR_ESTADO.get(est, (150, 150, 150)), (x0 + 12, y, PANEL_W - 24, 26), border_radius=5)
        self.text("FSM: " + est, x0 + 20, y + 5, (10, 10, 10), self.fb)
        y += 34
        loc = "ODOMETRIA (deriva!)" if sim.p.use_odom else "ground truth"
        self.text(f"Modo: {sim.p.mode:<9} Loc.: {loc}", x0 + 12, y)
        y += 16
        self.text(f"Ruído: {'ON' if sim.p.noise else 'OFF'}   Sim: {self.speed}x"
                  f"{'  [PAUSADO]' if self.paused else ''}", x0 + 12, y)
        y += 22
        r, o = sim.robot, sim.odom
        err, derr = sim.odom_error()
        self.text("TELEMETRIA", x0 + 12, y, (130, 200, 255), self.fb)
        y += 18
        for ln in (
            f"Real  x={r.x:5.2f} y={r.y:5.2f} θ={math.degrees(r.th):6.1f}°",
            f"Odom  x={o.x:5.2f} y={o.y:5.2f} θ={math.degrees(o.th):6.1f}°",
            f"Erro odom: {err:5.2f} m  {derr:+6.1f}°",
            f"v={r.v:5.2f} m/s  ω={r.w:5.2f} rad/s",
            f"roda E={r.vl:5.2f}  D={r.vr:5.2f} (lim {MAX_WHEEL})",
            f"Distância: {r.dist:5.1f} m   Colisões: {r.collisions}",
            f"Limpeza: {sim.mission.clean_pct * 100:5.1f}%",
        ):
            self.text(ln, x0 + 12, y)
            y += 15
        y += 8
        self.text("LiDAR (setores filtrados)", x0 + 12, y, (130, 200, 255), self.fb)
        y += 18
        sec = sim.brain.sec
        for nome in ("esquerda", "frente", "direita"):
            d = sec[nome]
            w = int((PANEL_W - 130) * min(d, 3.0) / 3.0)
            col = (240, 80, 80) if d < D_CRITICA else (255, 200, 60) if d < 0.8 else (80, 220, 130)
            self.text(f"{nome[:6]:<6}{d:4.2f}", x0 + 12, y)
            pygame.draw.rect(sc, (50, 52, 60), (x0 + 112, y + 2, PANEL_W - 130, 10))
            pygame.draw.rect(sc, col, (x0 + 112, y + 2, w, 10))
            y += 15
        y += 8
        self.text("TÓPICOS (estilo ROS 2)", x0 + 12, y, (130, 200, 255), self.fb)
        y += 18
        for tp in ("/scan", "/scan_sectors", "/odom", "/ground_truth", "/path", "/cmd_vel", "/wheel_cmd", "/map", "/zippy/status"):
            hz = sim.bus.rate.get(tp, 0.0)
            fresh = (sim.bus.now - sim.bus.stamp.get(tp, -9)) < 0.15
            pygame.draw.circle(sc, (90, 240, 140) if fresh else (70, 74, 82), (x0 + 18, y + 7), 4)
            msg = sim.bus.last.get(tp)
            extra = ""
            if tp == "/cmd_vel" and msg:
                extra = f"v={msg[0]:.2f} ω={msg[1]:.2f}"
            elif tp == "/scan" and msg is not None:
                extra = "360 raios"
            elif tp == "/path" and msg:
                extra = f"{len(msg['pts'])} wp"
            elif tp == "/map" and msg:
                extra = f"cob={msg['coverage'] * 100:.0f}%"
            self.text(f"{tp:<14}{hz:5.1f}Hz {extra}", x0 + 28, y, (200, 200, 208), self.fs)
            y += 14
        y += 8

        map_w = PANEL_W - 64
        mh = int(map_w * MAPH / MAPW)

        map_x = x0 + 12
        map_y = self.H - mh - 28

        self.text(
            f"MAPA APRENDIDO  cobertura {sim.mapper.coverage * 100:3.0f}%",
            map_x,
            map_y - 20,
            (130, 200, 255),
            self.fb
        )

        m = pygame.transform.smoothscale(
            self.learned_surface(),
            (map_w, mh)
        )

        sc.blit(m, (map_x, map_y))

        pygame.draw.rect(
            sc,
            (90, 90, 100),
            (map_x, map_y, map_w, mh),
            1
        )

        k = map_w / MAPW
        px, py = m2px(r.x, r.y)

        pygame.draw.circle(
            sc,
            (0, 170, 255),
            (
                int(map_x + px * k),
                int(map_y + py * k)
            ),
            4
        )

    def draw_bottom(self):
        sc = self.screen
        pygame.draw.rect(sc, (16, 18, 22), (0, self.ih, self.iw, BOTTOM_H))
        y = self.ih + 4
        for ln in list(self.sim.logs)[-3:]:
            self.text(ln, 8, y, (255, 225, 140), self.fs)
            y += 13
        y += 3
        for ln in HELP:
            self.text(ln, 8, y, (150, 160, 175), self.fs)
            y += 12

    def handle(self, e):
        sim = self.sim
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.MOUSEBUTTONDOWN and e.pos[0] < self.iw and e.pos[1] < self.ih:
            px, py = self.s2p(*e.pos)
            if e.button == 1:
                if 0 <= px < IMG_W and 0 <= py < IMG_H:
                    sim.go_to_px(px, py)
            elif e.button == 3:
                res = sim.world.toggle_box(px, py, (sim.robot.x, sim.robot.y))
                msgs = {"criada": "Caixa surpresa colocada! (o robô vai perceber pelo LiDAR)",
                        "removida": "Caixa removida", "ocupado": "Aí já tem móvel/parede",
                        "perto": "Muito perto do robô"}
                sim.log(msgs[res])
        elif e.type == pygame.KEYDOWN:
            k = e.key
            names = list(COMODOS.keys())
            if k == pygame.K_ESCAPE:
                return False
            elif pygame.K_1 <= k <= pygame.K_6:
                sim.go_to_room(names[k - pygame.K_1])
            elif k == pygame.K_7:
                sim.mission.start_total_clean()
            elif k == pygame.K_p:
                sim.p.mode = "PATRULHA"
                sim.brain.estado = "PARADO" if sim.brain.estado == "MANUAL" else sim.brain.estado
                sim.log("Modo patrulha: Cozinha > Jantar > Sala > Banheiro > Quarto > Entrada")
            elif k == pygame.K_e:
                sim.p.mode = "EXPLORAR"
                sim.mission.blacklist, sim.mission.cur_frontier = [], None
                sim.brain.estado = "PARADO"
                sim.log("Exploração autônoma por fronteiras (sem usar o mapa pronto)")
            elif k == pygame.K_TAB:
                if sim.p.mode == "MANUAL":
                    sim.stop()
                    sim.log("Modo automático")
                else:
                    sim.p.mode = "MANUAL"
                    sim.planner.clear()
                    sim.brain.path = []
                    sim.log("Modo MANUAL: use as setas")
            elif k == pygame.K_s:
                sim.stop()
                sim.log("Parado")
            elif k == pygame.K_v:
                self.view = (self.view + 1) % 3
            elif k == pygame.K_l:
                self.show_lidar = not self.show_lidar
            elif k == pygame.K_b:
                self.show_barriers = not self.show_barriers
            elif k == pygame.K_o:
                self.show_odom = not self.show_odom
            elif k == pygame.K_u:
                sim.p.use_odom = not sim.p.use_odom
                sim.log("Guiando pela ODOMETRIA (vai derivar)" if sim.p.use_odom else "Guiando pelo ground truth")
            elif k == pygame.K_z:
                sim.odom.relocalize()
                sim.log("Odometria relocalizada (reset da deriva)")
            elif k == pygame.K_n:
                sim.p.noise = not sim.p.noise
                sim.log("Ruído " + ("LIGADO" if sim.p.noise else "DESLIGADO"))
            elif k == pygame.K_SPACE:
                self.paused = not self.paused
            elif k == pygame.K_RIGHTBRACKET:
                self.speed = min(8, self.speed * 2)
            elif k == pygame.K_LEFTBRACKET:
                self.speed = max(1, self.speed // 2)
            elif k == pygame.K_r:
                sim.reset()
                self.map_surf = None
        return True

    def manual_keys(self):
        keys = pygame.key.get_pressed()
        v = w = 0.0
        if keys[pygame.K_UP]:
            v += 0.6
        if keys[pygame.K_DOWN]:
            v -= 0.4
        if keys[pygame.K_LEFT]:
            w += 1.8
        if keys[pygame.K_RIGHT]:
            w -= 1.8
        self.sim.p.manual = (v, w)

    def run(self):
        running = True
        while running:
            for e in pygame.event.get():
                running = self.handle(e) and running
            if self.sim.p.mode == "MANUAL":
                self.manual_keys()
            if not self.paused:
                for _ in range(self.speed):
                    self.sim.step()
            self.screen.fill((0, 0, 0))
            self.draw_world()
            self.draw_panel()
            self.draw_bottom()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

def main():
    App().run()

if __name__ == "__main__":
    main()