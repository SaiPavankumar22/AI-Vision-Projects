import sys, traceback, time, random, math
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto

print(f"[DEBUG] Python {sys.version}")

for pkg, pip_name in [("cv2","opencv-python"),("mediapipe","mediapipe==0.10.9"),("numpy","numpy")]:
    try:
        mod = __import__(pkg)
        print(f"[OK]    {pip_name} {getattr(mod,'__version__','?')}")
    except ImportError:
        print(f"[FATAL] Missing: pip install {pip_name}"); sys.exit(1)

import cv2, mediapipe as mp, numpy as np

GRID          = 3
PUZZLE_SIZE   = 450
CELL          = PUZZLE_SIZE // GRID

PINCH_THRESH      = 0.055
PINCH_RELEASE     = 0.075
PALM_HOLD_SEC     = 1.0
SNAP_DIST         = CELL * 0.6
SOLVED_SHOW       = 5.0

PINCH_LOCK_FRAMES = 8

C_WHITE  = (255,255,255)
C_BLACK  = (0,0,0)
C_GREEN  = (0,220,100)
C_YELLOW = (0,230,230)
C_CYAN   = (230,220,0)
C_ORANGE = (0,140,255)
C_RED    = (60,60,240)
C_BLUE   = (220,100,0)
C_PURPLE = (220,60,200)


class State(Enum):
    WAITING_FOR_PALM = auto()
    CAPTURE_FACE     = auto()
    SHUFFLE_PUZZLE   = auto()
    PLAYING          = auto()
    SOLVED           = auto()


@dataclass
class PuzzleBlock:
    idx:         int
    image:       np.ndarray
    correct_pos: tuple
    current_pos: tuple
    rect:        list
    dragging:    bool  = False
    drag_offset: tuple = (0, 0)
    _prev_pos:   tuple = (0, 0)
    vis_x:       float = 0.0
    vis_y:       float = 0.0
    glow_phase:  float = 0.0


@dataclass
class Particle:
    x: float; y: float
    vx: float; vy: float
    life: float
    color: tuple
    size: float


@dataclass
class GameState:
    state:          State = State.WAITING_FOR_PALM
    blocks:         list  = field(default_factory=list)
    face_img:       Optional[np.ndarray] = None
    selected_block: Optional[int] = None
    start_time:     float = 0.0
    elapsed:        float = 0.0
    palm_start:     float = 0.0
    palm_active:    bool  = False
    solved_time:    float = 0.0
    drag_x:         float = 0.0
    drag_y:         float = 0.0
    particles:      list  = field(default_factory=list)
    pinch_locked:   bool  = False
    pinch_frames:   int   = 0
    release_frames: int   = 0
    solved_anim_t:  float = 0.0
    confetti:       list  = field(default_factory=list)


class HandTracker:
    def __init__(self):
        self._mph  = mp.solutions.hands
        self.hands = self._mph.Hands(False, 1, 1, 0.7, 0.6)
        self.draw  = mp.solutions.drawing_utils
        self.style = mp.solutions.drawing_styles
        print("[OK]    HandTracker ready.")

    def process(self, rgb): return self.hands.process(rgb)
    def draw_lm(self, f, hl):
        self.draw.draw_landmarks(f, hl, self._mph.HAND_CONNECTIONS,
            self.style.get_default_hand_landmarks_style(),
            self.style.get_default_hand_connections_style())
    def close(self): self.hands.close()


class FaceDetector:
    def __init__(self):
        self._mpf    = mp.solutions.face_detection
        self.det     = self._mpf.FaceDetection(0, 0.5)
        self._miss   = 0
        print("[OK]    FaceDetector ready.")

    def detect(self, rgb):
        h, w = rgb.shape[:2]
        res  = self.det.process(rgb)
        if not res.detections: return []
        boxes = []
        for d in res.detections:
            bb = d.location_data.relative_bounding_box
            x,y,bw,bh = int(bb.xmin*w),int(bb.ymin*h),int(bb.width*w),int(bb.height*h)
            pad = int(min(bw,bh)*0.15)
            boxes.append((max(0,x-pad),max(0,y-pad),
                          min(w-x,bw+2*pad),min(h-y,bh+2*pad)))
        return boxes
    def close(self): self.det.close()


def lm_arr(hl, w, h):
    return np.array([[int(l.x*w),int(l.y*h)] for l in hl.landmark], np.float32)

def open_palm(hl, w, h):
    p = lm_arr(hl, w, h)
    te = abs(p[4][0]-p[0][0]) > abs(p[3][0]-p[0][0])
    return te and all(p[t][1]<p[pip][1] for t,pip in [(8,6),(12,10),(16,14),(20,18)])

def pinch_info(hl, w, h):
    l = hl.landmark
    dx,dy = l[4].x-l[8].x, l[4].y-l[8].y
    dist  = (dx**2+dy**2)**0.5
    mx,my = int((l[4].x+l[8].x)/2*w), int((l[4].y+l[8].y)/2*h)
    return dist, mx, my


def build_puzzle(face, origin):
    ox, oy = origin
    sq     = cv2.resize(face, (PUZZLE_SIZE, PUZZLE_SIZE))
    blocks = []
    for r in range(GRID):
        for c in range(GRID):
            img = sq[r*CELL:(r+1)*CELL, c*CELL:(c+1)*CELL].copy()
            bx, by = ox+c*CELL, oy+r*CELL
            b = PuzzleBlock(r*GRID+c, img, (r,c), (r,c), [bx,by,CELL,CELL])
            b.vis_x, b.vis_y = float(bx), float(by)
            blocks.append(b)
    pos = [(b.current_pos, b.rect[:]) for b in blocks]
    random.shuffle(pos)
    for i,blk in enumerate(blocks):
        blk.current_pos = pos[i][0]
        r,c = blk.current_pos
        blk.rect = [ox+c*CELL, oy+r*CELL, CELL, CELL]
        blk.vis_x, blk.vis_y = float(blk.rect[0]), float(blk.rect[1])
    print("[DEBUG] Puzzle built & shuffled.")
    return blocks

def solved(blocks): return all(b.correct_pos==b.current_pos for b in blocks)

def snap(block, origin):
    ox, oy = origin
    cx = block.rect[0]+CELL//2-ox
    cy = block.rect[1]+CELL//2-oy
    col = max(0,min(GRID-1, round((cx-CELL//2)/CELL)))
    row = max(0,min(GRID-1, round((cy-CELL//2)/CELL)))
    tx,ty = ox+col*CELL, oy+row*CELL
    if ((block.rect[0]-tx)**2+(block.rect[1]-ty)**2)**0.5 < SNAP_DIST:
        block.rect[0],block.rect[1] = tx,ty
        block.current_pos = (row,col)
    else:
        r,c = block.current_pos
        block.rect[0],block.rect[1] = ox+c*CELL, oy+r*CELL

def swap_slot(blocks, moving, origin):
    ox,oy = origin
    tp = moving.current_pos
    for b in blocks:
        if b.idx!=moving.idx and b.current_pos==tp and not b.dragging:
            b.current_pos = moving._prev_pos
            r,c = b.current_pos
            b.rect[0],b.rect[1] = ox+c*CELL, oy+r*CELL
            break


CONFETTI_COLORS = [(0,230,255),(0,200,100),(200,50,255),(0,140,255),(0,230,180)]

def spawn_confetti(gs, fw, fh, n=120):
    gs.confetti = []
    for _ in range(n):
        gs.confetti.append(Particle(
            x=random.uniform(0,fw), y=random.uniform(-50,0),
            vx=random.uniform(-2,2), vy=random.uniform(3,9),
            life=1.0,
            color=random.choice(CONFETTI_COLORS),
            size=random.uniform(6,14),
        ))

def spawn_snap_burst(gs, cx, cy, n=18):
    for _ in range(n):
        angle = random.uniform(0, 2*math.pi)
        speed = random.uniform(2, 7)
        gs.particles.append(Particle(
            x=cx, y=cy,
            vx=math.cos(angle)*speed, vy=math.sin(angle)*speed,
            life=1.0, color=random.choice(CONFETTI_COLORS), size=random.uniform(3,7)
        ))

def update_particles(gs, dt):
    alive = []
    for p in gs.particles:
        p.x += p.vx; p.y += p.vy
        p.vy += 0.15
        p.life -= dt * 2.5
        if p.life > 0: alive.append(p)
    gs.particles = alive

    alive2 = []
    for p in gs.confetti:
        p.x += p.vx; p.y += p.vy
        p.vx += random.uniform(-0.2,0.2)
        p.life -= dt * 0.25
        if p.life > 0 and p.y < 800: alive2.append(p)
    gs.confetti = alive2

def draw_particles(frame, gs):
    for p in gs.particles:
        alpha = max(0, p.life)
        r2 = int(p.size * alpha)
        if r2 < 1: continue
        cx,cy = int(p.x), int(p.y)
        overlay = frame.copy()
        cv2.circle(overlay, (cx,cy), r2, p.color, -1)
        cv2.addWeighted(overlay, alpha*0.85, frame, 1-alpha*0.85, 0, frame)

    for p in gs.confetti:
        cx,cy = int(p.x), int(p.y)
        if 0<=cx<frame.shape[1] and 0<=cy<frame.shape[0]:
            hw = max(2,int(p.size))
            pts = np.array([[cx-hw,cy],[cx,cy-hw],[cx+hw,cy],[cx,cy+hw]], np.int32)
            cv2.fillPoly(frame, [pts], p.color)


def put_center(img, txt, cx, cy, scale=0.8, color=C_WHITE, thick=2):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw,th),_ = cv2.getTextSize(txt, font, scale, thick)
    cv2.putText(img, txt, (cx-tw//2, cy+th//2), font, scale, color, thick, cv2.LINE_AA)

def draw_glow_rect(frame, x, y, w, h, color, layers=6, base_alpha=0.4):
    for i in range(layers, 0, -1):
        t  = i * 2
        a  = base_alpha * (1 - i/layers) * 0.6
        ov = frame.copy()
        cv2.rectangle(ov, (x-t,y-t), (x+w+t,y+h+t), color, 2)
        cv2.addWeighted(ov, a, frame, 1-a, 0, frame)
    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

def draw_grid_ghost(frame, origin):
    ox,oy = origin
    for r in range(GRID):
        for c in range(GRID):
            gx,gy = ox+c*CELL, oy+r*CELL
            ov = frame.copy()
            cv2.rectangle(ov,(gx,gy),(gx+CELL,gy+CELL),(40,40,60),-1)
            cv2.addWeighted(ov,0.3,frame,0.7,0,frame)
            cv2.rectangle(frame,(gx,gy),(gx+CELL,gy+CELL),(80,80,120),1)

def draw_blocks(frame, blocks, sel_idx, origin, t_now):
    ox,oy = origin
    draw_grid_ghost(frame, origin)

    for blk in blocks:
        if blk.idx == sel_idx: continue
        blk.vis_x += (blk.rect[0] - blk.vis_x) * 0.25
        blk.vis_y += (blk.rect[1] - blk.vis_y) * 0.25
        x,y = int(blk.vis_x), int(blk.vis_y)
        x = max(0,min(frame.shape[1]-CELL,x))
        y = max(0,min(frame.shape[0]-CELL,y))
        frame[y:y+CELL,x:x+CELL] = blk.image
        cv2.rectangle(frame,(x,y),(x+CELL,y+CELL),(100,100,140),1)
        if blk.correct_pos == blk.current_pos:
            cv2.rectangle(frame,(x,y),(x+CELL,y+CELL),(0,180,80),2)

    if sel_idx is not None:
        blk = next((b for b in blocks if b.idx==sel_idx), None)
        if blk:
            blk.glow_phase = (blk.glow_phase + 0.08) % (2*math.pi)
            pulse = 0.5 + 0.5*math.sin(blk.glow_phase)
            blk.vis_x += (blk.rect[0] - blk.vis_x) * 0.55
            blk.vis_y += (blk.rect[1] - blk.vis_y) * 0.55
            x,y = int(blk.vis_x), int(blk.vis_y)
            x = max(0,min(frame.shape[1]-CELL,x))
            y = max(0,min(frame.shape[0]-CELL,y))
            sv = frame.copy()
            cv2.rectangle(sv,(x+5,y+5),(x+CELL+5,y+CELL+5),(10,10,10),-1)
            cv2.addWeighted(sv,0.45,frame,0.55,0,frame)
            frame[y:y+CELL,x:x+CELL] = blk.image
            gc = tuple(int(c*pulse + 200*(1-pulse)) for c in C_CYAN)
            draw_glow_rect(frame, x, y, CELL, CELL, gc, layers=7, base_alpha=0.5)

def draw_timer(frame, elapsed, x, y):
    m,s = int(elapsed)//60, int(elapsed)%60
    ms  = int((elapsed%1)*10)
    cv2.putText(frame,f"{m:02d}:{s:02d}.{ms}",(x,y),
                cv2.FONT_HERSHEY_DUPLEX,0.9,C_GREEN,2,cv2.LINE_AA)

def draw_palm_bar(frame, prog, fw):
    bw = int(fw*0.4); bh=14
    bx = (fw-bw)//2; by = frame.shape[0]-40
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(40,40,40),-1)
    fill = int(bw*min(prog,1.0))
    for i in range(fill):
        ratio = i/max(bw,1)
        col = (0, int(150+80*ratio), int(80+140*ratio))
        cv2.line(frame,(bx+i,by),(bx+i,by+bh),col,1)
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(100,100,100),1)
    put_center(frame,"Hold open palm to start / reset",fw//2,by-12,
               scale=0.55,color=C_YELLOW,thick=1)

def draw_pinch_cursor(frame, px, py, is_pinching):
    if is_pinching:
        cv2.circle(frame,(px,py),14,C_CYAN,2)
        cv2.circle(frame,(px,py),5,C_CYAN,-1)
        cv2.circle(frame,(px,py),14,(255,255,255),1)
    else:
        cv2.circle(frame,(px,py),10,(180,180,180),1)

def draw_hud(frame, state, fw):
    labels = {
        State.WAITING_FOR_PALM:"Show Open Palm to Start",
        State.CAPTURE_FACE:    "Detecting Face...",
        State.SHUFFLE_PUZZLE:  "Shuffling...",
        State.PLAYING:         "Pinch + Drag to Move Blocks",
        State.SOLVED:          "SOLVED!",
    }
    txt = labels.get(state,"")
    (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    ov = frame.copy()
    cv2.rectangle(ov,(6,6),(tw+18,th+18),(20,20,30),-1)
    cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
    cv2.putText(frame,txt,(12,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,C_YELLOW,2,cv2.LINE_AA)

def draw_solved_screen(frame, elapsed, anim_t):
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(frame.shape[1],frame.shape[0]),(5,5,25),-1)
    cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
    cx,cy = frame.shape[1]//2, frame.shape[0]//2
    scale = 1.3 + 0.12*math.sin(anim_t*4)
    put_center(frame,"PUZZLE SOLVED!", cx, cy-55, scale, C_GREEN, 3)
    m,s = int(elapsed)//60, int(elapsed)%60
    put_center(frame,f"Time: {m:02d}:{s:02d}", cx, cy+20, 1.0, C_YELLOW, 2)
    put_center(frame,"Show open palm to play again", cx, cy+80, 0.6, C_WHITE, 1)

def draw_debug(frame, fps, hand, palm, pinch, dist, face, state):
    lines = [
        f"FPS:{fps:.0f}  Hand:{'Y' if hand else 'N'}  Palm:{'Y' if palm else 'N'}",
        f"Pinch:{'YES' if pinch else 'no '} dist={dist:.3f} thr={PINCH_THRESH}",
        f"Face:{'OK' if face else 'NOT FOUND'}  State:{state.name}",
        "[q]quit [r]reset [d]debug",
    ]
    ph = len(lines)*20+10; pw=360
    x0,y0 = 8, frame.shape[0]-ph-8
    ov = frame.copy()
    cv2.rectangle(ov,(x0,y0),(x0+pw,y0+ph),(15,15,15),-1)
    cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
    for i,ln in enumerate(lines):
        col = C_RED if ("NOT" in ln or ("no" in ln and "Pinch" in ln)) else C_WHITE
        cv2.putText(frame,ln,(x0+6,y0+18+i*20),cv2.FONT_HERSHEY_SIMPLEX,0.46,col,1,cv2.LINE_AA)


def find_cam():
    print("[DEBUG] Scanning cameras 0-3...")
    for i in range(4):
        c = cv2.VideoCapture(i)
        if c.isOpened():
            r,f = c.read(); c.release()
            if r and f is not None:
                print(f"[OK]    Camera {i} works."); return i
        c.release()
    return -1


class FacePuzzleApp:
    def __init__(self, cam_idx):
        self.cap = cv2.VideoCapture(cam_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        if not self.cap.isOpened(): raise RuntimeError("Cannot open camera.")
        r,t = self.cap.read()
        if not r: raise RuntimeError("Cannot read frames.")
        print(f"[OK]    Camera {cam_idx}: {t.shape[1]}x{t.shape[0]}")

        self.ht   = HandTracker()
        self.fd   = FaceDetector()
        self.gs   = GameState()
        self.origin = None
        self.debug  = True
        self._pt    = time.time()
        self._fps   = 0.0
        self._miss  = 0
        self._shuf_ts = None

    def _reset(self):
        print("[DEBUG] Reset → WAITING_FOR_PALM")
        self.gs = GameState()
        self._miss = 0
        self._shuf_ts = None

    def _try_capture(self, frame):
        h,w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        boxes = self.fd.detect(rgb)
        if not boxes:
            self._miss += 1
            if self._miss%30==0:
                print(f"[DEBUG] No face {self._miss} frames — ensure good lighting & face camera.")
            return False
        self._miss = 0
        x,y,bw,bh = boxes[0]
        side = max(bw,bh)
        cx,cy = x+bw//2, y+bh//2
        x1,y1 = max(0,cx-side//2), max(0,cy-side//2)
        x2,y2 = min(w,x1+side), min(h,y1+side)
        face  = frame[y1:y2,x1:x2]
        if face.size==0: return False
        print(f"[DEBUG] Face captured {face.shape[1]}x{face.shape[0]}")
        self.gs.face_img = face
        self.gs.blocks   = build_puzzle(face, self.origin)
        self.gs.state    = State.SHUFFLE_PUZZLE
        self._shuf_ts    = time.time()
        return True

    def _playing(self, frame, pinch_raw, px, py, dist):
        gs  = self.gs
        org = self.origin

        if pinch_raw:
            gs.pinch_frames   += 1
            gs.release_frames  = 0
        else:
            gs.release_frames += 1
            gs.pinch_frames    = 0

        pinch_confirmed   = gs.pinch_frames  >= 2
        release_confirmed = gs.release_frames >= 3

        if pinch_confirmed and not gs.pinch_locked and gs.selected_block is None:
            for blk in reversed(gs.blocks):
                x,y,w,h = blk.rect
                vx,vy = int(blk.vis_x), int(blk.vis_y)
                if vx<=px<=vx+w and vy<=py<=vy+h:
                    gs.selected_block = blk.idx
                    gs.pinch_locked   = True
                    blk.dragging      = True
                    blk._prev_pos     = blk.current_pos
                    blk.drag_offset   = (px-vx, py-vy)
                    gs.drag_x         = float(vx)
                    gs.drag_y         = float(vy)
                    blk.current_pos   = (-1,-1)
                    print(f"[DEBUG] Grabbed block {blk.idx}")
                    break

        if gs.pinch_locked and gs.selected_block is not None:
            blk = next((b for b in gs.blocks if b.idx==gs.selected_block), None)
            if blk:
                tx = px - blk.drag_offset[0]
                ty = py - blk.drag_offset[1]
                gs.drag_x += (tx - gs.drag_x) * 0.5
                gs.drag_y += (ty - gs.drag_y) * 0.5
                blk.rect[0] = int(gs.drag_x)
                blk.rect[1] = int(gs.drag_y)
                blk.vis_x   = gs.drag_x
                blk.vis_y   = gs.drag_y

        if gs.pinch_locked and release_confirmed and gs.selected_block is not None:
            blk = next((b for b in gs.blocks if b.idx==gs.selected_block), None)
            if blk:
                prev_cp = blk._prev_pos
                snap(blk, org)
                swap_slot(gs.blocks, blk, org)
                blk.dragging      = False
                if blk.current_pos != prev_cp:
                    cx = blk.rect[0]+CELL//2
                    cy = blk.rect[1]+CELL//2
                    spawn_snap_burst(gs, cx, cy)
                gs.selected_block = None
                gs.pinch_locked   = False
                gs.pinch_frames   = 0
                gs.release_frames = 0
                print(f"[DEBUG] Dropped block → {blk.current_pos}")

        gs.elapsed = time.time() - gs.start_time

        if solved(gs.blocks):
            gs.state       = State.SOLVED
            gs.solved_time = gs.elapsed
            spawn_confetti(gs, frame.shape[1], frame.shape[0])
            print(f"[DEBUG] SOLVED in {gs.solved_time:.2f}s")

    def run(self):
        print("\n[INFO] Running. [q]=quit [r]=reset [d]=debug\n")
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("[ERROR] Lost camera."); break

            frame = cv2.flip(frame,1)
            fh,fw = frame.shape[:2]
            now   = time.time()
            dt    = max(now - self._pt, 1e-6)
            self._fps = 1.0/dt
            self._pt  = now

            if self.origin is None:
                self.origin = (fw-PUZZLE_SIZE-20, (fh-PUZZLE_SIZE)//2)
                print(f"[DEBUG] Puzzle origin: {self.origin}")

            ox,oy = self.origin
            gs    = self.gs

            rgb  = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            hres = self.ht.process(rgb)

            palm=False; pinch_raw=False; dist=1.0; px=py=0
            hand=False
            if hres.multi_hand_landmarks:
                hand = True
                hl   = hres.multi_hand_landmarks[0]
                self.ht.draw_lm(frame, hl)
                palm = open_palm(hl,fw,fh)
                dist, px, py = pinch_info(hl,fw,fh)
                pinch_raw = dist < PINCH_THRESH

            face_now = False

            if gs.state == State.WAITING_FOR_PALM:
                if palm:
                    if not gs.palm_active:
                        gs.palm_start = now; gs.palm_active=True
                        print("[DEBUG] Palm detected — hold 1s...")
                    held = now-gs.palm_start
                    draw_palm_bar(frame, held/PALM_HOLD_SEC, fw)
                    if held >= PALM_HOLD_SEC:
                        gs.palm_active=False
                        gs.state=State.CAPTURE_FACE
                        print("[DEBUG] → CAPTURE_FACE")
                else:
                    gs.palm_active=False

            elif gs.state == State.CAPTURE_FACE:
                face_now = bool(self.fd.detect(rgb))
                cv2.putText(frame,"Face the camera clearly",(10,70),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,C_YELLOW,2,cv2.LINE_AA)
                self._try_capture(frame)

            elif gs.state == State.SHUFFLE_PUZZLE:
                draw_blocks(frame, gs.blocks, None, self.origin, now)
                cv2.putText(frame,"Get Ready!",(10,70),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,C_GREEN,2,cv2.LINE_AA)
                if self._shuf_ts and now-self._shuf_ts > 1.3:
                    gs.state=State.PLAYING; gs.start_time=now
                    print("[DEBUG] → PLAYING")

            elif gs.state == State.PLAYING:
                self._playing(frame, pinch_raw, px, py, dist)
                update_particles(gs, dt)
                draw_blocks(frame, gs.blocks, gs.selected_block, self.origin, now)
                draw_particles(frame, gs)
                draw_timer(frame, gs.elapsed, fw-200, 30)
                draw_pinch_cursor(frame, px, py, pinch_raw)
                if gs.elapsed > 3.0:
                    if palm:
                        if not gs.palm_active:
                            gs.palm_start=now; gs.palm_active=True
                        held=now-gs.palm_start
                        draw_palm_bar(frame,held/PALM_HOLD_SEC,fw)
                        if held>=PALM_HOLD_SEC: self._reset(); continue
                    else: gs.palm_active=False

            elif gs.state == State.SOLVED:
                gs.solved_anim_t += dt
                update_particles(gs, dt)
                draw_blocks(frame, gs.blocks, None, self.origin, now)
                draw_particles(frame, gs)
                draw_solved_screen(frame, gs.solved_time, gs.solved_anim_t)
                if now-(gs.start_time+gs.solved_time) > SOLVED_SHOW:
                    self._reset()
                elif palm:
                    if not gs.palm_active:
                        gs.palm_start=now; gs.palm_active=True
                    held=now-gs.palm_start
                    draw_palm_bar(frame,held/PALM_HOLD_SEC,fw)
                    if held>=PALM_HOLD_SEC: self._reset()
                else: gs.palm_active=False

            draw_hud(frame, gs.state, fw)
            if gs.state in (State.PLAYING, State.SHUFFLE_PUZZLE, State.SOLVED):
                draw_glow_rect(frame,ox-3,oy-3,PUZZLE_SIZE+6,PUZZLE_SIZE+6,C_BLUE,layers=4)

            if self.debug:
                draw_debug(frame,self._fps,hand,palm,pinch_raw,dist,face_now,gs.state)

            cv2.imshow("Face Puzzle", frame)
            k = cv2.waitKey(1)&0xFF
            if k==ord('q'): break
            elif k==ord('r'): self._reset()
            elif k==ord('d'): self.debug=not self.debug

        self._cleanup()

    def _cleanup(self):
        self.cap.release(); self.ht.close(); self.fd.close()
        cv2.destroyAllWindows(); print("[INFO] Closed.")


if __name__=="__main__":
    try:
        idx = find_cam()
        if idx==-1:
            print("[FATAL] No webcam found."); sys.exit(1)
        FacePuzzleApp(idx).run()
    except RuntimeError as e:
        print(f"[FATAL] {e}"); sys.exit(1)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"[FATAL] {e}"); traceback.print_exc()
        cv2.destroyAllWindows(); sys.exit(1)
