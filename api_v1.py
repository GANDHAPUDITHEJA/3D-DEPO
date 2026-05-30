"""
3D-DEPO API v1  —  3D Point Cloud + Knowledge Graph + Neo4j Export
Fixes:
  • fill() uses cv2.fillPoly mask — handles complex/concave shapes correctly
  • Contour gap-closing after annotation removal
  • Smart crop: two-view landscape drawings → use annotated (left) half
  • White-background drawings handled robustly
Run:  python app.py  →  http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2, numpy as np, os, io, base64, traceback, tempfile, re, json
import datetime as _dt, itertools as _it
from scipy.interpolate import interp1d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

app  = Flask(__name__)
CORS(app)

_D = dict(min_hole_frac=0.0005, outer_frac=0.008,
          min_blob_frac=0.00005, voxel_frac=0.6)

# ═══════════════════════════════════════════════════════════════
#  SMART CROP
#  • Single-view portrait   → full image
#  • Two-view landscape     → annotated half (whichever has green dims)
# ═══════════════════════════════════════════════════════════════

def smart_crop(img):
    h, w = img.shape[:2]
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gmask= cv2.inRange(hsv, np.array([35,40,40]), np.array([95,255,255]))

    lg = gmask[:, :w//2].sum()
    rg = gmask[:, w//2:].sum()

    # Both halves have similar annotation density → single view, use all
    if lg > 0 and rg > 0 and min(lg,rg) > max(lg,rg)*0.25:
        return img.copy(), "full"

    # No green at all → full image
    if lg == 0 and rg == 0:
        return img.copy(), "full"

    # Two-view landscape: use the annotated half
    if w > h * 1.3:
        if lg >= rg:
            return img[:, :w//2].copy(), "left_half"
        else:
            return img[:, w//2:].copy(), "right_half"

    # Single view portrait → full
    return img.copy(), "full"


# ═══════════════════════════════════════════════════════════════
#  ANNOTATION REMOVAL
# ═══════════════════════════════════════════════════════════════

def remove_annotations(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gm  = cv2.inRange(hsv, np.array([35,40,40]), np.array([95,255,255]))
    k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    gm  = cv2.dilate(gm, k, iterations=2)
    cleaned = cv2.inpaint(img, gm, 5, cv2.INPAINT_TELEA)
    return cleaned, gm


# ═══════════════════════════════════════════════════════════════
#  BINARIZATION
# ═══════════════════════════════════════════════════════════════

def _clean(b, mb):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=3)
    n,l,s,_ = cv2.connectedComponentsWithStats(b)
    c = np.zeros_like(b)
    for lb in range(1,n):
        if s[lb, cv2.CC_STAT_AREA] >= mb: c[l==lb]=255
    return c

def _score(b, om):
    cts, hier = cv2.findContours(b, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hier is None: return 0
    return sum(1 for c,hi in zip(cts,hier[0])
               if hi[3]==-1 and cv2.contourArea(c)>om)

def binarize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    area = gray.shape[0]*gray.shape[1]
    mb   = max(30,  int(area*_D["min_blob_frac"]))
    om   = max(200, int(area*_D["outer_frac"]))

    # --- Three candidate binarizations ---
    _, bA = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    bA = _clean(bA, mb); sA = _score(bA, om)

    cl = cv2.createCLAHE(3.0,(8,8))
    _, bB = cv2.threshold(cl.apply(gray), 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    bB = _clean(bB, mb); sB = _score(bB, om)

    blk = max(11,(min(gray.shape)//20)|1)
    bC  = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV,blk,3)
    bC = _clean(bC, mb); sC = _score(bC, om)

    best = max([(bA,sA,"Otsu"),(bB,sB,"CLAHE+Otsu"),(bC,sC,"Adaptive")],
               key=lambda x:(x[1],(x[0]>0).sum()))

    # Fallback with looser threshold
    if best[1] == 0:
        om //= 5
        best = max([(bA,_score(bA,om),"Otsu"),
                    (bB,_score(bB,om),"CLAHE+Otsu"),
                    (bC,_score(bC,om),"Adaptive")],
                   key=lambda x:(x[1],(x[0]>0).sum()))

    # Extra morphological closing to heal gaps left by annotation inpainting
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    healed = cv2.morphologyEx(best[0], cv2.MORPH_CLOSE, k2, iterations=2)

    return healed, gray, best[2]


# ═══════════════════════════════════════════════════════════════
#  CONTOUR EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_contours(binary, area, cleaned=None):
    mh  = max(50,  int(area*_D["min_hole_frac"]))
    om  = max(200, int(area*_D["outer_frac"]))
    cts, hier = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hier is None: raise RuntimeError("No contours found")

    outers, holes = [], []
    for c,hi in zip(cts,hier[0]):
        a = cv2.contourArea(c)
        if   hi[3]==-1 and a>om: outers.append((a,c))
        elif hi[3]>=0  and a>mh: holes.append((a,c))

    if not outers:
        om //= 3
        for c,hi in zip(cts,hier[0]):
            if hi[3]==-1 and cv2.contourArea(c)>om:
                outers.append((cv2.contourArea(c),c))

    # Canny-based edge fallback
    if not outers and cleaned is not None:
        gray2  = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        edges  = cv2.Canny(gray2,20,80)
        k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        ec     = cv2.morphologyEx(cv2.dilate(edges,k,iterations=2),
                                   cv2.MORPH_CLOSE, k, iterations=5)
        cts2,_ = cv2.findContours(ec,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        valid  = [(cv2.contourArea(c),c) for c in cts2 if cv2.contourArea(c)>5000]
        if valid:
            valid.sort(key=lambda x:x[0], reverse=True)
            outers = [valid[0]]

    if not outers:
        raise RuntimeError("No outer profile detected — try a clearer image or higher contrast")

    outers.sort(key=lambda x:x[0], reverse=True)
    holes.sort (key=lambda x:x[0], reverse=True)
    return outers[0][1], [c for _,c in holes]


# ═══════════════════════════════════════════════════════════════
#  SCALE + MM CONVERSION
# ═══════════════════════════════════════════════════════════════

def calibrate_scale(oc, dim):
    x0,y0,bw,bh = cv2.boundingRect(oc)
    return dim/bh, (x0,y0,bw,bh)

def to_mm(c, scale, bbox):
    x0,y0,_,bh = bbox
    pts = c.reshape(-1,2).astype(float)
    mm  = (pts - np.array([x0,y0]))*scale
    mm[:,1] = bh*scale - mm[:,1]
    return mm

def resample(pts, sp):
    if len(pts)<3: return pts
    d   = np.diff(pts,axis=0)
    arc = np.concatenate([[0],np.cumsum(np.sqrt((d**2).sum(1)))])
    tot = arc[-1]
    if tot<sp: return pts
    n   = max(4,int(tot/sp))
    t   = np.linspace(0,tot,n,endpoint=False)
    ac  = np.append(arc,arc[-1]+arc[1])
    pc  = np.vstack([pts,pts[0]])
    return np.column_stack([interp1d(ac,pc[:,0],'linear')(t),
                             interp1d(ac,pc[:,1],'linear')(t)])


# ═══════════════════════════════════════════════════════════════
#  FILL INTERIOR  ← KEY FIX: cv2.fillPoly mask instead of
#                              matplotlib Path.contains_points
#  Works correctly for concave, complex, multi-lobed outlines
# ═══════════════════════════════════════════════════════════════

def fill(outer_rs, holes_rs, sp):
    mn = outer_rs.min(0)
    mx = outer_rs.max(0)

    # Mask at exactly sp resolution (1 pixel = 1 sample point)
    res = sp
    W   = int((mx[0]-mn[0])/res) + 4
    H   = int((mx[1]-mn[1])/res) + 4

    # Guard against ridiculous sizes
    if W*H > 20_000_000:
        res = sp * 2
        W   = int((mx[0]-mn[0])/res) + 4
        H   = int((mx[1]-mn[1])/res) + 4

    mask = np.zeros((H, W), dtype=np.uint8)

    def to_px(pts):
        p = ((pts - mn) / res + 1).astype(np.int32)
        p[:,0] = np.clip(p[:,0], 0, W-1)
        p[:,1] = np.clip(p[:,1], 0, H-1)
        return p

    # Paint outer profile white, then subtract holes
    cv2.fillPoly(mask, [to_px(outer_rs)], 255)
    for h in holes_rs:
        if len(h) >= 3:
            cv2.fillPoly(mask, [to_px(h)], 0)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.empty((0,2))

    pts = np.column_stack([(xs - 1)*res + mn[0],
                            (ys - 1)*res + mn[1]])
    return pts


# ═══════════════════════════════════════════════════════════════
#  EXTRUDE + VOXEL + NORMALS
# ═══════════════════════════════════════════════════════════════

def extrude(bnd, interior, holes, thick, nl):
    zs  = np.linspace(0, thick, nl)
    out = []
    for z in zs: out.append(np.column_stack([bnd, np.full(len(bnd),z)]))
    for h in holes:
        for z in zs: out.append(np.column_stack([h,  np.full(len(h),z)]))
    for z in [0., thick]:
        if len(interior): out.append(np.column_stack([interior, np.full(len(interior),z)]))
    return np.vstack(out)

def voxel(xyz, vs):
    vi   = np.floor(xyz/vs).astype(int)
    seen = {}
    for i,v in enumerate(map(tuple,vi)):
        if v not in seen: seen[v]=i
    return xyz[list(seen.values())]

def normals(xyz):
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.0,max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)
        return pcd
    except Exception: return None


# ═══════════════════════════════════════════════════════════════
#  DIMENSION ANNOTATIONS
# ═══════════════════════════════════════════════════════════════

def fit_circle(pts):
    cx,cy = pts.mean(0)
    r     = np.sqrt(((pts-np.array([cx,cy]))**2).sum(1)).mean()
    return float(cx), float(cy), float(r)

def build_annotations(outer_rs, holes_rs, bbox_mm, thickness):
    W, H  = bbox_mm
    mz    = thickness / 2.0
    top_z = thickness
    anns  = []

    anns.append(dict(label=f"{H:.1f}", type="linear_v", color="#00BFFF",
                     ax=W+W*.06, ay=0,      az=mz,
                     tx=W+W*.06, ty=H,      tz=mz))
    anns.append(dict(label=f"{W:.1f}", type="linear_h", color="#00BFFF",
                     ax=0,       ay=-H*.06, az=mz,
                     tx=W,       ty=-H*.06, tz=mz))
    anns.append(dict(label=f"T={thickness:.0f}", type="linear_z", color="#00BFFF",
                     ax=W+W*.10, ay=H+H*.10, az=0,
                     tx=W+W*.10, ty=H+H*.10, tz=top_z))

    if len(outer_rs) >= 6:
        ocx,ocy,orr = fit_circle(outer_rs)
        ang = np.radians(135)
        anns.append(dict(label=f"R{orr:.1f}", type="radius", color="#FFD700",
                         ax=ocx, ay=ocy, az=mz,
                         tx=ocx+np.cos(ang)*orr, ty=ocy+np.sin(ang)*orr, tz=mz))

    hole_cols = ["#ADFF2F","#FF6B6B","#FF8C69","#FFA07A",
                 "#FFB347","#FFC0CB","#FF69B4","#DA70D6"]
    placed = []
    for i,h in enumerate(holes_rs):
        if len(h)<4: continue
        hcx,hcy,hr = fit_circle(h)
        if any(np.hypot(hcx-px,hcy-py)<hr*1.5 for px,py in placed): continue
        placed.append((hcx,hcy))
        col = hole_cols[i % len(hole_cols)]
        ang = np.radians(30+i*50)
        lbl = f"Ø{hr*2:.1f}" if i==0 else f"R{hr:.1f}"
        anns.append(dict(label=lbl, type="radius", color=col,
                         ax=hcx, ay=hcy, az=mz,
                         tx=hcx+np.cos(ang)*hr*2.2,
                         ty=hcy+np.sin(ang)*hr*2.2, tz=mz))
    return anns


# ═══════════════════════════════════════════════════════════════
#  KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════

def build_knowledge_graph(anns, bbox_mm, thickness,
                          part_name="Part", drawing_no="DC-0001"):
    W, H     = bbox_mm
    nodes, edges = [], []
    part_id  = "part_0"

    nodes.append({
        "id": part_id, "label": part_name, "type": "PART",
        "props": {"drawing_no": drawing_no,
                  "width_mm":   W,  "height_mm":    H,
                  "thickness_mm": thickness, "area_mm2": round(W*H,2)}
    })

    for lbl, val, dim_type in [
        (f"{W:.1f} mm",          W,         "WIDTH"),
        (f"{H:.1f} mm",          H,         "HEIGHT"),
        (f"T={thickness:.0f} mm",thickness, "THICKNESS"),
    ]:
        nid = f"dim_{dim_type.lower()}"
        nodes.append({"id":nid,"label":lbl,"type":"DIMENSION",
                      "props":{"dim_type":dim_type,"value":val,"unit":"mm"}})
        edges.append({"from":part_id,"to":nid,"rel":"HAS_DIMENSION","color":"#00BFFF"})

    seen_lbl = {}
    for i,ann in enumerate(anns):
        if ann["type"] in ("linear_v","linear_h","linear_z"): continue
        lbl   = ann["label"]
        cnt   = seen_lbl.get(lbl,0); seen_lbl[lbl]=cnt+1
        uid   = f"ann_{i}"
        val   = float(re.sub(r'[^0-9.]','',lbl) or 0)
        atype = "DIAMETER" if lbl.startswith("Ø") else "RADIUS"
        nodes.append({
            "id": uid,
            "label": lbl if cnt==0 else f"{lbl} ({cnt+1})",
            "type": "ANNOTATION",
            "props": {"annotation_type":atype, "value_mm":val,
                      "cx_mm":round(ann["ax"],2), "cy_mm":round(ann["ay"],2),
                      "color":ann["color"]}
        })
        edges.append({"from":part_id,"to":uid,"rel":"HAS_ANNOTATION","color":ann["color"]})

    lines = [
        "// 3D-DEPO v1 — Neo4j Cypher Export",
        "// Paste into Neo4j Browser or use neo4j-driver",
        "",
    ]
    p = nodes[0]["props"]
    lines += [
        f'MERGE (part:Part {{drawing_no: "{drawing_no}"}}) SET',
        f'  part.name="{part_name}", part.width_mm={p["width_mm"]},',
        f'  part.height_mm={p["height_mm"]}, part.thickness_mm={p["thickness_mm"]},',
        f'  part.area_mm2={p["area_mm2"]};', "",
    ]
    for n in nodes[1:]:
        ps  = ", ".join(f'{k}: {json.dumps(v)}' for k,v in n["props"].items())
        rel = next((e["rel"] for e in edges if e["to"]==n["id"]), "HAS_NODE")
        nt  = n["type"].capitalize(); var=f'n_{n["id"]}'
        lines += [
            f'MERGE ({var}:{nt} {{id: "{n["id"]}"}}) SET {var} += {{{ps}}};',
            f'MATCH (part:Part {{drawing_no: "{drawing_no}"}}), '
            f'({var}:{nt} {{id: "{n["id"]}"}}) MERGE (part)-[:{rel}]->({var});', "",
        ]
    return {"nodes": nodes, "edges": edges, "cypher": "\n".join(lines)}


# ═══════════════════════════════════════════════════════════════
#  NEO4J PUSH  (disabled — fill credentials to enable)
# ═══════════════════════════════════════════════════════════════

import re
from neo4j import GraphDatabase

def neo4j_push(graph_data):
    NEO4J_URI      = "neo4j+s://aae46cce.databases.neo4j.io"
    NEO4J_USER     = "neo4j"          # ← Aura default username, not the instance ID
    NEO4J_PASSWORD = "f391518YvUrHECPyiP8BXzcWFBO6ja3vW_W6B-wutKY"

    cypher_text = re.sub(r'cy-keyword">', '', graph_data["cypher"])
    cypher_text = re.sub(r'cy-label">',   '', cypher_text)

    stmts = []
    for stmt in cypher_text.split(";"):
        # Strip comment lines individually
        stmt = "\n".join(
            line for line in stmt.splitlines()
            if not line.strip().startswith("//")
        ).strip()
        if stmt:
            stmts.append(stmt)

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        connection_timeout=30,
        max_transaction_retry_time=15,
    )

    def _run_all(tx):
        for stmt in stmts:
            tx.run(stmt)

    try:
        with driver.session(database="neo4j") as session:
            session.execute_write(_run_all)
            print(f"✓ Pushed {len(stmts)} statements to Neo4j")
    except Exception as e:
        print("Neo4j push failed:", e)
    finally:
        driver.close()


# ═══════════════════════════════════════════════════════════════
#  3D RENDERING HELPERS
# ═══════════════════════════════════════════════════════════════

def _dim_line3d(ax,x1,y1,z1,x2,y2,z2,lbl,col,ox=0,oy=0,oz=0,fs=7):
    ax.plot([x1,x2],[y1,y2],[z1,z2], color=col, lw=0.9, alpha=0.9, linestyle='--')
    mx,my,mz = (x1+x2)/2+ox, (y1+y2)/2+oy, (z1+z2)/2+oz
    ln = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)+1e-9
    ux,uy,uz = (x2-x1)/ln, (y2-y1)/ln, (z2-z1)/ln
    hl = min(ln*.12, 4.)
    for sx,sy,sz,du in [(x1,y1,z1,1),(x2,y2,z2,-1)]:
        ax.quiver(sx,sy,sz, du*ux,du*uy,du*uz, length=hl, color=col,
                  linewidth=0., arrow_length_ratio=1., alpha=.9)
    ax.text(mx,my,mz, lbl, color=col, fontsize=fs, fontweight='bold',
            ha='center', va='center', zorder=10,
            bbox=dict(boxstyle='round,pad=.22',fc='#0a0a18',ec=col,alpha=.88,lw=.75))

def _ann_arrow3d(ax, ax0,ay0,az0, tx,ty,tz, lbl,col, fs=7):
    ax.plot([ax0,tx],[ay0,ty],[az0,tz], color=col, lw=0.9, alpha=0.9)
    ln = np.sqrt((tx-ax0)**2+(ty-ay0)**2+(tz-az0)**2)+1e-9
    ux,uy,uz = (tx-ax0)/ln,(ty-ay0)/ln,(tz-az0)/ln
    hl = min(ln*.22, 4.)
    ax.quiver(tx-ux*hl*.5,ty-uy*hl*.5,tz-uz*hl*.5, ux,uy,uz,
              length=hl, color=col, linewidth=0., arrow_length_ratio=1., alpha=.95)
    lx = tx+ux*ln*.18; ly = ty+uy*ln*.18; lz = tz+uz*ln*.18
    ax.text(lx,ly,lz, lbl, color=col, fontsize=fs, fontweight='bold',
            ha='center', va='center', zorder=10,
            bbox=dict(boxstyle='round,pad=.22',fc='#0a0a18',ec=col,alpha=.88,lw=.75))

def render_3d(ax, xyz, anns, thick, bbox_mm, bg):
    W,H = bbox_mm
    ax.set_facecolor(bg)
    idx = np.random.choice(len(xyz), min(15000,len(xyz)), replace=False)
    s   = xyz[idx]
    zn  = (s[:,2]-s[:,2].min())/(s[:,2].max()-s[:,2].min()+1e-9)
    ax.scatter(s[:,0],s[:,1],s[:,2], s=0.7, c=plt.cm.cool(zn),
               alpha=.72, depthshade=True)

    corners = np.array([[0,0,0],[W,0,0],[W,H,0],[0,H,0],
                         [0,0,thick],[W,0,thick],[W,H,thick],[0,H,thick]])
    for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)]:
        ax.plot(*zip(corners[a],corners[b]), color='#2a3a5a', lw=0.5, alpha=0.5)

    for ann in anns:
        ax0,ay0,az0 = ann['ax'],ann['ay'],ann['az']
        tx,ty,tz    = ann['tx'],ann['ty'],ann['tz']
        col,lbl,at  = ann['color'],ann['label'],ann['type']
        if   at=='radius':   _ann_arrow3d(ax,ax0,ay0,az0,tx,ty,tz,lbl,col)
        elif at=='linear_v': _dim_line3d(ax,ax0,0,az0,ax0,H,az0,lbl,col,ox=W*.04)
        elif at=='linear_h': _dim_line3d(ax,0,ay0,az0,W,ay0,az0,lbl,col,oy=-H*.04)
        elif at=='linear_z': _dim_line3d(ax,ax0,ay0,0,ax0,ay0,thick,lbl,col,ox=W*.04)

    ax.set_xlabel('X',color='#777',fontsize=6); ax.set_ylabel('Y',color='#777',fontsize=6)
    ax.set_zlabel('Z',color='#777',fontsize=6); ax.tick_params(colors='#444',labelsize=5)
    for p in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
        p.fill=False; p.set_edgecolor('#1a1a2e')
    ax.grid(True, alpha=.10, color='#333')
    ax.view_init(elev=28, azim=-55)

def draw_2d_anns(ax, anns, bbox_mm, bg="#0d0d1a"):
    W,H = bbox_mm
    for ann in anns:
        ax0,ay0 = ann['ax'],ann['ay']; tx,ty = ann['tx'],ann['ty']
        col,lbl,at = ann['color'],ann['label'],ann['type']
        if at=='radius':
            ax.annotate('', xy=(tx,ty), xytext=(ax0,ay0),
                        arrowprops=dict(arrowstyle='->',color=col,lw=1.0))
            ux,uy = tx-ax0, ty-ay0
            ln    = np.sqrt(ux**2+uy**2)+1e-9
            ax.text(tx+ux/ln*ln*.18, ty+uy/ln*ln*.18, lbl,
                    color=col, fontsize=6.5, fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=.22',fc=bg,ec=col,alpha=.88,lw=.75))
        elif at=='linear_v':
            ax.annotate('', xy=(ax0,H), xytext=(ax0,0),
                        arrowprops=dict(arrowstyle='<->',color=col,lw=.9))
            ax.text(ax0+W*.04, H/2, lbl, color=col, fontsize=6.5, fontweight='bold',
                    ha='left', va='center',
                    bbox=dict(boxstyle='round,pad=.22',fc=bg,ec=col,alpha=.88,lw=.75))
        elif at=='linear_h':
            ax.annotate('', xy=(W,ay0), xytext=(0,ay0),
                        arrowprops=dict(arrowstyle='<->',color=col,lw=.9))
            ax.text(W/2, ay0-H*.04, lbl, color=col, fontsize=6.5, fontweight='bold',
                    ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=.22',fc=bg,ec=col,alpha=.88,lw=.75))


# ═══════════════════════════════════════════════════════════════
#  TITLE BLOCK
# ═══════════════════════════════════════════════════════════════

_DRAW_CTR = _it.count(1)

def build_title_block(bbox_mm, thickness, nholes, bname, view, density, layers):
    W,H = bbox_mm; now = _dt.datetime.now()
    drw = f"DC-{now.strftime('%Y%m%d')}-{next(_DRAW_CTR):04d}"
    return {
        "drawing_no":drw, "part_name":"PART", "scale":"1:1",
        "material":"UNSPECIFIED", "sheet_no":"1", "sheet_total":"1",
        "unit":"MM", "standard":"AAI", "date":now.strftime("%Y-%m-%d"),
        "drawn_by":"3D-DEPO AI", "binarize":bname, "view_crop":view.upper(),
        "n_holes":nholes, "density_mm":density, "layers":layers,
        "dims_summary":f"W {W:.1f} × H {H:.1f} × T {thickness:.1f} mm",
        "width_mm":W, "height_mm":H, "thickness_mm":thickness,
    }

def draw_title_block_axes(fig, tb):
    bg="#0d0d1a"; border="#1c3a5a"
    txt_hi="#00d4ff"; txt_lo="#5a8aaa"; txt_val="#e8f0f8"
    ax = fig.add_axes([0.01,0.005,0.98,0.10])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off"); ax.set_facecolor(bg)

    def cell(x,y,w,h,lbl,val,lfs=5.2,vfs=7.,vc=None):
        vc = vc or txt_val
        ax.add_patch(plt.Rectangle((x,y),w,h, lw=0.7,edgecolor=border,facecolor=bg,zorder=1))
        if lbl: ax.text(x+.006,y+h-.06,lbl, color=txt_lo,fontsize=lfs,fontweight='bold',
                        va='top',ha='left',fontfamily='monospace',zorder=2)
        if val: ax.text(x+w/2,y+h*.38,str(val), color=vc,fontsize=vfs,fontweight='bold',
                        va='center',ha='center',fontfamily='monospace',zorder=2)

    cw=[.13,.12,.20,.12,.12,.31]; cx=[sum(cw[:i]) for i in range(len(cw))]; rh=1/3
    cell(cx[0],rh*2,cw[0],rh,"DRAWING NO.",tb["drawing_no"],vfs=6.)
    cell(cx[1],rh*2,cw[1]+cw[2],rh,"",tb["part_name"],vfs=11,vc=txt_hi)
    cell(cx[3],rh*2,cw[3],rh,"DATE",tb["date"],vfs=6.)
    cell(cx[4],rh*2,cw[4],rh,"DRAWN BY",tb["drawn_by"],vfs=5.5)
    cell(cx[5],rh*2,cw[5],rh,"DIMS",tb["dims_summary"],vfs=5.5,vc="#ADFF2F")
    cell(cx[0],rh,cw[0],rh,"SCALE",tb["scale"],vfs=8.)
    cell(cx[1],rh,cw[1],rh,"HOLES",str(tb["n_holes"]),vfs=8.,vc="#FF8C69")
    cell(cx[2],rh,cw[2],rh,"MATERIAL",tb["material"],vfs=7.)
    cell(cx[3],rh,cw[3],rh,"BINARIZE",tb["binarize"],vfs=5.8)
    cell(cx[4],rh,cw[4],rh,"VIEW CROP",tb["view_crop"],vfs=7.)
    cell(cx[5],rh,cw[5],rh,"DENSITY/LAYERS",
         f"{tb['density_mm']} mm / {tb['layers']} layers",vfs=6.5)
    cell(cx[0],0,cw[0],rh,"SHEET NO.",tb["sheet_no"],vfs=8.)
    cell(cx[1],0,cw[1],rh,"SHEET",f"{tb['sheet_no']}/{tb['sheet_total']}",vfs=7.)
    cell(cx[2],0,cw[2],rh,"",
         f"W{tb['width_mm']:.1f} H{tb['height_mm']:.1f} T{tb['thickness_mm']:.1f} mm",
         vfs=7.,vc=txt_hi)
    cell(cx[3],0,cw[3],rh,"UNIT",tb["unit"],vfs=8.)
    cell(cx[4],0,cw[4],rh,"STANDARD",tb["standard"],vfs=8.)
    cell(cx[5],0,cw[5],rh,"",f"SHEET {tb['sheet_no']}/{tb['sheet_total']}",vfs=6.5)
    ax.add_patch(plt.Rectangle((0,0),1,1, lw=1.4,edgecolor=txt_hi,facecolor="none",zorder=3))


# ═══════════════════════════════════════════════════════════════
#  PIPELINE IMAGE (6-panel + title block)
# ═══════════════════════════════════════════════════════════════

def pipeline_image(region, cleaned, binary, bname, outer_c, hole_cs,
                   outer_rs, holes_rs, interior, xyz, nholes,
                   anns, thickness, bbox_mm, tb):
    bg = "#0d0d1a"
    fig = plt.figure(figsize=(22,12), facecolor=bg)
    fig.subplots_adjust(left=.02,right=.98,top=.95,bottom=.14,hspace=.35,wspace=.22)

    ax1 = fig.add_subplot(2,3,1)
    ax1.imshow(cv2.cvtColor(region,cv2.COLOR_BGR2RGB)); ax1.axis("off")
    ax1.set_title("① Input View",color='w',fontsize=9)

    ax2 = fig.add_subplot(2,3,2)
    ax2.imshow(cv2.cvtColor(cleaned,cv2.COLOR_BGR2RGB)); ax2.axis("off")
    ax2.set_title("② Annotations Removed",color='w',fontsize=9)

    ax3 = fig.add_subplot(2,3,3)
    ax3.imshow(binary,cmap="gray"); ax3.axis("off")
    ax3.set_title(f"③ Binarized [{bname}]",color='w',fontsize=9)

    canvas = cv2.cvtColor(cleaned,cv2.COLOR_BGR2RGB).copy()
    cv2.drawContours(canvas,[outer_c],-1,(0,220,255),3)
    for hc in hole_cs[:20]: cv2.drawContours(canvas,[hc],-1,(255,70,70),2)
    ax4 = fig.add_subplot(2,3,4)
    ax4.imshow(canvas); ax4.axis("off")
    ax4.set_title(f"④ Contours (outer=cyan · {nholes} holes=red)",color='w',fontsize=9)

    ax5 = fig.add_subplot(2,3,5); ax5.set_facecolor(bg)
    ax5.set_title(f"⑤ 2D Profile + Dims ({len(outer_rs):,} bnd · {len(interior):,} fill)",
                  color='w',fontsize=9,pad=5)
    c5 = outer_rs[:,1]/(outer_rs[:,1].max()+1e-9)
    ax5.scatter(*outer_rs.T, s=0.8, c=c5, cmap='plasma', alpha=0.9)
    for h in holes_rs: ax5.scatter(*h.T, s=0.8, c='tomato', alpha=0.6)
    if len(interior): ax5.scatter(*interior.T, s=0.3, c='#10b981', alpha=0.3)
    draw_2d_anns(ax5, anns, bbox_mm, bg)
    ax5.set_aspect('equal')
    ax5.set_xlabel('X (mm)',color='#666',fontsize=7)
    ax5.set_ylabel('Y (mm)',color='#666',fontsize=7)
    ax5.tick_params(colors='#555',labelsize=6)
    ax5.grid(True,alpha=.12,color='#333')
    for sp in ax5.spines.values(): sp.set_color('#222')

    ax6 = fig.add_subplot(2,3,6, projection='3d')
    ax6.set_title(f"⑥ 3D Cloud + Dims ({len(xyz):,} pts)",color='w',fontsize=9,pad=5)
    render_3d(ax6, xyz, anns, thickness, bbox_mm, bg)

    draw_title_block_axes(fig, tb)
    fig.suptitle("3D-DEPO v1  ·  2D Drawing → 3D Cloud + Knowledge Graph",
                 color='white', fontsize=11, fontweight='bold', y=.985)

    buf = io.BytesIO()
    fig.savefig(buf, dpi=140, bbox_inches='tight',
                facecolor=bg, edgecolor='none', format='png')
    plt.close(); buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","version":"3D-DEPO-v1"})

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status":"ok"})

@app.route("/process", methods=["POST"])
def process():
    try:
        if "image" not in request.files:
            return jsonify({"error":"No image provided"}), 400
        buf = np.frombuffer(request.files["image"].read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error":"Cannot decode image"}), 400

        thickness = float(request.form.get("thickness", 10.0))
        density   = float(request.form.get("density",    0.5))
        layers    = int  (request.form.get("layers",      25))
        known_dim = float(request.form.get("known_dim", 200.0))
        part_name = request.form.get("part_name", "Part")

        region,   view        = smart_crop(img)
        area                  = region.shape[0]*region.shape[1]
        cleaned,  _           = remove_annotations(region)
        binary,   gray, bname = binarize(cleaned)
        outer_c,  hole_cs     = extract_contours(binary, area, cleaned)
        scale,    bbox        = calibrate_scale(outer_c, known_dim)
        outer_mm              = to_mm(outer_c, scale, bbox)
        holes_mm              = [to_mm(hc, scale, bbox) for hc in hole_cs]
        outer_rs              = resample(outer_mm, density)
        holes_rs              = [resample(h, density) for h in holes_mm]
        boundary              = np.vstack([outer_rs]+holes_rs)
        interior              = fill(outer_rs, holes_rs, density)   # ← fixed
        xyz                   = extrude(boundary, interior, holes_rs, thickness, layers)
        xyz                   = voxel(xyz, density*_D["voxel_frac"])
        pcd                   = normals(xyz)

        bbox_mm    = [round(float(v),1) for v in [bbox[2]*scale, bbox[3]*scale]]
        anns       = build_annotations(outer_rs, holes_rs, bbox_mm, thickness)
        tb         = build_title_block(bbox_mm, thickness, len(hole_cs),
                                       bname, view, density, layers)
        graph_data = build_knowledge_graph(anns, bbox_mm, thickness,
                                           part_name, tb["drawing_no"])
        neo4j_push(graph_data)   
        # ← uncomment + fill credentials to enable

        png = pipeline_image(region, cleaned, binary, bname,
                             outer_c, hole_cs, outer_rs, holes_rs,
                             interior, xyz, len(hole_cs),
                             anns, thickness, bbox_mm, tb)

        xyz_full    = "X Y Z\n"+"\n".join(
            f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" for p in xyz)
        xyz_preview = "X Y Z\n"+"\n".join(
            f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}" for p in xyz[:500])

        ply_b64 = None
        if pcd is not None:
            try:
                import open3d as o3d
                with tempfile.NamedTemporaryFile(suffix=".ply",delete=False) as tf:
                    o3d.io.write_point_cloud(tf.name, pcd)
                    ply_b64 = base64.b64encode(open(tf.name,"rb").read()).decode()
                    os.unlink(tf.name)
            except Exception: pass

        ann_json = [
            {"label":a["label"],"type":a["type"],"color":a["color"],
             "ax":float(a["ax"]),"ay":float(a["ay"]),"az":float(a["az"]),
             "tx":float(a["tx"]),"ty":float(a["ty"]),"tz":float(a["tz"])}
            for a in anns
        ]

        return jsonify({
            "success":       True,
            "view":          view,
            "binarize_name": bname,
            "total_points":  len(xyz),
            "n_holes":       len(hole_cs),
            "bbox_mm":       bbox_mm,
            "thickness":     thickness,
            "pipeline_png":  base64.b64encode(png).decode(),
            "neo4j_cypher":  graph_data["cypher"],
            "graph_data":    {"nodes":graph_data["nodes"],"edges":graph_data["edges"]},
            "xyz_preview":   xyz_preview,
            "xyz_b64":       base64.b64encode(xyz_full.encode()).decode(),
            "ply_b64":       ply_b64,
            "annotations":   ann_json,
            "title_block":   tb,
        })

    except Exception as e:
        return jsonify({"error":str(e),"trace":traceback.format_exc()}), 500


if __name__ == "__main__":
    print("3D-DEPO API v1  →  http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)