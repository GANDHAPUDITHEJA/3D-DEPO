# 3D-DEPO · 2D Drawing → 3D Point Cloud

> Convert any 2D mechanical drawing into an annotated 3D point cloud, knowledge graph, and Neo4j Cypher export — in a single API call.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-0.110.0-lightgrey?style=flat-square&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green?style=flat-square&logo=opencv)
![Open3D](https://img.shields.io/badge/Open3D-0.18-orange?style=flat-square)
![Neo4j](https://img.shields.io/badge/Neo4j-Aura-008CC1?style=flat-square&logo=neo4j)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## What It Does

3D-DEPO is a computer-vision pipeline that accepts a photograph or screenshot of a 2D engineering drawing and returns:

- **3D Point Cloud** — extruded solid with surface normals (`.xyz` and `.ply`)
- **Dimension Annotations** — width, height, thickness, radius, and diameter callouts
- **Knowledge Graph** — Part → Dimension / Annotation nodes with edge relationships
- **Neo4j Cypher Script** — ready-to-paste `MERGE` statements for Neo4j Browser
- **Pipeline Figure** — 6-panel matplotlib summary image (PNG)

The companion frontend (`draw_cloud_v2.html`) is a zero-dependency SPA with a Three.js 3D viewer and D3 force-directed graph — no build step required.

---

## Demo

| Input Drawing | 3D Point Cloud | Knowledge Graph |
|---|---|---|
| 2D CAD screenshot | Interactive Three.js viewer | D3 force layout |

---

## Quick Start

**1 · Clone and install**

```bash
git clone https://github.com/GANDHAPUDITHEJA/3D-DEPO.git
cd 3D-DEPO
pip install -r requirements.txt
```

> ⚠️ The `neo4j` Python driver is used but not listed in `requirements.txt`. Add it if you want database push:
> ```bash
> pip install neo4j
> ```

**2 · Start the API**

```bash
python api_v2.py
# → Serving on http://0.0.0.0:5000
```

**3 · Open the UI**

Open `draw_cloud_v2.html` directly in your browser — no server needed. Point the **API** field at `http://localhost:5000`.

---

## API Reference

### `GET /health`
Service health check.
```json
{ "status": "ok", "version": "3D-DEPO-v6.1" }
```

### `GET /ping`
Lightweight liveness probe.
```json
{ "status": "ok" }
```

### `POST /process`
Main pipeline endpoint. Accepts `multipart/form-data`.

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | File (**required**) | — | PNG / JPG / JPEG / BMP of the 2D drawing |
| `thickness` | float | `10.0` | Part extrusion depth in mm |
| `density` | float | `0.5` | Point spacing in mm (smaller = denser) |
| `layers` | int | `25` | Z-slices through the thickness |
| `known_dim` | float | `200.0` | Real-world drawing height in mm (sets scale) |
| `part_name` | string | `"Part"` | Label stored in the knowledge graph |

**Example cURL:**
```bash
curl -X POST http://localhost:5000/process \
  -F "image=@bracket.png" \
  -F "thickness=12" \
  -F "density=0.4" \
  -F "layers=30" \
  -F "known_dim=150" \
  -F "part_name=BracketA"
```

**Success response fields:**

| Field | Type | Description |
|---|---|---|
| `pipeline_png` | base64 string | 6-panel matplotlib figure |
| `xyz_b64` | base64 string | Full XYZ point cloud (plain text) |
| `ply_b64` | base64 string / null | Binary PLY with normals |
| `neo4j_cypher` | string | Ready-to-paste Cypher MERGE script |
| `graph_data` | object | `{ nodes[], edges[] }` for D3 |
| `annotations` | object[] | Dimension annotation objects |
| `title_block` | object | Engineering title block metadata |
| `total_points` | int | Point count after voxel dedup |
| `n_holes` | int | Detected interior holes |
| `bbox_mm` | float[2] | `[width_mm, height_mm]` |

---

## Pipeline Stages

```
Smart Crop → Annotation Removal → Binarize → Contour Extraction
    → Scale Calibration → Fill Interior → Extrude → Voxel Filter → Normals
```

| Stage | Function | Detail |
|---|---|---|
| **Smart Crop** | `smart_crop()` | Detects two-view landscape drawings; crops to the annotated half |
| **Annotation Removal** | `remove_annotations()` | Inpaints green HSV dimension lines using `cv2.INPAINT_TELEA` |
| **Binarize** | `binarize()` | Best-of-three competition: Otsu, CLAHE+Otsu, Adaptive |
| **Contour Extraction** | `extract_contours()` | `RETR_TREE` hierarchy; Canny edge fallback if primary fails |
| **Scale Calibration** | `calibrate_scale()` | Maps bounding-box pixels to mm via `known_dim` |
| **Fill Interior** | `fill()` | `cv2.fillPoly` mask — handles concave and multi-lobed profiles correctly |
| **Extrude** | `extrude()` | Stacks boundary + interior points across `layers` Z values |
| **Voxel Filter** | `voxel()` | Deduplicates overlapping points; grid size = `density × 0.6` |
| **Normals** | `normals()` | Open3D KDTree hybrid search; tangent-plane orientation |

---

## Project Structure

```
3D-DEPO/
├── api_v2.py              # Flask API server — full CV + 3D pipeline
├── draw_cloud_v2.html     # Frontend SPA — Three.js viewer + D3 graph
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `flask` | 0.110.0 | HTTP server |
| `opencv-python-headless` | 4.9.0.80 | Image processing, binarisation, contour detection |
| `numpy` | 1.26.4 | Array mathematics |
| `matplotlib` | 3.8.3 | Pipeline figure rendering |
| `scipy` | 1.12.0 | Contour resampling via 1D interpolation |
| `open3d` | 0.18.0 | Surface normal estimation; PLY export |
| `pillow` | 10.2.0 | Image I/O |
| `uvicorn[standard]` | 0.27.1 | ASGI runner (optional production use) |
| `python-multipart` | 0.0.6 | Multipart file upload parsing |
| `jinja2` | 3.1.3 | Flask template engine |

Frontend CDN libraries (no install needed):
- [Three.js r128](https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js)
- [D3 v7.8.5](https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js)

---

## Configuration

Global tuning constants at the top of `api_v2.py`:

```python
_D = dict(
    min_hole_frac = 0.0005,   # min hole area as fraction of image area
    outer_frac    = 0.008,    # min outer profile area fraction
    min_blob_frac = 0.00005,  # noise-removal threshold
    voxel_frac    = 0.6,      # voxel size = density × voxel_frac
)
```

---

## ⚠️ Security

**Before deploying to any non-local environment:**

- **Rotate your Neo4j Aura password.** The current code contains hardcoded credentials in `api_v2.py`. Move them to environment variables:

```python
import os
NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USER     = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
```

- **Restrict CORS.** `CORS(app)` currently allows all origins. Set `origins=["https://your-domain.com"]` in production.
- **Add file-size limits.** The `/process` endpoint has no upload cap. Use Flask's `MAX_CONTENT_LENGTH` config.

---

## Known Issues / Roadmap

- [ ] Add `neo4j` to `requirements.txt`
- [ ] Replace hardcoded Neo4j credentials with environment variables
- [ ] Add file size and MIME type validation on `/process`
- [ ] Restrict CORS origins for production
- [ ] Drawing number counter resets on server restart — consider persistent storage
- [ ] `flask-cors` is used but not listed in `requirements.txt`

---

## License

MIT — see [LICENSE](LICENSE) for details.