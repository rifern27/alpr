from rapidocr_onnxruntime import RapidOCR
import cv2
import numpy as np
import os, uuid, hashlib, re, threading

rapid_reader = RapidOCR(use_cls=False, det_limit_side_len=960, det_limit_type='max')

def _warmup():
    try:
        rapid_reader(np.zeros((32, 256, 3), dtype=np.uint8))
    except Exception:
        pass

threading.Thread(target=_warmup, daemon=True).start()


_MAX_OCR_SIDE = 1280

_PLATE_RE  = re.compile(r'([A-Z]{1,2})\s*(\d{1,4})(?:\s*([A-Z]{1,3}))?')
_NUM_FIX   = str.maketrans('OILSBZG', '0115826')
_BLOCKLIST = {'POS', 'KELUAR', 'MASUK', 'CAMERA', 'CAM', 'GATE', 'TIKET', 'PARKIR', 'PLAT', 'PK1CAM', 'IPC'}
_DIGIT_AS_PREFIX = {'8': 'B', '0': 'D', '6': 'G', '5': 'S', '1': 'I', '4': 'A'}
_INVALID_ALPHA_PREFIX = {'I': 'A', 'O': 'D', 'J': 'B', 'C': 'G', 'Q': 'D', 'U': 'D'}

_VALID_PREFIXES = {
    'A','B','C','D','E','F','G','H','K','L','M','N','P','R','S','T','W','Z',
    'AA','AB','AD','AE','AG','BA','BB','BD','BE','BG','BH','BK','BL','BM','BN',
    'BP','DA','DB','DD','DE','DG','DH','DK','DL','DM','DN','DP','DR','DT',
    'EA','EB','ED','EE','KB','KD','KE','KF','KG','KH','KT','KU','PA','PB','PD','PE'
}

def _try_prefix_fix(text: str) -> str:
    if not text:
        return text
    if text[0] in _DIGIT_AS_PREFIX:
        return _DIGIT_AS_PREFIX[text[0]] + text[1:]
    if text[0] in _INVALID_ALPHA_PREFIX:
        return _INVALID_ALPHA_PREFIX[text[0]] + text[1:]
    return text

def compute_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()

def compute_phash(img_bgr: np.ndarray, size: int = 32, small: int = 8) -> str:
    if img_bgr is None or img_bgr.size == 0:
        return ""
    gray = cv2.cvtColor(
        cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)
    block = cv2.dct(gray)[:small, :small].flatten()
    mean  = (block.sum() - block[0]) / (len(block) - 1)
    return ''.join('1' if v > mean else '0' for v in block)

def _fmt(p, n, s) -> str:
    suffix = f" {s.upper()}" if s else ""
    return f"{p.upper()} {n.translate(_NUM_FIX)}{suffix}"

def _find_plate_text(results, img_h: int) -> tuple:
    if not results:
        return "", 0.0, None

    items = []
    for (bbox, text, conf) in results:
        cx    = sum(p[0] for p in bbox) / 4
        cy    = sum(p[1] for p in bbox) / 4
        h_box = max(p[1] for p in bbox) - min(p[1] for p in bbox)
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())

        if not clean: continue
        if clean in _BLOCKLIST: continue
        if re.search(r'\d{8,}', clean): continue
        if len(clean) > 10: continue
        if "PLAT" in clean or "PK1CAM" in clean: continue

        if not re.search(r'\d', clean) and len(clean) > 3: continue

        items.append((cx, cy, clean, float(conf), bbox, h_box))

    if not items:
        return "", 0.0, None

    heights  = sorted(item[5] for item in items)
    median_h = heights[len(heights) // 2]

    items = [item for item in items if item[5] >= median_h * 0.6]
    if not items:
        return "", 0.0, None

    band = max(median_h * 0.8, img_h * 0.02)

    def _search(text: str):
        m_noise = re.match(r'^([A-Z]{2})(\d+.*)', text)
        if m_noise:
            pref, rest = m_noise.groups()
            if pref not in _VALID_PREFIXES and pref[0] in _VALID_PREFIXES:
                text = pref[0] + rest

        m = _PLATE_RE.fullmatch(text) or _PLATE_RE.search(text)
        if m:
            return m
        fixed = _try_prefix_fix(text)
        if fixed != text:
            return _PLATE_RE.fullmatch(fixed) or _PLATE_RE.search(fixed)
        return None

    for (cx, cy, clean, conf, bbox, _) in items:
        m = _search(clean)
        if m:
            return _fmt(m.group(1), m.group(2), m.group(3)), conf, [bbox]

    sorted_items = sorted(items, key=lambda x: (x[1], x[0]))
    n            = len(sorted_items)
    best_text, best_conf, best_bboxes = "", 0.0, None

    for i in range(n):
        for j in range(i + 1, min(i + 10, n)):
            if abs(sorted_items[i][1] - sorted_items[j][1]) > band:
                break
            group    = sorted(sorted_items[i:j + 1], key=lambda x: x[0])
            combined = ''.join(g[2] for g in group)
            m = _search(combined)
            if m:
                avg_conf = sum(g[3] for g in group) / len(group)
                text     = _fmt(m.group(1), m.group(2), m.group(3))
                if avg_conf > best_conf:
                    best_text, best_conf, best_bboxes = text, avg_conf, [g[4] for g in group]

    if not best_text:
        for anchor in sorted_items:
            band_group = sorted(
                [x for x in sorted_items if abs(x[1] - anchor[1]) <= band],
                key=lambda x: x[0]
            )
            if len(band_group) < 3:
                continue
            combined = ''.join(g[2] for g in band_group)
            m = _search(combined)
            if m:
                avg_conf = sum(g[3] for g in band_group) / len(band_group)
                text     = _fmt(m.group(1), m.group(2), m.group(3))
                if avg_conf > best_conf:
                    best_text, best_conf, best_bboxes = text, avg_conf, [g[4] for g in band_group]

    if best_text:
        return best_text, best_conf, best_bboxes

    for (cx, cy, clean, conf, bbox, _) in items:
        ds = re.search(r'(\d{2,4})([A-Z]{1,3})', clean)
        if not ds:
            continue
        digits, suffix = ds.group(1), ds.group(2)
        for (px, py, pclean, pconf, pbbox, _) in items:
            if (len(pclean) == 1 and pclean.isalpha()
                    and abs(py - cy) <= band
                    and px < cx - median_h * 0.3):
                plate    = _fmt(pclean, digits, suffix)
                avg_conf = (conf + pconf) / 2
                if avg_conf > best_conf:
                    best_text, best_conf, best_bboxes = plate, avg_conf, [pbbox, bbox]

    if best_text:
        return best_text, best_conf, best_bboxes

    combined  = ''.join(g[2] for g in sorted_items)
    clean_all = re.sub(r'[^A-Z0-9]', '', combined)

    if 2 <= len(clean_all) <= 10:
        m = _search(clean_all)
        if m:
            return _fmt(m.group(1), m.group(2), m.group(3)), sorted_items[0][3] * 0.4, [g[4] for g in sorted_items]

        if re.search(r'\d', clean_all):
            return clean_all, sorted_items[0][3] * 0.5, [g[4] for g in sorted_items]
    elif len(clean_all) > 10:
        m = _search(clean_all)
        if m:
            return _fmt(m.group(1), m.group(2), m.group(3)), sorted_items[0][3] * 0.3, [g[4] for g in sorted_items]

    return "", 0.0, None

def _bbox_key(bbox) -> tuple:
    return tuple(tuple(float(x) for x in pt) for pt in bbox)

def _split_box_per_char(bbox, text: str, img: np.ndarray | None = None) -> list:
    n = len(text)
    if n <= 1:
        return [bbox]

    if img is not None:
        segs = _segment_chars_contour(img, bbox, n)
        if segs:
            return segs

    tl, tr, br, bl = bbox
    result = []
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        new_tl = [tl[0] + t0 * (tr[0] - tl[0]), tl[1] + t0 * (tr[1] - tl[1])]
        new_tr = [tl[0] + t1 * (tr[0] - tl[0]), tl[1] + t1 * (tr[1] - tl[1])]
        new_bl = [bl[0] + t0 * (br[0] - bl[0]), bl[1] + t0 * (br[1] - bl[1])]
        new_br = [bl[0] + t1 * (br[0] - bl[0]), bl[1] + t1 * (br[1] - bl[1])]
        result.append([new_tl, new_tr, new_br, new_bl])
    return result

def _crop_plate_roi(img: np.ndarray, plate_bboxes) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if not plate_bboxes:
        return img, (0, 0, img.shape[1], img.shape[0])
    all_pts = [pt for bbox in plate_bboxes for pt in bbox]
    x1 = max(0, int(min(p[0] for p in all_pts)) - 8)
    y1 = max(0, int(min(p[1] for p in all_pts)) - 8)
    x2 = min(img.shape[1], int(max(p[0] for p in all_pts)) + 8)
    y2 = min(img.shape[0], int(max(p[1] for p in all_pts)) + 8)
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)

def _preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 2)
    return cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)

def _red_ratio(img_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_lo = cv2.inRange(hsv, np.array([0, 60, 35]),   np.array([12, 255, 255]))
    mask_hi = cv2.inRange(hsv, np.array([170, 60, 35]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask_lo, mask_hi)
    return cv2.countNonZero(mask) / float(mask.size)

def _preprocess_red_plate_text(img: np.ndarray) -> np.ndarray:
    b_ch, g_ch, r_ch = cv2.split(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s_ch, v_ch = cv2.split(hsv)

    rgb_min = cv2.min(cv2.min(b_ch, g_ch), r_ch)
    white_from_hsv = cv2.subtract(v_ch, s_ch)
    white_emphasis = cv2.max(rgb_min, white_from_hsv)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    white_emphasis = clahe.apply(white_emphasis)
    white_emphasis = cv2.GaussianBlur(white_emphasis, (3, 3), 0)
    white_bin = cv2.adaptiveThreshold(
        white_emphasis,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        2,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_bin = cv2.morphologyEx(white_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.cvtColor(white_bin, cv2.COLOR_GRAY2BGR)

def _segment_chars_contour(img_bgr: np.ndarray, bbox: list, n_chars: int) -> list:
    pts   = np.array([[int(p[0]), int(p[1])] for p in bbox])
    x1    = max(0, int(pts[:, 0].min()) - 2)
    y1    = max(0, int(pts[:, 1].min()) - 2)
    x2    = min(img_bgr.shape[1], int(pts[:, 0].max()) + 2)
    y2    = min(img_bgr.shape[0], int(pts[:, 1].max()) + 2)

    if x2 - x1 < 6 or y2 - y1 < 6:
        return []

    region = img_bgr[y1:y2, x1:x2].copy()
    h, w   = region.shape[:2]

    scale = max(1.0, 60.0 / h)
    if scale > 1.0:
        region = cv2.resize(region, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_CUBIC)
    h_s = region.shape[0]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    for use_inv in (False, True):
        g = cv2.bitwise_not(gray) if use_inv else gray
        _, thresh = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(thresh)

        boxes = []
        for lbl in range(1, n_lbl):
            cx   = stats[lbl, cv2.CC_STAT_LEFT]
            cy   = stats[lbl, cv2.CC_STAT_TOP]
            cw   = stats[lbl, cv2.CC_STAT_WIDTH]
            ch   = stats[lbl, cv2.CC_STAT_HEIGHT]
            area = stats[lbl, cv2.CC_STAT_AREA]
            if ch < h_s * 0.25 or ch > h_s * 1.15 or cw < 3 or area < 30:
                continue
            boxes.append((cx, cy, cw, ch))

        boxes.sort(key=lambda b: b[0])

        if len(boxes) == n_chars:
            result = []
            for (cx, cy, cw, ch) in boxes:
                ox1 = x1 + cx / scale
                oy1 = y1 + cy / scale
                ox2 = x1 + (cx + cw) / scale
                oy2 = y1 + (cy + ch) / scale
                result.append([[ox1, oy1], [ox2, oy1], [ox2, oy2], [ox1, oy2]])
            return result

    return []

def _find_plate_rect(binary: np.ndarray, img_w: int, img_h: int):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch < 6:
            continue
        aspect = cw / ch
        if not (1.8 <= aspect <= 8.5):
            continue
        if not (img_w * 0.03 <= cw <= img_w * 0.80):
            continue
        if not (img_h * 0.006 <= ch <= img_h * 0.28):
            continue
        aspect_score = max(0.1, 1.0 - abs(aspect - 4.0) / 4.0)
        y_center_frac = (y + ch / 2) / img_h
        pos_score = 1.0 if y_center_frac >= 0.30 else 0.35
        score = area * aspect_score * pos_score
        if score > best_score:
            best_score, best = score, (x, y, cw, ch)
    return best

def _crop_detected_region(img_bgr: np.ndarray, best):
    if best is None:
        return None, None
    h, w = img_bgr.shape[:2]
    x, y, cw, ch = best
    pad_x = max(8, int(cw * 0.10))
    pad_y = max(6, int(ch * 0.28))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + cw + pad_x)
    y2 = min(h, y + ch + pad_y)
    return img_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)

def _detect_red_plate_region(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    red_lo = cv2.inRange(hsv, np.array([0, 100, 80]),   np.array([8, 255, 255]))
    red_hi = cv2.inRange(hsv, np.array([172, 100, 80]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_lo, red_hi)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
                                iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                iterations=1)
    best = _find_plate_rect(red_mask, w, h)
    return _crop_detected_region(img_bgr, best)

def _detect_plate_region(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]
    k_wide = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    k_sq   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, 35, 15]), np.array([148, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_wide, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_sq,   iterations=1)
    best = _find_plate_rect(mask, w, h)

    if best is None:
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, gbin = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gbin = cv2.morphologyEx(gbin, cv2.MORPH_CLOSE, k_wide, iterations=5)
        best = _find_plate_rect(gbin, w, h)

    return _crop_detected_region(img_bgr, best)

def _order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).flatten()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)

def _deskew_plate(plate_img: np.ndarray):
    h, w = plate_img.shape[:2]
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plate_cnt = None
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > w * h * 0.25:
            plate_cnt = approx
            break

    eye = np.eye(3, dtype=np.float32)
    if plate_cnt is None:
        return plate_img, eye, eye

    src = _order_points(plate_cnt)
    out_w = max(w, int(h * 4.5))
    out_h = int(out_w / 4.5)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M     = cv2.getPerspectiveTransform(src, dst)
    M_inv = np.linalg.inv(M)
    corrected = cv2.warpPerspective(plate_img, M, (out_w, out_h),
                                    flags=cv2.INTER_CUBIC)
    return corrected, M, M_inv

def _merge_close_chars(chars: list, plate_h: int) -> list:
    if not chars:
        return []
    min_gap = max(3, plate_h * 0.08)
    merged = [list(chars[0])]
    for cx, cy, cw, ch in chars[1:]:
        prev = merged[-1]
        if cx - (prev[0] + prev[2]) < min_gap:
            nx1, ny1 = min(prev[0], cx), min(prev[1], cy)
            nx2, ny2 = max(prev[0] + prev[2], cx + cw), max(prev[1] + prev[3], cy + ch)
            merged[-1] = [nx1, ny1, nx2 - nx1, ny2 - ny1]
        else:
            merged.append([cx, cy, cw, ch])
    return [tuple(m) for m in merged]

def _segment_chars_from_plate(plate_img: np.ndarray) -> list:
    h, w = plate_img.shape[:2]
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)

    candidates = [
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.threshold(cv2.bitwise_not(gray), 0, 255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 15, 4),
        cv2.adaptiveThreshold(cv2.bitwise_not(gray), 255,
                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 15, 4),
    ]

    for binary in candidates:
        n_lbl, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        chars = []
        for lbl in range(1, n_lbl):
            cx   = stats[lbl, cv2.CC_STAT_LEFT]
            cy   = stats[lbl, cv2.CC_STAT_TOP]
            cw   = stats[lbl, cv2.CC_STAT_WIDTH]
            ch   = stats[lbl, cv2.CC_STAT_HEIGHT]
            area = stats[lbl, cv2.CC_STAT_AREA]
            if not (h * 0.30 <= ch <= h * 0.95):
                continue
            if cw < max(3, w * 0.015) or cw > w * 0.40:
                continue
            if area < max(40, h * cw * 0.15):
                continue
            if not (0.08 <= cw / ch <= 2.8):
                continue
            chars.append((cx, cy, cw, ch))

        chars.sort(key=lambda c: c[0])
        merged = _merge_close_chars(chars, h)

        if 4 <= len(merged) <= 10:
            return merged

    return []

def _run_ocr_best(img: np.ndarray) -> list:
    img_h, img_w = img.shape[:2]
    is_plate = (img_w >= img_h * 2.0 and img_w >= 150)
    red_ratio = _red_ratio(img) if is_plate else 0.0
    is_red_plate = is_plate and red_ratio >= 0.06

    def run_rapid(img_input):
        result, _ = rapid_reader(img_input)
        if not result:
            return []
        return [(item[0], item[1], float(item[2])) for item in result]

    if is_red_plate:
        red_proc = _preprocess_red_plate_text(img)
        res = run_rapid(red_proc)
        txt, conf, _ = _find_plate_text(res, img_h)
        if txt and _PLATE_RE.search(txt):
            return res

        res_inv = run_rapid(cv2.bitwise_not(red_proc))
        txt_inv, conf_inv, _ = _find_plate_text(res_inv, img_h)
        if txt_inv and _PLATE_RE.search(txt_inv):
            return res_inv

        proc = _preprocess_for_ocr(img)
        res_std = run_rapid(proc)
        txt_std, conf_std, _ = _find_plate_text(res_std, img_h)
        if txt_std and _PLATE_RE.search(txt_std):
            return res_std

        candidates = [(res, conf), (res_inv, conf_inv), (res_std, conf_std)]
    else:
        proc = _preprocess_for_ocr(img)
        res = run_rapid(proc)
        txt, conf, _ = _find_plate_text(res, img_h)
        if txt and _PLATE_RE.search(txt):
            return res

        res_inv = run_rapid(cv2.bitwise_not(proc))
        txt_inv, conf_inv, _ = _find_plate_text(res_inv, img_h)
        if txt_inv and _PLATE_RE.search(txt_inv):
            return res_inv

        if is_plate:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe_g = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            gray = clahe_g.apply(gray)
            bin_img = cv2.adaptiveThreshold(gray, 255,
                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                          cv2.THRESH_BINARY, 15, 3)
            res_bin = run_rapid(cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR))
            txt_bin, conf_bin, _ = _find_plate_text(res_bin, img_h)
            if txt_bin and _PLATE_RE.search(txt_bin):
                return res_bin
            candidates = [(res, conf), (res_inv, conf_inv), (res_bin, conf_bin)]
        else:
            candidates = [(res, conf), (res_inv, conf_inv)]

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def _normalize_crop_area(crop_area, img_w: int, img_h: int):
    if not crop_area:
        return None
    cx1, cy1, cx2, cy2 = (float(v) for v in crop_area)
    if cx2 <= 1.0 and cy2 <= 1.0:
        if cx1 <= 0.01 and cy1 <= 0.01 and cx2 >= 0.99 and cy2 >= 0.99:
            return None
        return cx1, cy1, cx2, cy2
    if cx1 <= 2 and cy1 <= 2 and cx2 >= img_w - 2 and cy2 >= img_h - 2:
        return None
    return cx1, cy1, cx2, cy2

def _run_ocr_candidate(ocr_img: np.ndarray, offset_x: int, offset_y: int):
    work_img = ocr_img
    ocr_scale = 1.0
    max_side = max(work_img.shape[:2])
    if max_side > _MAX_OCR_SIDE:
        ocr_scale = _MAX_OCR_SIDE / max_side
    elif work_img.shape[1] < 960 and (offset_x > 0 or offset_y > 0):
        ocr_scale = 960 / work_img.shape[1]

    if ocr_scale != 1.0:
        new_w = int(work_img.shape[1] * ocr_scale)
        new_h = int(work_img.shape[0] * ocr_scale)
        work_img = cv2.resize(work_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    results = _run_ocr_best(work_img)

    if ocr_scale != 1.0:
        results = [([[pt[0] / ocr_scale, pt[1] / ocr_scale] for pt in bbox], text, prob)
                   for bbox, text, prob in results]

    if offset_x > 0 or offset_y > 0:
        adjusted_results = []
        for (bbox, text, prob) in results:
            adj_bbox = [[pt[0] + offset_x, pt[1] + offset_y] for pt in bbox]
            adjusted_results.append((adj_bbox, text, prob))
        results = adjusted_results

    return work_img, results, ocr_scale

def process_image_ocr(file_bytes: bytes, filename: str, crop_area: tuple | None = None):
    image_hash = compute_hash(file_bytes)
    nparr      = np.frombuffer(file_bytes, np.uint8)
    original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if original_img is None:
        raise ValueError(f"Failed to decode image: {filename}")
    img_h, img_w = original_img.shape[:2]

    ocr_img = original_img
    offset_x, offset_y = 0, 0
    normalized_crop = _normalize_crop_area(crop_area, img_w, img_h)

    if normalized_crop:
        cx1, cy1, cx2, cy2 = normalized_crop
        if cx2 <= 1.0 and cy2 <= 1.0:
            x1, y1 = int(img_w * cx1), int(img_h * cy1)
            x2, y2 = int(img_w * cx2), int(img_h * cy2)
        else:
            x1, y1, x2, y2 = int(cx1), int(cy1), int(cx2), int(cy2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 > x1 and y2 > y1:
            ocr_img = original_img[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
    else:
        detected, det_coords = _detect_plate_region(original_img)
        if detected is not None and det_coords is not None:
            ocr_img  = detected
            offset_x = det_coords[0]
            offset_y = det_coords[1]

    ocr_img, results, ocr_scale = _run_ocr_candidate(ocr_img, offset_x, offset_y)
    plate_text, plate_conf, plate_bboxes = _find_plate_text(results, img_h)

    # Fallback: crop auto-detect gagal → retry full image
    if not plate_text and normalized_crop is None and (offset_x > 0 or offset_y > 0):
        full_img2, full_results, full_scale = _run_ocr_candidate(original_img, 0, 0)
        full_text, full_conf, full_bboxes = _find_plate_text(full_results, img_h)
        if full_text:
            ocr_img      = full_img2
            results      = full_results
            ocr_scale    = full_scale
            offset_x     = 0
            offset_y     = 0
            plate_text   = full_text
            plate_conf   = full_conf
            plate_bboxes = full_bboxes

    clean_current = re.sub(r'[^A-Z0-9]', '', plate_text) if plate_text else ""
    if not plate_text or not _PLATE_RE.search(plate_text) or len(clean_current) < 4:
        red_detected, red_coords = _detect_red_plate_region(original_img)
        if red_detected is not None and red_coords is not None:
            red_img, red_results, red_scale = _run_ocr_candidate(red_detected, red_coords[0], red_coords[1])
            red_text, red_conf, red_bboxes = _find_plate_text(red_results, img_h)
            if red_text and _PLATE_RE.search(red_text):
                clean_red = re.sub(r'[^A-Z0-9]', '', red_text)
                if len(clean_red) > len(clean_current):
                    ocr_img = red_img
                    results = red_results
                    ocr_scale = red_scale
                    offset_x = red_coords[0]
                    offset_y = red_coords[1]
                    plate_text = red_text
                    plate_conf = red_conf
                    plate_bboxes = red_bboxes

    if plate_bboxes:
        roi, (rx1, ry1, rx2, ry2) = _crop_plate_roi(original_img, plate_bboxes)
    else:
        ry1 = int(img_h * 0.55)
        rx1 = int(img_w * 0.20)
        rx2 = int(img_w * 0.80)
        ry2 = img_h
        roi  = original_img[ry1:ry2, rx1:rx2]

    plate_phash = compute_phash(roi)

    clean_plate = re.sub(r'[^A-Z0-9]', '', plate_text) if plate_text else ""
    is_plate_crop = (offset_x > 0 or offset_y > 0 or normalized_crop is not None)
    seg_chars = _segment_chars_from_plate(ocr_img) if (is_plate_crop and clean_plate) else []

    if seg_chars and len(seg_chars) == len(clean_plate):
        details = []
        for i, (cx, cy, cw, ch) in enumerate(seg_chars):
            ox1 = int(cx / ocr_scale + offset_x)
            oy1 = int(cy / ocr_scale + offset_y)
            ox2 = int((cx + cw) / ocr_scale + offset_x)
            oy2 = int((cy + ch) / ocr_scale + offset_y)
            clean_box = [[ox1, oy1], [ox2, oy1], [ox2, oy2], [ox1, oy2]]
            char_text = clean_plate[i]
            char_crop = original_img[max(0, oy1):max(0, oy2), max(0, ox1):max(0, ox2)]
            c_phash   = compute_phash(char_crop) if char_crop.size > 0 else ""
            details.append({
                "box":        clean_box,
                "text":       char_text,
                "confidence": 0.85,
                "is_learned": False,
                "is_plate":   True,
                "char_phash": c_phash,
            })
    else:
        plate_bbox_keys = set(_bbox_key(pb) for pb in plate_bboxes) if plate_bboxes else set()
        details = []
        for (bbox, text, prob) in results:
            is_plate_box = _bbox_key(bbox) in plate_bbox_keys
            char_bboxes  = _split_box_per_char(bbox, text, original_img)
            for char, char_bbox in zip(text, char_bboxes):
                clean_box = [[int(pt[0]), int(pt[1])] for pt in char_bbox]
                c_phash = ""
                if is_plate_box:
                    x_coords = [p[0] for p in clean_box]
                    y_coords = [p[1] for p in clean_box]
                    char_crop = original_img[max(0, min(y_coords)):max(0, max(y_coords)),
                                             max(0, min(x_coords)):max(0, max(x_coords))]
                    c_phash = compute_phash(char_crop) if char_crop.size > 0 else ""
                details.append({
                    "box":        clean_box,
                    "text":       char,
                    "confidence": float(prob),
                    "is_learned": False,
                    "is_plate":   is_plate_box,
                    "char_phash": c_phash,
                })

    return {
        "filename":         filename,
        "image_hash":       image_hash,
        "plate_phash":      plate_phash,
        "plate_confidence": float(plate_conf) if plate_conf else 0.0,
        "roi_path":         None,
        "full_text":        plate_text,
        "details":          details,
        "source":           "ocr"
    }


# =============================================================================
# API
# =============================================================================
import json, secrets, time, hmac, hashlib, base64
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# --- Auth config (set via env var, fallback ke default) ---
_SECRET_KEY      = os.getenv("SECRET_KEY", secrets.token_hex(32))
_API_USERNAME    = os.getenv("API_USERNAME", "CODA")
_API_PASSWORD    = os.getenv("API_PASSWORD") or secrets.token_urlsafe(12)
_TOKEN_EXPIRE_H  = int(os.getenv("TOKEN_EXPIRE_HOURS", "24"))

_bearer = HTTPBearer(auto_error=False)

def _make_token(username: str) -> str:
    exp     = int(time.time()) + _TOKEN_EXPIRE_H * 3600
    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": username, "exp": exp}).encode()
    ).decode().rstrip("=")
    sig = hmac.new(_SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"

def _verify_token(token: str) -> str | None:
    try:
        payload_b64, sig = token.rsplit(".", 1)
        expected = hmac.new(_SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        padding  = "=" * (-len(payload_b64) % 4)
        data     = json.loads(base64.urlsafe_b64decode(payload_b64 + padding))
        if data["exp"] < time.time():
            return None
        return data["sub"]
    except Exception:
        return None

def _require_auth(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    token = creds.credentials if creds else None
    if not token:
        raise HTTPException(401, "Token tidak ditemukan", headers={"WWW-Authenticate": "Bearer"})
    user = _verify_token(token)
    if not user:
        raise HTTPException(401, "Token tidak valid atau sudah expired", headers={"WWW-Authenticate": "Bearer"})
    return user


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        rapid_reader(np.zeros((100, 100, 3), dtype=np.uint8))
    except Exception:
        pass
    print("=" * 48)
    print(f"  Username : {_API_USERNAME}")
    print(f"  Password : {_API_PASSWORD}")
    print(f"  Token expires in {_TOKEN_EXPIRE_H}h")
    print("=" * 48)
    yield

app = FastAPI(
    title="Parking LPR API",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/token")
async def login(username: str = Form(...), password: str = Form(...)):
    if username != _API_USERNAME or password != _API_PASSWORD:
        raise HTTPException(401, "Username atau password salah")
    token = _make_token(username)
    return {
        "access_token": token,
        "token_type":   "bearer",
        "expires_in":   _TOKEN_EXPIRE_H * 3600,
    }


@app.post("/recognize")
async def recognize(
    image: UploadFile = File(...),
    crop_area: Optional[str] = Form(None),
    _user: str = Depends(_require_auth),
):
    if not image:
        raise HTTPException(400, "File kosong")
    contents = await image.read()
    if not contents:
        raise HTTPException(400, "File kosong")

    crop_tuple = None
    if crop_area:
        try:
            d = json.loads(crop_area)
            crop_tuple = (float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"]))
        except Exception:
            crop_tuple = None

    ext           = (image.filename or "upload.jpg").rsplit(".", 1)[-1]
    safe_filename = f"{uuid.uuid4()}.{ext}"

    try:
        result = await run_in_threadpool(process_image_ocr, contents, safe_filename, crop_tuple)
    except ValueError as e:
        raise HTTPException(400, str(e))

    plate_text = (result.get("full_text") or "").strip().upper()

    if not plate_text and crop_tuple is not None:
        try:
            result     = await run_in_threadpool(process_image_ocr, contents, safe_filename, None)
            plate_text = (result.get("full_text") or "").strip().upper()
        except Exception:
            pass

    if not plate_text and result.get("details"):
        try:
            plate_text = result["details"][0].get("text", "").strip().upper()
        except Exception:
            plate_text = ""

    return {"full_text": plate_text if plate_text else None}


def _debug_ocr(file_bytes: bytes) -> dict:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "gagal decode gambar"}

    h, w = img.shape[:2]

    try:
        raw, _ = rapid_reader(img)
        raw_texts = [(item[1], round(float(item[2]), 3)) for item in raw] if raw else []
    except Exception as e:
        raw_texts = [f"ERROR: {e}"]

    detected, det_coords = _detect_plate_region(img)
    plate_region = str(det_coords) if det_coords else "tidak terdeteksi"

    plate_texts = []
    if detected is not None:
        try:
            raw2, _ = rapid_reader(detected)
            plate_texts = [(item[1], round(float(item[2]), 3)) for item in raw2] if raw2 else []
        except Exception as e:
            plate_texts = [f"ERROR: {e}"]

    return {
        "img_size":        f"{w}x{h}",
        "raw_ocr_count":   len(raw_texts),
        "raw_ocr_texts":   raw_texts,
        "plate_region":    plate_region,
        "plate_ocr_count": len(plate_texts),
        "plate_ocr_texts": plate_texts,
    }

@app.post("/recognize-debug")
async def recognize_debug(
    image: UploadFile = File(...),
    _user: str = Depends(_require_auth),
):
    if not image:
        raise HTTPException(400, "File kosong")
    contents = await image.read()
    if not contents:
        raise HTTPException(400, "File kosong")
    return await run_in_threadpool(_debug_ocr, contents)


if __name__ == "__main__":
    import uvicorn
    workers = int(os.getenv("WORKERS", "1"))
    reload  = os.getenv("APP_RELOAD") == "1" and workers == 1
    uvicorn.run("app:app", reload=reload, workers=workers if workers > 1 else None,
                host="0.0.0.0", port=9000)
