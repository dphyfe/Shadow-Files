import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def _load_image(path, flags=cv2.IMREAD_UNCHANGED):
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def _ensure_bgr(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def _extract_alpha(fg_image):
    if fg_image.ndim == 3 and fg_image.shape[2] == 4:
        alpha = fg_image[:, :, 3]
        bgr = fg_image[:, :, :3]
        return bgr, alpha
    return fg_image, None


def _grabcut_mask(bgr, rect_margin=0.03, iterations=7):
    h, w = bgr.shape[:2]
    margin_x = int(w * rect_margin)
    margin_y = int(h * rect_margin)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)

    border = max(5, int(min(h, w) * 0.04))
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    border_pixels = bgr.copy().astype(np.float32)
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True

    bg_samples = border_pixels[border_mask]
    if bg_samples.size:
        mean = bg_samples.mean(axis=0)
        std = bg_samples.std(axis=0)
        dist = np.linalg.norm(bgr.astype(np.float32) - mean, axis=2)
        bg_dist = np.linalg.norm(bg_samples - mean, axis=1)
        thresh = float(bg_dist.mean() + bg_dist.std() * 1.5 + 10.0)
        probable_fg = dist > thresh
        mask[probable_fg] = cv2.GC_PR_FGD

    center = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(center, (w // 2, h // 2), (int(w * 0.24), int(h * 0.34)), 0, 0, 360, 255, -1)
    mask[center > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
    mask_bin = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    if mask_bin.sum() < (h * w * 0.05):
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        mask_bin = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    return mask_bin


def _grabcut_refine(initial_mask, bgr, fg_erode=3, bg_dilate=5, iterations=3):
    h, w = initial_mask.shape
    gc_mask = np.full((h, w), cv2.GC_PR_FGD, np.uint8)

    fg_seed = cv2.erode(initial_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fg_erode, fg_erode)))
    bg_seed = cv2.dilate(255 - initial_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_dilate, bg_dilate)))

    gc_mask[fg_seed > 0] = cv2.GC_FGD
    gc_mask[bg_seed > 200] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, gc_mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)

    mask_bin = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return mask_bin


def _suppress_bg_by_color(mask, bgr, border_frac=0.08, sigma_scale=1.2, min_thresh=10.0, min_retain=0.7):
    h, w = mask.shape
    border = max(2, int(min(h, w) * border_frac))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True

    samples = bgr[border_mask].astype(np.float32)
    if samples.size == 0:
        return mask
    mean = samples.mean(axis=0)
    std = samples.std(axis=0).mean()
    thresh = max(min_thresh, std * sigma_scale + 5.0)

    dist = np.linalg.norm(bgr.astype(np.float32) - mean, axis=2)
    bg_like = dist < thresh

    mask_clean = mask.copy()
    before = float(mask_clean.sum())
    mask_clean[(bg_like) & (mask_clean > 0)] = 0
    mask_clean = _largest_component(mask_clean)
    mask_clean = cv2.dilate(mask_clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    after = float(mask_clean.sum())

    if before == 0:
        return mask
    if after < before * min_retain:  # too destructive, fallback
        return mask
    return mask_clean


def _protect_bottom(mask, frac=0.1):
    h, w = mask.shape
    band = max(1, int(h * frac))
    band_mask = np.zeros_like(mask)
    band_mask[h - band :, :] = 255

    band_region = cv2.bitwise_and(mask, band_mask)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
    band_region = cv2.morphologyEx(band_region, cv2.MORPH_CLOSE, close_k, iterations=2)
    band_region = cv2.dilate(band_region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    combined = mask.copy()
    combined[h - band :, :] = np.maximum(combined[h - band :, :], band_region[h - band :, :])
    return combined


def _suppress_gray(mask, bgr, sat_thresh=28, v_min=25, v_max=230, min_retain=0.8, aggressive=False):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    gray_like = (sat < sat_thresh) & (val > v_min) & (val < v_max)

    if aggressive:
        edges = cv2.Canny(cv2.GaussianBlur(bgr, (5, 5), 0), 30, 100)
        edges_dilated = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        gray_like = gray_like & (edges_dilated == 0)

    mask_clean = mask.copy()
    before = float(mask_clean.sum())
    mask_clean[(gray_like) & (mask_clean > 0)] = 0

    if aggressive:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_clean = cv2.erode(mask_clean, kernel, iterations=1)

    mask_clean = _largest_component(mask_clean)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    after = float(mask_clean.sum())
    if before == 0:
        return mask
    if after < before * min_retain:
        return mask
    return mask_clean


def _largest_component(mask):
    num, labels = cv2.connectedComponents(mask, connectivity=8)
    if num <= 1:
        return mask
    counts = np.bincount(labels.flatten())
    counts[0] = 0
    max_label = counts.argmax()
    return np.where(labels == max_label, 255, 0).astype(np.uint8)


def _fill_holes(mask):
    h, w = mask.shape
    inv = cv2.bitwise_not(mask)
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    flood = inv.copy()
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, flood_inv)
    return filled


def _refine_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.medianBlur(closed, 5)
    filled = _fill_holes(closed)
    refined = cv2.dilate(filled, kernel, iterations=1)

    refined = _largest_component(refined)
    tight_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, tight_kernel, iterations=1)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, tight_kernel, iterations=1)
    return refined


def _load_or_create_mask(fg_path, mask_path=None):
    fg_raw = _load_image(fg_path)
    fg_bgr, alpha = _extract_alpha(fg_raw)

    if mask_path:
        mask = _load_image(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (fg_bgr.shape[1], fg_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        return fg_bgr, mask
    elif alpha is not None:
        mask = alpha
    else:
        mask = _grabcut_mask(fg_bgr)

    mask = _refine_mask(mask)
    mask = _suppress_bg_by_color(mask, fg_bgr, min_retain=0.80)
    mask = _suppress_gray(mask, fg_bgr, sat_thresh=35, v_min=15, v_max=240, min_retain=0.75, aggressive=True)

    # secondary GrabCut refinement using mask as trimap
    mask = _grabcut_refine(mask, fg_bgr, fg_erode=6, bg_dilate=6, iterations=3)
    mask = _refine_mask(mask)
    mask = _suppress_bg_by_color(mask, fg_bgr, border_frac=0.06, sigma_scale=0.9, min_thresh=6.0, min_retain=0.85)
    mask = _suppress_gray(mask, fg_bgr, sat_thresh=32, v_min=12, v_max=238, min_retain=0.80, aggressive=True)
    mask = _suppress_gray(mask, fg_bgr, sat_thresh=38, v_min=10, v_max=242, min_retain=0.75, aggressive=True)
    mask = _refine_mask(mask)
    mask = _protect_bottom(mask, frac=0.12)

    return fg_bgr, mask


def _place_on_canvas(image, canvas_shape, offset_x, offset_y):
    h, w = canvas_shape[:2]
    ih, iw = image.shape[:2]
    canvas = np.zeros((h, w) + image.shape[2:], dtype=image.dtype) if image.ndim == 3 else np.zeros((h, w), dtype=image.dtype)

    x0 = max(0, offset_x)
    y0 = max(0, offset_y)
    x1 = min(w, offset_x + iw)
    y1 = min(h, offset_y + ih)

    src_x0 = max(0, -offset_x)
    src_y0 = max(0, -offset_y)
    src_x1 = src_x0 + (x1 - x0)
    src_y1 = src_y0 + (y1 - y0)

    if x1 <= x0 or y1 <= y0:
        return canvas

    if image.ndim == 3:
        canvas[y0:y1, x0:x1, :] = image[src_y0:src_y1, src_x0:src_x1, :]
    else:
        canvas[y0:y1, x0:x1] = image[src_y0:src_y1, src_x0:src_x1]

    return canvas


def _compute_contact_line(mask):
    h, w = mask.shape
    contact = np.zeros_like(mask, dtype=np.uint8)
    for x in range(w):
        ys = np.where(mask[:, x] > 0)[0]
        if ys.size:
            contact[ys.max(), x] = 255
    return contact


def _build_shadow(mask, contact_line, light_angle, elevation, base_opacity, blur_near, blur_far, falloff):
    h, w = mask.shape

    shadow_angle = (light_angle + 180.0) % 360.0
    angle_rad = math.radians(shadow_angle)
    dir_x = math.cos(angle_rad)
    dir_y = math.sin(angle_rad)

    elevation = max(1e-3, min(89.0, elevation))
    elev_rad = math.radians(elevation)

    ys = np.where(mask > 0)[0]
    if ys.size == 0:
        return np.zeros_like(mask, dtype=np.float32)
    y0 = ys.max()
    height = max(1.0, ys.max() - ys.min())

    shadow_length = height / math.tan(elev_rad)
    k = shadow_length / height

    a = dir_x * k
    b = dir_y * k

    M = np.array([[1.0, -a, a * y0], [0.0, 1.0 - b, b * y0]], dtype=np.float32)

    shadow_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    shadow_mask_f = shadow_mask.astype(np.float32) / 255.0

    contact_warped = cv2.warpAffine(contact_line, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    dist_input = np.full((h, w), 255, dtype=np.uint8)
    dist_input[contact_warped > 0] = 0
    dist = cv2.distanceTransform(dist_input, cv2.DIST_L2, 5)
    dist = dist * (shadow_mask_f > 0)

    max_dist = float(dist[shadow_mask_f > 0].max()) if np.any(shadow_mask_f > 0) else 1.0
    max_dist = max(1.0, max_dist)
    d_norm = dist / max_dist

    blur_near_img = cv2.GaussianBlur(shadow_mask_f, (0, 0), blur_near)
    blur_far_img = cv2.GaussianBlur(shadow_mask_f, (0, 0), blur_far)

    blended = blur_near_img * (1.0 - d_norm) + blur_far_img * d_norm
    opacity = base_opacity * np.exp(-falloff * d_norm)

    shadow_alpha = np.clip(blended * opacity, 0.0, 1.0)

    # contact shadow
    contact_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    contact_blur = cv2.GaussianBlur(cv2.dilate(contact_warped, contact_kernel), (0, 0), 1.0)
    contact_alpha = (contact_blur.astype(np.float32) / 255.0) * min(1.0, base_opacity * 1.5)

    shadow_alpha = np.clip(shadow_alpha + contact_alpha, 0.0, 1.0)

    return shadow_alpha


def _apply_depth_warp(shadow_alpha, depth_map, light_angle, elevation, strength):
    if depth_map is None:
        return shadow_alpha

    h, w = shadow_alpha.shape
    depth = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    shadow_angle = (light_angle + 180.0) % 360.0
    angle_rad = math.radians(shadow_angle)
    dir_x = math.cos(angle_rad)
    dir_y = math.sin(angle_rad)

    elevation = max(1e-3, min(89.0, elevation))
    elev_rad = math.radians(elevation)
    length_scale = 1.0 / math.tan(elev_rad)

    shift = (0.5 - depth) * strength * length_scale * 200.0

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + dir_x * shift).astype(np.float32)
    map_y = (grid_y + dir_y * shift).astype(np.float32)

    warped = cv2.remap(shadow_alpha, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


def _composite(bg, fg, fg_alpha, shadow_alpha):
    h, w = bg.shape[:2]
    fg_alpha_f = fg_alpha.astype(np.float32) / 255.0

    shadow_alpha_f = shadow_alpha.astype(np.float32)
    shadow_alpha_f = np.clip(shadow_alpha_f, 0.0, 1.0)

    shadow_layer = bg.astype(np.float32) * (1.0 - shadow_alpha_f[:, :, None])

    fg_layer = fg.astype(np.float32) * fg_alpha_f[:, :, None] + shadow_layer * (1.0 - fg_alpha_f[:, :, None])

    return np.clip(fg_layer, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Realistic shadow generator")
    parser.add_argument("--foreground", required=True, help="Path to foreground image")
    parser.add_argument("--background", required=True, help="Path to background image")
    parser.add_argument("--mask", help="Optional mask image (white subject)")
    parser.add_argument("--depth", help="Optional depth map (grayscale)")
    parser.add_argument("--angle", type=float, default=135.0, help="Light angle (0-360). Shadow is cast opposite.")
    parser.add_argument("--elevation", type=float, default=35.0, help="Light elevation (0-90)")
    parser.add_argument("--opacity", type=float, default=0.6, help="Base shadow opacity")
    parser.add_argument("--blur-near", type=float, default=1.5, help="Blur near contact")
    parser.add_argument("--blur-far", type=float, default=12.0, help="Blur far away")
    parser.add_argument("--falloff", type=float, default=2.0, help="Opacity falloff strength")
    parser.add_argument("--depth-strength", type=float, default=0.6, help="Depth warp strength")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale foreground size")
    parser.add_argument("--offset-x", type=int, default=None, help="Foreground x offset (default center)")
    parser.add_argument("--offset-y", type=int, default=None, help="Foreground y offset (default bottom aligned)")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")

    args = parser.parse_args()

    fg_path = Path(args.foreground)
    bg_path = Path(args.background)
    mask_path = Path(args.mask) if args.mask else None
    depth_path = Path(args.depth) if args.depth else None

    bg = _load_image(bg_path, cv2.IMREAD_COLOR)
    bg = _ensure_bgr(bg)
    bh, bw = bg.shape[:2]

    fg_bgr, mask = _load_or_create_mask(fg_path, mask_path)

    if args.scale != 1.0:
        fg_bgr = cv2.resize(fg_bgr, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (fg_bgr.shape[1], fg_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    fh, fw = fg_bgr.shape[:2]
    if args.offset_x is None:
        offset_x = (bw - fw) // 2
    else:
        offset_x = args.offset_x
    if args.offset_y is None:
        offset_y = bh - fh
    else:
        offset_y = args.offset_y

    fg_canvas = _place_on_canvas(fg_bgr, bg.shape, offset_x, offset_y)
    mask_canvas = _place_on_canvas(mask, bg.shape, offset_x, offset_y)
    mask_canvas = cv2.GaussianBlur(mask_canvas, (0, 0), 0.5)

    contact_line = _compute_contact_line(mask_canvas)

    shadow_alpha = _build_shadow(
        mask_canvas,
        contact_line,
        args.angle,
        args.elevation,
        args.opacity,
        args.blur_near,
        args.blur_far,
        args.falloff,
    )

    depth_map = _load_image(depth_path, cv2.IMREAD_GRAYSCALE) if depth_path else None
    shadow_alpha = _apply_depth_warp(shadow_alpha, depth_map, args.angle, args.elevation, args.depth_strength)

    composite = _composite(bg, fg_canvas, mask_canvas, shadow_alpha)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Outputs
    composite_path = out_dir / "composite.png"
    shadow_only_path = out_dir / "shadow_only.png"
    mask_debug_path = out_dir / "mask_debug.png"

    cv2.imwrite(str(composite_path), composite)

    shadow_rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
    shadow_rgba[:, :, 3] = np.clip(shadow_alpha * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(shadow_only_path), shadow_rgba)

    cv2.imwrite(str(mask_debug_path), mask_canvas)

    print(f"Saved: {composite_path}")
    print(f"Saved: {shadow_only_path}")
    print(f"Saved: {mask_debug_path}")


if __name__ == "__main__":
    main()
