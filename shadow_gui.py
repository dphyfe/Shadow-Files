import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading

# Import core functions from shadow_generator
from shadow_generator import _load_image, _ensure_bgr, _load_or_create_mask, _place_on_canvas, _compute_contact_line, _build_shadow, _apply_depth_warp, _composite


class ShadowGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Shadow Generator")
        self.root.geometry("1400x900")

        # State
        self.fg_path = None
        self.bg_path = None
        self.mask_path = None
        self.depth_path = None
        self.fg_bgr = None
        self.mask = None
        self.bg = None
        self.depth_map = None
        self.preview_img = None

        # Create UI
        self._create_ui()

    def _create_ui(self):
        # Left panel - controls
        left_frame = ttk.Frame(self.root, padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # File selection
        ttk.Label(left_frame, text="Images", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        ttk.Button(left_frame, text="Select Foreground", command=self._select_foreground).grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(left_frame, text="Select Background", command=self._select_background).grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(left_frame, text="Select Mask (Optional)", command=self._select_mask).grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(left_frame, text="Select Depth Map (Optional)", command=self._select_depth).grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)

        # Shadow controls
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(left_frame, text="Shadow Controls", font=("Arial", 12, "bold")).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        # Light angle
        row = 7
        ttk.Label(left_frame, text="Light Angle (0-360°):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.angle_var = tk.DoubleVar(value=135.0)
        self.angle_label = ttk.Label(left_frame, text="135.0")
        self.angle_label.grid(row=row, column=1, sticky=tk.E)
        angle_slider = ttk.Scale(left_frame, from_=0, to=360, variable=self.angle_var, command=lambda v: self._update_label(self.angle_label, v))
        angle_slider.grid(row=row + 1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)

        # Light elevation
        row += 2
        ttk.Label(left_frame, text="Light Elevation (0-90°):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.elevation_var = tk.DoubleVar(value=35.0)
        self.elevation_label = ttk.Label(left_frame, text="35.0")
        self.elevation_label.grid(row=row, column=1, sticky=tk.E)
        elevation_slider = ttk.Scale(left_frame, from_=1, to=90, variable=self.elevation_var, command=lambda v: self._update_label(self.elevation_label, v))
        elevation_slider.grid(row=row + 1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)

        # Shadow opacity
        row += 2
        ttk.Label(left_frame, text="Shadow Opacity:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.opacity_var = tk.DoubleVar(value=0.6)
        self.opacity_label = ttk.Label(left_frame, text="0.60")
        self.opacity_label.grid(row=row, column=1, sticky=tk.E)
        opacity_slider = ttk.Scale(left_frame, from_=0, to=1, variable=self.opacity_var, command=lambda v: self._update_label(self.opacity_label, v, fmt=".2f"))
        opacity_slider.grid(row=row + 1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)

        # Blur near
        row += 2
        ttk.Label(left_frame, text="Blur Near Contact:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.blur_near_var = tk.DoubleVar(value=1.5)
        self.blur_near_label = ttk.Label(left_frame, text="1.50")
        self.blur_near_label.grid(row=row, column=1, sticky=tk.E)
        blur_near_slider = ttk.Scale(left_frame, from_=0.1, to=10, variable=self.blur_near_var, command=lambda v: self._update_label(self.blur_near_label, v, fmt=".2f"))
        blur_near_slider.grid(row=row + 1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)

        # Blur far
        row += 2
        ttk.Label(left_frame, text="Blur Far Away:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.blur_far_var = tk.DoubleVar(value=12.0)
        self.blur_far_label = ttk.Label(left_frame, text="12.00")
        self.blur_far_label.grid(row=row, column=1, sticky=tk.E)
        blur_far_slider = ttk.Scale(left_frame, from_=1, to=30, variable=self.blur_far_var, command=lambda v: self._update_label(self.blur_far_label, v, fmt=".2f"))
        blur_far_slider.grid(row=row + 1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)

        # Falloff
        row += 2
        ttk.Label(left_frame, text="Shadow Falloff:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.falloff_var = tk.DoubleVar(value=2.0)
        self.falloff_label = ttk.Label(left_frame, text="2.00")
        self.falloff_label.grid(row=row, column=1, sticky=tk.E)
        falloff_slider = ttk.Scale(left_frame, from_=0.1, to=5, variable=self.falloff_var, command=lambda v: self._update_label(self.falloff_label, v, fmt=".2f"))
        falloff_slider.grid(row=row + 1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)

        # Depth strength
        row += 2
        ttk.Label(left_frame, text="Depth Warp Strength:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.depth_strength_var = tk.DoubleVar(value=0.6)
        self.depth_strength_label = ttk.Label(left_frame, text="0.60")
        self.depth_strength_label.grid(row=row, column=1, sticky=tk.E)
        depth_strength_slider = ttk.Scale(left_frame, from_=0, to=2, variable=self.depth_strength_var, command=lambda v: self._update_label(self.depth_strength_label, v, fmt=".2f"))
        depth_strength_slider.grid(row=row + 1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)

        # Buttons
        row += 2
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        row += 1
        ttk.Button(left_frame, text="Generate Preview", command=self._generate_preview).grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1
        ttk.Button(left_frame, text="Save Outputs", command=self._save_outputs).grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Right panel - preview
        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(right_frame, text="Preview", font=("Arial", 12, "bold")).pack(pady=(0, 10))

        self.canvas = tk.Canvas(right_frame, width=800, height=800, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

    def _update_label(self, label, value, fmt=".1f"):
        label.config(text=f"{float(value):{fmt}}")

    def _select_foreground(self):
        path = filedialog.askopenfilename(title="Select Foreground Image", filetypes=[("Images", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG")])
        if path:
            self.fg_path = Path(path)
            messagebox.showinfo("Success", f"Foreground loaded: {self.fg_path.name}")

    def _select_background(self):
        path = filedialog.askopenfilename(title="Select Background Image", filetypes=[("Images", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG")])
        if path:
            self.bg_path = Path(path)
            messagebox.showinfo("Success", f"Background loaded: {self.bg_path.name}")

    def _select_mask(self):
        path = filedialog.askopenfilename(title="Select Mask Image (Optional)", filetypes=[("Images", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG")])
        if path:
            self.mask_path = Path(path)
            messagebox.showinfo("Success", f"Mask loaded: {self.mask_path.name}")
        else:
            self.mask_path = None

    def _select_depth(self):
        path = filedialog.askopenfilename(title="Select Depth Map (Optional)", filetypes=[("Images", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG")])
        if path:
            self.depth_path = Path(path)
            messagebox.showinfo("Success", f"Depth map loaded: {self.depth_path.name}")
        else:
            self.depth_path = None

    def _generate_preview(self):
        if not self.fg_path or not self.bg_path:
            messagebox.showerror("Error", "Please select both foreground and background images")
            return

        # Run in thread to avoid freezing UI
        thread = threading.Thread(target=self._generate_composite)
        thread.start()

    def _generate_composite(self):
        try:
            # Load images
            self.bg = _load_image(self.bg_path, cv2.IMREAD_COLOR)
            self.bg = _ensure_bgr(self.bg)
            bh, bw = self.bg.shape[:2]

            self.fg_bgr, self.mask = _load_or_create_mask(self.fg_path, self.mask_path)

            fh, fw = self.fg_bgr.shape[:2]
            offset_x = (bw - fw) // 2
            offset_y = bh - fh

            fg_canvas = _place_on_canvas(self.fg_bgr, self.bg.shape, offset_x, offset_y)
            mask_canvas = _place_on_canvas(self.mask, self.bg.shape, offset_x, offset_y)
            mask_canvas = cv2.GaussianBlur(mask_canvas, (0, 0), 0.5)

            contact_line = _compute_contact_line(mask_canvas)

            shadow_alpha = _build_shadow(
                mask_canvas,
                contact_line,
                self.angle_var.get(),
                self.elevation_var.get(),
                self.opacity_var.get(),
                self.blur_near_var.get(),
                self.blur_far_var.get(),
                self.falloff_var.get(),
            )

            if self.depth_path:
                self.depth_map = _load_image(self.depth_path, cv2.IMREAD_GRAYSCALE)
                shadow_alpha = _apply_depth_warp(shadow_alpha, self.depth_map, self.angle_var.get(), self.elevation_var.get(), self.depth_strength_var.get())

            composite = _composite(self.bg, fg_canvas, mask_canvas, shadow_alpha)

            # Store for saving
            self.preview_img = composite

            # Display preview
            self._display_preview(composite)

        except Exception as e:
            messagebox.showerror("Error", f"Generation failed: {str(e)}")

    def _display_preview(self, img):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to fit canvas
        h, w = img_rgb.shape[:2]
        max_w, max_h = 800, 800
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Convert to PhotoImage
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # Keep reference

    def _save_outputs(self):
        if self.preview_img is None:
            messagebox.showerror("Error", "Generate preview first")
            return

        out_dir = Path("outputs")
        out_dir.mkdir(exist_ok=True)

        cv2.imwrite(str(out_dir / "composite.png"), self.preview_img)

        # Shadow only
        if hasattr(self, "shadow_alpha"):
            bh, bw = self.bg.shape[:2]
            shadow_rgba = np.zeros((bh, bw, 4), dtype=np.uint8)
            shadow_rgba[:, :, 3] = np.clip(self.shadow_alpha * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(out_dir / "shadow_only.png"), shadow_rgba)

        # Mask debug
        if self.mask is not None:
            mask_canvas = _place_on_canvas(self.mask, self.bg.shape, (self.bg.shape[1] - self.mask.shape[1]) // 2, self.bg.shape[0] - self.mask.shape[0])
            cv2.imwrite(str(out_dir / "mask_debug.png"), mask_canvas)

        messagebox.showinfo("Success", f"Outputs saved to {out_dir.absolute()}")


def main():
    root = tk.Tk()
    app = ShadowGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
