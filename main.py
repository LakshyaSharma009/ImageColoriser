import logging
import os
import threading
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from colorizer import MAX_IMAGE_PIXELS, VALID_EXTENSIONS, colorize_image, configure_logging, load_model

configure_logging()
logger = logging.getLogger(__name__)


def cv2_to_tk(cv_img, max_dim=450):
    """Convert a BGR OpenCV image to a Tkinter-displayable image, resized to fit."""
    h, w = cv_img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        cv_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)))
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil_img)


class ColorizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Colorizer")
        self.img = None
        self.colorized = None
        self.model_ready = False

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.open_button = tk.Button(btn_frame, text="Open Image", command=self.open_image, width=15)
        self.open_button.pack(side="left", padx=5)
        self.colorize_button = tk.Button(btn_frame, text="Colorize", command=self.run_colorize, width=15)
        self.colorize_button.pack(side="left", padx=5)
        self.save_button = tk.Button(btn_frame, text="Save Result", command=self.save_result, width=15)
        self.save_button.pack(side="left", padx=5)

        batch_frame = tk.Frame(root)
        batch_frame.pack(pady=(0, 10))
        self.batch_button = tk.Button(batch_frame, text="Batch Colorize Folder", command=self.batch_colorize, width=25)
        self.batch_button.pack()

        # Status label (used for batch progress)
        self.status_label = tk.Label(root, text="", fg="blue")
        self.status_label.pack(pady=(0, 5))

        # Image display area
        img_frame = tk.Frame(root)
        img_frame.pack(padx=10, pady=10)

        self.original_label = tk.Label(img_frame, text="Original", compound="top")
        self.original_label.pack(side="left", padx=10)

        self.colorized_label = tk.Label(img_frame, text="Colorized", compound="top")
        self.colorized_label.pack(side="left", padx=10)

        self.root.after(0, self.load_model_async)

    def load_model_async(self):
        self.set_controls_enabled(False)
        self.set_status("Loading model...")
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _load_model_worker(self):
        try:
            load_model()
        except Exception as exc:
            logger.exception("Model failed to load")
            self.root.after(0, lambda: self._on_model_load_failed(exc))
            return
        self.root.after(0, self._on_model_load_succeeded)

    def _on_model_load_succeeded(self):
        self.model_ready = True
        self.set_controls_enabled(True)
        self.set_status("Model loaded.")
        logger.info("Model loaded successfully")

    def _on_model_load_failed(self, exc):
        self.set_status("Model failed to load.")
        messagebox.showerror(
            "Model load failed",
            f"Could not load the colorization model:\n{exc}\n\n"
            "Make sure the Model/ folder contains the prototxt, caffemodel, and pts_in_hull.npy files.",
        )

    def set_controls_enabled(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.open_button.configure(state=state)
        self.colorize_button.configure(state=state)
        self.save_button.configure(state=state)
        self.batch_button.configure(state=state)

    def set_status(self, text):
        self.status_label.configure(text=text)

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not load image.")
            return
        if img.shape[0] * img.shape[1] > MAX_IMAGE_PIXELS:
            messagebox.showerror("Error", "Image is too large to process safely.")
            return
        self.img = img

        tk_img = cv2_to_tk(self.img)
        self.original_label.configure(image=tk_img, text="")
        self.original_label.image = tk_img  # keep reference
        self.colorized_label.configure(image="", text="Colorized")
        self.status_label.configure(text="")

    def run_colorize(self):
        if not self.model_ready:
            messagebox.showwarning("Model not ready", "The colorization model is still loading.")
            return
        if self.img is None:
            messagebox.showwarning("No image", "Please open an image first.")
            return
        try:
            self.colorized = colorize_image(self.img)
        except Exception as e:
            logger.exception("Colorization failed")
            messagebox.showerror("Error", f"Colorization failed: {e}")
            return
        tk_img = cv2_to_tk(self.colorized)
        self.colorized_label.configure(image=tk_img, text="")
        self.colorized_label.image = tk_img  # keep reference

    def save_result(self):
        if self.colorized is None:
            messagebox.showwarning("Nothing to save", "Colorize an image first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if path:
            cv2.imwrite(path, self.colorized)
            messagebox.showinfo("Saved", f"Saved to {path}")

    def batch_colorize(self):
        if not self.model_ready:
            messagebox.showwarning("Model not ready", "The colorization model is still loading.")
            return
        input_dir = filedialog.askdirectory(title="Select folder with images to colorize")
        if not input_dir:
            return

        files = [f for f in os.listdir(input_dir) if f.lower().endswith(VALID_EXTENSIONS)]
        if not files:
            messagebox.showwarning("No images found", "No supported image files in that folder.")
            return

        self.set_controls_enabled(False)
        self.set_status("Starting batch processing...")

        worker = threading.Thread(target=self._batch_colorize_worker, args=(input_dir, files), daemon=True)
        worker.start()

    def _batch_colorize_worker(self, input_dir, files):
        output_dir = os.path.join(input_dir, "colorized_output")
        os.makedirs(output_dir, exist_ok=True)

        total = len(files)
        succeeded = 0
        failed = []
        error_text = None

        try:
            for i, filename in enumerate(files, start=1):
                self.root.after(0, lambda i=i, filename=filename: self.set_status(f"Processing {i}/{total}: {filename}"))

                in_path = os.path.join(input_dir, filename)
                img = cv2.imread(in_path)
                if img is None or img.shape[0] * img.shape[1] > MAX_IMAGE_PIXELS:
                    failed.append(filename)
                    continue

                try:
                    result = colorize_image(img)
                    out_name = os.path.splitext(filename)[0] + "_colorized.png"
                    out_path = os.path.join(output_dir, out_name)
                    cv2.imwrite(out_path, result)
                    succeeded += 1
                except Exception:
                    logger.exception("Failed to colorize %s", filename)
                    failed.append(filename)
        except Exception as exc:
            logger.exception("Batch colorization stopped early")
            error_text = str(exc)
        finally:
            summary = f"Colorized {succeeded}/{total} images.\nSaved to:\n{output_dir}"
            if failed:
                summary += f"\n\nFailed ({len(failed)}): {', '.join(failed)}"
            if error_text:
                summary += f"\n\nBatch stopped early: {error_text}"

            def finish():
                self.set_status(f"Done: {succeeded}/{total} colorized. Saved to {output_dir}")
                self.set_controls_enabled(True)
                messagebox.showinfo("Batch Complete", summary)

            self.root.after(0, finish)


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorizerApp(root)
    root.mainloop()
