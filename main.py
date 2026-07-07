import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from colorizer_core import colorize_image, load_model

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# ---------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------
print("Loading model...")
load_model()
print("✅ Model loaded!")


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

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Open Image", command=self.open_image, width=15).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Colorize", command=self.run_colorize, width=15).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Save Result", command=self.save_result, width=15).pack(side="left", padx=5)

        batch_frame = tk.Frame(root)
        batch_frame.pack(pady=(0, 10))
        tk.Button(batch_frame, text="Batch Colorize Folder", command=self.batch_colorize, width=25).pack()

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

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not path:
            return
        self.img = cv2.imread(path)
        if self.img is None:
            messagebox.showerror("Error", "Could not load image.")
            return

        tk_img = cv2_to_tk(self.img)
        self.original_label.configure(image=tk_img, text="")
        self.original_label.image = tk_img  # keep reference
        self.colorized_label.configure(image="", text="Colorized")
        self.status_label.configure(text="")

    def run_colorize(self):
        if self.img is None:
            messagebox.showwarning("No image", "Please open an image first.")
            return
        try:
            self.colorized = colorize_image(self.img)
        except Exception as e:
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
        input_dir = filedialog.askdirectory(title="Select folder with images to colorize")
        if not input_dir:
            return

        files = [f for f in os.listdir(input_dir) if f.lower().endswith(VALID_EXTENSIONS)]
        if not files:
            messagebox.showwarning("No images found", "No supported image files in that folder.")
            return

        output_dir = os.path.join(input_dir, "colorized_output")
        os.makedirs(output_dir, exist_ok=True)

        total = len(files)
        succeeded = 0
        failed = []

        for i, filename in enumerate(files, start=1):
            self.status_label.configure(text=f"Processing {i}/{total}: {filename}")
            self.root.update_idletasks()

            in_path = os.path.join(input_dir, filename)
            img = cv2.imread(in_path)
            if img is None:
                failed.append(filename)
                continue

            try:
                result = colorize_image(img)
                out_name = os.path.splitext(filename)[0] + "_colorized.png"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, result)
                succeeded += 1
            except Exception:
                failed.append(filename)

        self.status_label.configure(text=f"Done: {succeeded}/{total} colorized. Saved to {output_dir}")

        summary = f"Colorized {succeeded}/{total} images.\nSaved to:\n{output_dir}"
        if failed:
            summary += f"\n\nFailed ({len(failed)}): {', '.join(failed)}"
        messagebox.showinfo("Batch Complete", summary)


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorizerApp(root)
    root.mainloop()
