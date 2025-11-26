import cv2
import mediapipe as mp
import pyautogui
import webbrowser
import datetime
import math
import numpy as np
import os
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ==================== HAND FEATURE & GESTURE HELPERS ====================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def count_fingers(landmarks):
    """Count raised fingers using MediaPipe landmarks."""
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    fingers = []

    # Thumb (x-axis comparison)
    if landmarks[thumb_tip].x < landmarks[thumb_tip - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers (y-axis comparison)
    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)


def extract_hand_feature(landmarks):
    """
    Generate a normalized feature vector from 21 hand landmarks.
    Normalized relative to the wrist and hand size.
    """
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    wrist_x, wrist_y = xs[0], ys[0]

    # Hand size: distance between wrist and index fingertip
    dx = xs[8] - wrist_x
    dy = ys[8] - wrist_y
    hand_size = math.sqrt(dx * dx + dy * dy) + 1e-6

    feat = []
    for x, y in zip(xs, ys):
        feat.append((x - wrist_x) / hand_size)
        feat.append((y - wrist_y) / hand_size)

    return np.array(feat, dtype=np.float32)


# ==================== APPLICATION CLASS (GUI + LOGIC) ====================

class HandGestureLockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Lock & Control")
        self.root.configure(bg="#020617")  # very dark background

        # -------- STATE VARIABLES --------
        self.command_cooldown = 0
        self.COOLDOWN_TIME = 30  # ~1 second

        self.prev_x = None
        self.move_counter = 0
        self.SWIPE_THRESHOLD = 50   # daha hassas swipe
        self.SWIPE_FRAMES = 2       # 2 frame üst üste

        self.TEMPLATE_PATH = "hand_template.npy"
        self.FEATURE_THRESHOLD = 0.12
        self.is_locked = True          # app starts locked
        self.hand_template = None
        self.last_feature = None

        # Animation state
        self.scan_phase = 0            # for scanning pulse when locked
        self.save_anim_frames = 0      # after saving template
        self.unlock_anim_frames = 0    # short animation when unlock happens
        self.photo_anim_frames = 0     # visual feedback after photo/screenshot

        # Photo / screenshot system
        self.photo_dir = "photos"
        self.screenshot_dir = "screenshots"
        os.makedirs(self.photo_dir, exist_ok=True)
        os.makedirs(self.screenshot_dir, exist_ok=True)

        self.photo_countdown_active = False
        self.photo_countdown_end_time = 0.0

        # -------- MEDIAPIPE --------
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7
        )

        # -------- CAMERA --------
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            raise RuntimeError("Camera could not be opened.")

        # Load saved hand template (if exists)
        if os.path.exists(self.TEMPLATE_PATH):
            try:
                self.hand_template = np.load(self.TEMPLATE_PATH)
                print("[+] Loaded saved hand template.")
            except Exception as e:
                print("[!] Failed to load hand template:", e)

        # -------- STYLES --------
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.style.configure(
            "Accent.TButton",
            background="#38bdf8",   # sky-400
            foreground="#0f172a",   # slate-900
            borderwidth=0,
            focusthickness=0,
            padding=(14, 6),
        )
        self.style.map(
            "Accent.TButton",
            background=[
                ("active", "#0ea5e9"),   # sky-500
                ("pressed", "#0284c7"),  # sky-600
            ]
        )

        self.style.configure(
            "Ghost.TButton",
            background="#0f172a",
            foreground="#e5e7eb",
            borderwidth=1,
            focusthickness=0,
            padding=(14, 6),
        )
        self.style.map(
            "Ghost.TButton",
            background=[
                ("active", "#111827"),
                ("pressed", "#020617"),
            ]
        )

        # -------- LAYOUT --------
        main_frame = tk.Frame(self.root, bg="#020617")
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Left: video
        video_card = tk.Frame(main_frame, bg="#020617")
        video_card.grid(row=0, column=0, padx=16, pady=16)

        video_container = tk.Frame(video_card, bg="#0b1120", bd=0,
                                   highlightthickness=1, highlightbackground="#1e293b")
        video_container.pack()
        self.video_label = tk.Label(video_container, bg="#0b1120")
        self.video_label.pack(padx=4, pady=4)

        # Right: side panel
        side_frame = tk.Frame(main_frame, bg="#020617")
        side_frame.grid(row=0, column=1, sticky="n", padx=(0, 16), pady=16)

        title = tk.Label(
            side_frame,
            text="Hand Gesture Lock & Control",
            font=("Segoe UI", 16, "bold"),
            fg="#e5e7eb",
            bg="#020617"
        )
        title.pack(anchor="w", pady=(0, 10))

        subtitle = tk.Label(
            side_frame,
            text="Unlock and control your computer using your hand.",
            font=("Segoe UI", 10),
            fg="#9ca3af",
            bg="#020617"
        )
        subtitle.pack(anchor="w", pady=(0, 12))

        status_card = tk.Frame(side_frame, bg="#0b1120", bd=0,
                               highlightthickness=1, highlightbackground="#1e293b")
        status_card.pack(anchor="w", fill="x", pady=(0, 10))

        self.lock_label = tk.Label(
            status_card,
            text="Lock: LOCKED",
            font=("Segoe UI", 12, "bold"),
            fg="#f97373",
            bg="#0b1120"
        )
        self.lock_label.pack(anchor="w", padx=10, pady=(8, 0))

        self.status_label = tk.Label(
            status_card,
            text="Status: Ready",
            font=("Segoe UI", 10),
            fg="#e5e7eb",
            bg="#0b1120",
            wraplength=260,
            justify="left"
        )
        self.status_label.pack(anchor="w", padx=10, pady=(4, 10))

        # Buttons
        btn_frame = tk.Frame(side_frame, bg="#020617")
        btn_frame.pack(anchor="w", pady=(4, 8))

        self.save_btn = ttk.Button(
            btn_frame,
            text="Save Hand",
            style="Accent.TButton",
            command=self.save_hand_template
        )
        self.save_btn.grid(row=0, column=0, padx=(0, 8), pady=4)

        self.gallery_btn = ttk.Button(
            btn_frame,
            text="Photos",
            style="Ghost.TButton",
            command=self.open_gallery
        )
        self.gallery_btn.grid(row=0, column=1, padx=(0, 8), pady=4)

        self.quit_btn = ttk.Button(
            btn_frame,
            text="Quit",
            style="Ghost.TButton",
            command=self.on_close
        )
        self.quit_btn.grid(row=0, column=2, padx=(0, 0), pady=4)

        # Hover effect for buttons
        for btn in (self.save_btn, self.gallery_btn, self.quit_btn):
            btn.bind("<Enter>", self._on_button_enter)
            btn.bind("<Leave>", self._on_button_leave)
            btn.bind("<ButtonPress-1>", self._on_button_press)
            btn.bind("<ButtonRelease-1>", self._on_button_release)

        instructions = (
            "Workflow:\n"
            "• Show your hand and click 'Save Hand' to enroll.\n"
            "• When your hand is recognized, the lock unlocks automatically.\n\n"
            "Gestures (when unlocked):\n"
            "• 1 finger  → Open YouTube\n"
            "• 2 fingers → 3s countdown → Take a photo\n"
            "• 3 fingers → Copy (Ctrl + C)\n"
            "• 4 fingers → Paste (Ctrl + V)\n"
            "• Swipe right → left → Screenshot (saved)\n\n"
            "Shortcuts:\n"
            "• Q → Quit\n"
            "• E → Save hand"
        )
        instr_label = tk.Label(
            side_frame,
            text=instructions,
            font=("Segoe UI", 9),
            fg="#9ca3af",
            bg="#020617",
            justify="left",
            wraplength=280
        )
        instr_label.pack(anchor="w", pady=(10, 0))

        # Window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Keyboard shortcuts
        self.root.bind("<KeyPress-e>", lambda e: self.save_hand_template())
        self.root.bind("<KeyPress-q>", lambda e: self.on_close())
        # Optional dev shortcut: re-lock
        self.root.bind("<KeyPress-l>", lambda e: self._dev_relock())

        # Start update loop
        self.update_frame()

    # ------------------ BUTTON ANIMATIONS ------------------

    def _on_button_enter(self, event):
        btn = event.widget
        btn.configure(cursor="hand2")

    def _on_button_leave(self, event):
        btn = event.widget
        btn.configure(cursor="")

    def _on_button_press(self, event):
        btn = event.widget
        style_name = btn.cget("style") or "TButton"
        self.style.configure(style_name, padding=(12, 5))

    def _on_button_release(self, event):
        btn = event.widget
        style_name = btn.cget("style") or "TButton"
        self.style.configure(style_name, padding=(14, 6))

    # ------------------ LOGIC METHODS ------------------

    def save_hand_template(self):
        if self.last_feature is not None:
            self.hand_template = self.last_feature
            np.save(self.TEMPLATE_PATH, self.hand_template)
            self.set_status("Hand template saved.", important=True)
            self.save_anim_frames = 20  # yellow highlight animation
            print("[+] Hand template saved.")
        else:
            self.set_status("No hand detected. Cannot save template.", important=True)
            print("[!] No hand detected. Cannot save template.")

    def _dev_relock(self):
        """Optional dev shortcut: re-lock without a visible button."""
        self.is_locked = True
        self.update_lock_label()
        self.set_status("Lock enabled (dev shortcut).", important=True)

    def update_lock_label(self):
        if self.is_locked:
            self.lock_label.config(text="Lock: LOCKED", fg="#f97373")  # red-ish
        else:
            self.lock_label.config(text="Lock: UNLOCKED", fg="#4ade80")  # green

    def set_status(self, text, important=False):
        self.status_label.config(text=f"Status: {text}")
        if important:
            self.status_label.config(fg="#facc15")  # amber-400
        else:
            self.status_label.config(fg="#e5e7eb")

    # ------------------ GALLERY WINDOW ------------------

    def open_gallery(self):
        files = [
            f for f in os.listdir(self.photo_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        files = sorted(
            files,
            key=lambda f: os.path.getmtime(os.path.join(self.photo_dir, f)),
            reverse=True
        )

        win = tk.Toplevel(self.root)
        win.title("Captured Photos")
        win.configure(bg="#020617")

        if not files:
            msg = tk.Label(
                win,
                text="No photos captured yet.",
                font=("Segoe UI", 10),
                fg="#e5e7eb",
                bg="#020617"
            )
            msg.pack(padx=20, pady=20)
            return

        canvas = tk.Canvas(win, bg="#020617", highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)

        inner = tk.Frame(canvas, bg="#020617")
        canvas.create_window((0, 0), window=inner, anchor="nw")

        thumbs = []
        max_per_row = 3
        thumb_size = (180, 120)

        for idx, fname in enumerate(files):
            path = os.path.join(self.photo_dir, fname)
            try:
                img = Image.open(path)
                img.thumbnail(thumb_size)
                imgtk = ImageTk.PhotoImage(img)
                thumbs.append(imgtk)
                row = idx // max_per_row
                col = idx % max_per_row
                lbl = tk.Label(inner, image=imgtk, bg="#020617")
                lbl.grid(row=row, column=col, padx=8, pady=8)
            except Exception as e:
                print("Error loading image:", path, e)

        inner.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        win.geometry("600x400")

        # Keep reference to prevent GC
        win.thumbs = thumbs

    # ------------------ MAIN FRAME UPDATE ------------------

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.set_status("Failed to read from camera.", important=True)
            self.root.after(30, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_for_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_for_mp)

        # Base ROI rectangle
        cv2.rectangle(frame, (100, 100), (540, 380), (30, 64, 175), 1)

        status_text = "Ready"
        hand_center = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            cx = int(landmarks[9].x * w)
            cy = int(landmarks[9].y * h)
            hand_center = (cx, cy)

            # Hand feature
            current_feature = extract_hand_feature(landmarks)
            self.last_feature = current_feature

            # ---- LOCK MODE ----
            if self.is_locked:
                if self.hand_template is not None:
                    dist = np.linalg.norm(current_feature - self.hand_template)
                    status_text = f"Hand distance: {dist:.3f}"
                    # scanning pulse animation
                    self.scan_phase = (self.scan_phase + 1) % 40
                    radius = 40 + int(10 * math.sin(self.scan_phase * math.pi / 20))
                    cv2.circle(frame, hand_center, radius, (56, 189, 248), 2)  # sky-400
                    if dist < self.FEATURE_THRESHOLD:
                        status_text = "Hand recognized → Lock unlocked"
                        self.is_locked = False
                        self.update_lock_label()
                        self.unlock_anim_frames = 20  # green animation
                else:
                    status_text = "No enrolled hand. Click 'Save Hand'."
                    # subtle pulsing to indicate "waiting"
                    self.scan_phase = (self.scan_phase + 1) % 40
                    base_radius = 30 + int(6 * math.sin(self.scan_phase * math.pi / 20))
                    if hand_center:
                        cv2.circle(frame, hand_center, base_radius, (148, 163, 184), 1)  # gray-ish
            else:
                # ---- UNLOCKED: GESTURE COMMANDS ----

                # Swipe: right to left -> screenshot (saved to file)
                if self.prev_x is not None:
                    move_distance = self.prev_x - cx
                    if move_distance > self.SWIPE_THRESHOLD:
                        self.move_counter += 1
                    else:
                        self.move_counter = 0

                    if (self.move_counter >= self.SWIPE_FRAMES and
                            self.command_cooldown == 0 and
                            not self.photo_countdown_active):
                        filename = os.path.join(
                            self.screenshot_dir,
                            f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        )
                        screenshot = pyautogui.screenshot()
                        screenshot.save(filename)
                        print(f"[+] Screenshot saved: {filename}")
                        status_text = "Swipe left → Screenshot saved"
                        self.photo_anim_frames = 12  # use same flash animation
                        self.command_cooldown = self.COOLDOWN_TIME
                        self.move_counter = 0
                self.prev_x = cx

                # Finger count commands
                total_fingers = count_fingers(landmarks)
                cv2.putText(frame, f"Fingers: {total_fingers}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (248, 250, 252), 2)

                # Start 3-second countdown for photo when 2 fingers shown
                if self.command_cooldown == 0 and not self.photo_countdown_active:
                    if total_fingers == 2:
                        self.photo_countdown_active = True
                        self.photo_countdown_end_time = time.time() + 3
                        status_text = "2 fingers → Photo in 3 seconds..."
                    elif total_fingers == 1:
                        status_text = "1 finger → Opening YouTube"
                        webbrowser.open("https://www.youtube.com")
                        self.command_cooldown = self.COOLDOWN_TIME
                    elif total_fingers == 3:
                        status_text = "3 fingers → Copy (Ctrl+C)"
                        pyautogui.hotkey('ctrl', 'c')
                        self.command_cooldown = self.COOLDOWN_TIME
                    elif total_fingers == 4:
                        status_text = "4 fingers → Paste (Ctrl+V)"
                        pyautogui.hotkey('ctrl', 'v')
                        self.command_cooldown = self.COOLDOWN_TIME

        else:
            self.prev_x = None
            self.move_counter = 0
            status_text = "No hand detected"

        # Handle photo countdown (3 seconds)
        if self.photo_countdown_active:
            remaining = self.photo_countdown_end_time - time.time()
            if remaining > 0:
                sec = int(math.ceil(remaining))
                # big countdown in center
                cv2.putText(frame, str(sec),
                            (int(w / 2) - 30, int(h / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            3.0,
                            (250, 204, 21),
                            6)
                status_text = f"Taking photo in {sec}..."
            else:
                # Time is up: capture photo
                filename = os.path.join(
                    self.photo_dir,
                    f"photo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                cv2.imwrite(filename, frame)
                print(f"[+] Photo saved: {filename}")
                self.photo_anim_frames = 12  # trigger flash animation
                self.command_cooldown = self.COOLDOWN_TIME
                self.photo_countdown_active = False
                status_text = "Photo captured."

        # Cooldown decrease
        if self.command_cooldown > 0:
            self.command_cooldown -= 1

        # ---- SHORT ANIMATIONS (SAVE / UNLOCK / PHOTO / SCREENSHOT) ----
        if hand_center:
            # Save animation: yellow ring
            if self.save_anim_frames > 0:
                radius = 50 + (self.save_anim_frames * 2)
                cv2.circle(frame, hand_center, radius, (250, 204, 21), 2)  # amber-400
                self.save_anim_frames -= 1

            # Unlock animation: green glow
            if self.unlock_anim_frames > 0:
                radius = 60 + (20 - self.unlock_anim_frames)
                cv2.circle(frame, hand_center, radius, (74, 222, 128), 3)  # green-400
                self.unlock_anim_frames -= 1

        # Photo / screenshot flash animation (screen border)
        if self.photo_anim_frames > 0:
            thickness = 4 + self.photo_anim_frames
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 255, 255), thickness)
            self.photo_anim_frames -= 1

        # Update status text
        self.set_status(status_text)

        # Convert BGR frame to RGB for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Schedule next frame
        self.root.after(30, self.update_frame)

    # ------------------ CLOSE APP ------------------

    def on_close(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


# ==================== MAIN ====================

if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureLockApp(root)
    root.mainloop()
