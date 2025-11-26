# Hand Gesture Lock & Control

A desktop application that uses **hand gestures** as a kind of "visual keyboard" to unlock and control your computer.

The app uses **computer vision + hand landmarks + simple gesture rules** to:
- Unlock when your enrolled hand is recognized
- Control common actions with intuitive finger gestures

---

## ✨ Features

- 🔐 **Hand-based lock**
  - You enroll your own hand once (“Save Hand”).
  - When your hand is detected and matched, the UI changes from **LOCKED** to **UNLOCKED**.
  - All gesture commands only work when the app is unlocked.

- 🖐️ **Gesture controls (when unlocked)**
  - **1 finger** → Open YouTube in the default browser  
  - **2 fingers** → Start a **3-second countdown**, then capture a photo  
  - **3 fingers** → Copy (`Ctrl + C`)  
  - **4 fingers** → Paste (`Ctrl + V`)  
  - **Swipe from right to left** → Take a **screenshot**, saved as a file

- 📸 **Photo capture**
  - Shows a big **3, 2, 1** countdown over the camera feed.
  - Saves a clean frame (without overlays / rectangles) into the `photos/` folder.
  - A white border flash animation is shown as visual feedback.

- 🖼️ **Photo gallery**
  - A **Photos** button in the UI opens a simple gallery window.
  - All images in `photos/` are shown as thumbnails in a scrollable view.

- 💻 **Modern dark UI**
  - Built with Tkinter and custom styles.
  - Status panel, lock state, and live camera preview.
  - Button hover & click effects for a more “product-like” feel.

---

## 🧰 Tech Stack

- **Language:** Python 3
- **UI:** Tkinter
- **Computer Vision:** OpenCV
- **Hand Tracking:** MediaPipe Hands
- **Automation:** PyAutoGUI
- **Image Handling:** Pillow (PIL)

---

## 📦 Installation

1. Clone or download this repository:

```bash
git clone https://github.com/<your-username>/hand-gesture-lock.git
cd hand-gesture-lock
