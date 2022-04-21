#!/usr/bin/env python3

import tkinter
from PIL import Image
from PIL import ImageTk
from pathlib import Path
import cv2


class COCOApp:
    def __init__(self) -> None:
        self.root = tkinter.Tk()
        self.images = None

    def make_ui(self):
        self.root.title("COCO Viewer")
        self.root.geometry("800x600+100+100")
        self.root.resizable(False, False)

        # frames
        self.dir_entry_frame = tkinter.Frame(self.root, bd=2)
        self.img_frame = tkinter.Frame(self.root, bd=2)
        self.img_list_frame = tkinter.Frame(self.root, bd=2)

        self.dir_entry_frame.pack(side="top", fill="both", expand=False)
        self.img_frame.pack(side="left", fill="both", expand=True)
        self.img_list_frame.pack(side="right", fill="both", expand=True)

        # dir entry
        self.dir_entry = tkinter.Entry(self.dir_entry_frame)
        self.dir_entry.bind("<Return>", self.set_image_dir)
        self.dir_entry.pack(side="left", fill="both")

        # side='left',fill='y'

        # listbox
        scrollbar = tkinter.Scrollbar(self.img_list_frame)
        scrollbar.pack(side="right", fill="y")
        self.file_listbox = tkinter.Listbox(
            self.img_list_frame, yscrollcommand=scrollbar.set
        )
        self.file_listbox.pack(side="left", expand=True)
        scrollbar["command"] = self.file_listbox.yview

    def set_image_dir(self, event):
        path = Path(self.dir_entry.get()).resolve().absolute()
        self.file_listbox.delete(0, self.file_listbox.size() - 1)
        for i, f in enumerate(path.iterdir()):
            self.file_listbox.insert(i, f)

    def show_image(self):
        src = cv2.imread("giraffe.jpg", cv2.IMREAD_UNCHANGED)
        src = cv2.resize(src, (800, 600))
        img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        label = tkinter.Label(self.root, image=imgtk)
        label.pack(side="left")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = COCOApp()
    app.make_ui()
    app.run()
