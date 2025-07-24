import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import requests
from io import BytesIO
import joblib
import os

class SentimentAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("News Sentiment Detector")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        
        self.model = None
        self.vectorizer = None
        self.bg_photo = None
        self.resources_loaded = False
        
        try:
            self.load_resources()
            self.resources_loaded = True
            self.create_widgets()
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize application: {str(e)}")
            self.root.destroy()
    
    def load_resources(self):
        if not all(os.path.exists(f) for f in ["svm_model.joblib", "tfidf_vectorizer.joblib"]):
            raise FileNotFoundError("Required model files not found in the current directory")
        
        self.model = joblib.load("svm_model.joblib")
        self.vectorizer = joblib.load("tfidf_vectorizer.joblib")
        
        try:
            bg_url = "https://i.pinimg.com/originals/90/88/8e/90888e294b6a07e4deac94f9df0a796b.jpg"
            response = requests.get(bg_url, timeout=10)
            response.raise_for_status()
            bg_image = Image.open(BytesIO(response.content))
            bg_image = bg_image.resize((600, 400), Image.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
        except requests.exceptions.RequestException:
            self.bg_photo = ImageTk.PhotoImage(Image.new('RGB', (600, 400), '#f0f0f0'))  # fallback solid background
    
    def create_widgets(self):
        self.background = tk.Label(self.root, image=self.bg_photo)
        self.background.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.main_frame = tk.Frame(self.root, bg="#ffffff", bd=2, relief=tk.RAISED)
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center", width=500, height=300)
        
        self.title_label = tk.Label(
            self.main_frame,
            text="NEWS SENTIMENT ANALYZER",
            font=("Helvetica", 16, "bold"),
            bg="#ffffff",
            fg="#333333"
        )
        self.title_label.pack(pady=(20, 10))
        
        input_frame = tk.Frame(self.main_frame, bg="#ffffff")
        input_frame.pack(pady=10)
        
        self.entry = tk.Entry(
            input_frame,
            font=("Helvetica", 12),
            width=40,
            bd=2,
            relief=tk.SUNKEN
        )
        self.entry.pack(padx=10, pady=5)
        self.entry.bind("<Return>", lambda event: self.analyze())
        
        self.analyze_btn = tk.Button(
            input_frame,
            text="Analyze Sentiment",
            command=self.analyze,
            font=("Helvetica", 12),
            bg="#4a6baf",
            fg="white",
            activebackground="#3a5a9f",
            relief=tk.RAISED
        )
        self.analyze_btn.pack(pady=10)
        
        self.result_label = tk.Label(
            self.main_frame,
            text="",
            font=("Helvetica", 14),
            bg="#ffffff",
            fg="#333333",
            wraplength=450
        )
        self.result_label.pack(pady=10)
        
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Helvetica", 10)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def analyze(self):
        if not self.resources_loaded:
            messagebox.showerror("Error", "Resources not loaded properly")
            return
        
        headline = self.entry.get().strip()
        if not headline:
            messagebox.showwarning("Input Error", "Please enter a news headline")
            return
        
        try:
            self.status_bar.config(text="Analyzing...")
            self.root.update_idletasks()
            
            X = self.vectorizer.transform([headline])
            prediction = self.model.predict(X)[0]
            
            # If prediction is numeric
            if isinstance(prediction, int):
                prediction = "positive" if prediction == 1 else "negative"
            
            prediction_str = str(prediction).lower()
            
            sentiment_map = {
                "positive": ("Positive 😊", "#4CAF50"),
                "negative": ("Negative 😞", "#F44336")
            }
            
            result_text, color = sentiment_map.get(
                prediction_str,
                (f"Prediction: {prediction_str}", "#2196F3")
            )
            
            self.result_label.config(
                text=f"Sentiment: {result_text}",
                fg=color
            )
            
            self.status_bar.config(text="Analysis complete")
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze: {str(e)}")
            self.status_bar.config(text="Error occurred")
    
    def run(self):
        if self.resources_loaded:
            self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalyzerApp(root)
    app.run()
