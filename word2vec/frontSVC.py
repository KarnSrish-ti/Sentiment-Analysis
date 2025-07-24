import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import requests
from io import BytesIO
import joblib
import os
import numpy as np
from gensim.models import Word2Vec

class SentimentAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("News Sentiment Detector")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        
        self.model = None
        self.w2v_model = None  # Changed from vectorizer to w2v_model for clarity
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
        """Load both the SVM classifier and Word2Vec model"""
        if not all(os.path.exists(f) for f in ["svm_classifier.joblib", "word2vec.model"]):
            raise FileNotFoundError("Required model files not found in the current directory")
        
        # Load SVM classifier
        self.model = joblib.load("svm_classifier.joblib")
        
        # Load Word2Vec model (using gensim's load function)
        self.w2v_model = Word2Vec.load("word2vec.model")
        
        # Load background image
        try:
            bg_url = "https://i.pinimg.com/originals/90/88/8e/90888e294b6a07e4deac94f9df0a796b.jpg"
            response = requests.get(bg_url, timeout=10)
            response.raise_for_status()
            bg_image = Image.open(BytesIO(response.content))
            bg_image = bg_image.resize((600, 400), Image.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
        except requests.exceptions.RequestException:
            self.bg_photo = ImageTk.PhotoImage(Image.new('RGB', (600, 400), '#f0f0f0'))
    
    def document_vector(self, tokens):
        """Convert tokens to document vector by averaging word vectors"""
        # Filter words that exist in the vocabulary
        words = [word for word in tokens if word in self.w2v_model.wv.key_to_index]
        
        if len(words) == 0:
            return np.zeros(self.w2v_model.vector_size)
        
        # Average the word vectors
        return np.mean([self.w2v_model.wv[word] for word in words], axis=0)
    
    def analyze(self):
    #Analyze the sentiment of the entered text
        if not self.resources_loaded:
            messagebox.showerror("Error", "Resources not loaded properly")
            return
            
        headline = self.entry.get().strip()
        
        if not headline:
            messagebox.showwarning("Input Error", "Please enter a news headline")
            return
        
        try:
            self.status_bar.config(text="Analyzing...")
            self.root.update_idletasks()  # Force UI update
            
            # Tokenize the input
            tokens = headline.lower().split()
            
            # Convert to document vector
            vector = np.array([self.document_vector(tokens)])
            
            # Predict sentiment - ensure we get the first prediction as string
            prediction = str(self.model.predict(vector)[0])
            
            # Format result
            sentiment_map = {
                "positive": ("Positive 😊", "#4CAF50"),
                "negative": ("Negative 😞", "#F44336"),
                
                "0": ("Negative 😞", "#F44336"),  # Handle numeric labels
                "1": ("Positive 😊", "#4CAF50")
            }
            
            # Use lowercase prediction for mapping
            result_text, color = sentiment_map.get(
                prediction.lower(), 
                (f"Prediction: {prediction}", "#2196F3")
            )
            
            self.result_label.config(
                text=f"Sentiment: {result_text}",
                fg=color
            )
            
            self.status_bar.config(text="Analysis complete")
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze: {str(e)}")
            self.status_bar.config(text="Error occurred")
    
    def create_widgets(self):
        """Create all GUI widgets"""
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
            text="Ready" if self.resources_loaded else "Models not loaded",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Helvetica", 10)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def run(self):
        if self.resources_loaded:
            self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentAnalyzerApp(root)
    app.run()