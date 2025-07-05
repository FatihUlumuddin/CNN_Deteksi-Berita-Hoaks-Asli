import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# --- Load dataset dan tokenizer ulang ---
df = pd.read_csv("data_berita.csv")
texts = df['teks'].astype(str).tolist()
labels = df['label'].tolist()

# Label encoding
le = LabelEncoder()
y = le.fit_transform(labels)

# Tokenizer
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Load model CNN
model = load_model("cnn_hoaks_asli_rpi.h5")

# --- Fungsi prediksi ---
def prediksi_berita():
    teks = entry_teks.get("1.0", tk.END).strip()
    if not teks:
        messagebox.showwarning("Kosong", "Masukkan teks berita terlebih dahulu.")
        return
    seq = tokenizer.texts_to_sequences([teks])
    pad_seq = pad_sequences(seq, maxlen=50)
    pred = model.predict(pad_seq)[0][0]
    hasil = "ASLI" if pred > 0.5 else "HOAKS"
    label_hasil.config(text=f"Hasil: {hasil} ({pred:.2f})", fg="green" if pred > 0.5 else "red")

# --- GUI Tkinter ---
root = tk.Tk()
root.title("Deteksi Berita Hoaks")
root.geometry("400x300")
root.configure(bg="#f5f5f5")

judul = tk.Label(root, text="Deteksi Berita Hoaks / Asli", font=("Helvetica", 14, "bold"), bg="#f5f5f5")
judul.pack(pady=10)

entry_teks = tk.Text(root, height=5, width=40, font=("Arial", 10))
entry_teks.pack(pady=10)

btn_prediksi = tk.Button(root, text="Prediksi", command=prediksi_berita, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
btn_prediksi.pack(pady=10)

label_hasil = tk.Label(root, text="", font=("Arial", 12, "bold"), bg="#f5f5f5")
label_hasil.pack(pady=10)

root.mainloop()
