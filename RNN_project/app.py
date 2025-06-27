import torch
import torch.nn as nn
import numpy as np
import tkinter as tk
from tkinter import ttk # Daha modern görünümlü widget'lar için

# ==============================================================================
# BÖLÜM 1: MODEL VE VERİ HAZIRLIĞI (Değişiklik yok)
# ==============================================================================

# Veri
sentence = "kedi koltuğun üzerine zıpladı"
tokens = sentence.split(' ')
word_list = sorted(list(set(tokens)))
word_to_ix = {word: i for i, word in enumerate(word_list)}
ix_to_word = {i: word for i, word in enumerate(word_list)}
vocab_size = len(word_list)

# Girdi ve Hedefler
inputs = []
targets = []
for i in range(len(tokens) - 1):
    inputs.append(word_to_ix[tokens[i]])
    targets.append(word_to_ix[tokens[i+1]])

inputs = torch.LongTensor(inputs)
targets = torch.LongTensor(targets)

# RNN Model Mimarisi
class WordRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(WordRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        x = self.embedding(x)
        outputs, hidden_state = self.rnn(x, hidden_state)
        outputs = self.fc(outputs)
        return outputs, hidden_state

# Model parametreleri ve modelin oluşturulması
embedding_dim = 10
hidden_size = 12
model = WordRNN(vocab_size, embedding_dim, hidden_size)

# Kayıp fonksiyonu ve optimize edici
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# ==============================================================================
# BÖLÜM 2: MODELİN EĞİTİLMESİ (Değişiklik yok)
# ==============================================================================
# NOT: Bu basit örnekte model her çalıştığında yeniden eğitilir.
# Gerçek bir uygulamada, model bir kez eğitilir ve ağırlıkları kaydedilip yüklenir.
print("Uygulama başlatılıyor, model eğitiliyor...")
epochs = 500
for epoch in range(epochs):
    model_inputs = inputs.unsqueeze(0)
    outputs, _ = model(model_inputs)
    loss = criterion(outputs.view(-1, vocab_size), targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("Model eğitimi tamamlandı. Arayüz hazır.")


# ==============================================================================
# BÖLÜM 3: TAHMİN FONKSİYONU (Değişiklik yok)
# ==============================================================================

def generate_sentence_from_word(start_word):
    """
    Kullanıcıdan bir başlangıç kelimesi alır ve eğitilmiş modeli kullanarak
    cümlenin geri kalanını tahmin eder.
    """
    if start_word not in word_to_ix:
        valid_words = ", ".join(word_list)
        return f"Hata: Geçerli bir kelime girin. Geçerli kelimeler: {valid_words}"

    model.eval()
    generated_sentence = [start_word]
    current_input = torch.LongTensor([[word_to_ix[start_word]]])
    hidden_state = None

    for _ in range(len(tokens) - 1):
        with torch.no_grad():
            output, hidden_state = model(current_input, hidden_state)
            _, predicted_ix = torch.max(output, 2)
            current_input = predicted_ix
            predicted_word = ix_to_word[predicted_ix.item()]
            if predicted_word in generated_sentence:
                break
            generated_sentence.append(predicted_word)
            
    return ' '.join(generated_sentence)


# ==============================================================================
# BÖLÜM 4: TKINTER MASAÜSTÜ ARAYÜZÜNÜN OLUŞTURULMASI
# ==============================================================================

# Butona tıklandığında çalışacak fonksiyon
def on_predict_click():
    # 1. Kullanıcının girdiği kelimeyi al
    start_word = entry_word.get()
    
    # 2. Tahmin fonksiyonunu çağır
    if start_word: # Girdi kutusunun boş olmadığından emin ol
        result = generate_sentence_from_word(start_word.lower().strip())
        
        # 3. Sonucu arayüzdeki etikete yazdır
        result_label_var.set(f"Sonuç: {result}")
    else:
        result_label_var.set("Lütfen bir kelime girin.")

# --- Arayüzü Oluşturma ---
# Ana pencereyi oluştur
root = tk.Tk()
root.title("Kelime Tahmin Eden RNN")
root.geometry("500x300") # Pencere boyutunu ayarla

# Pencere içeriğini ortalamak için bir çerçeve (frame) oluştur
main_frame = ttk.Frame(root, padding="20")
main_frame.pack(expand=True)

# Açıklama etiketi
intro_label = ttk.Label(
    main_frame, 
    text=f"Model '{sentence}' cümlesini öğrendi.\nŞu kelimelerden birini girin: {', '.join(word_list)}",
    justify=tk.CENTER
)
intro_label.pack(pady=10)

# Kelime giriş kutusu (Entry)
entry_word = ttk.Entry(main_frame, width=40)
entry_word.pack(pady=10)

# "Cümle Üret" butonu
predict_button = ttk.Button(
    main_frame, 
    text="Cümle Üret", 
    command=on_predict_click # Tıklandığında on_predict_click fonksiyonunu çalıştır
)
predict_button.pack(pady=10)

# Sonucun gösterileceği etiket
# Metni dinamik olarak değiştirmek için bir StringVar kullanıyoruz
result_label_var = tk.StringVar()
result_label_var.set("Sonuç burada gösterilecek...")
result_label = ttk.Label(main_frame, textvariable=result_label_var, font=("Helvetica", 12, "italic"))
result_label.pack(pady=20)


# Arayüzün sürekli çalışmasını sağlayan ana döngü
root.mainloop()