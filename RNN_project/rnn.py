import torch
import torch.nn as nn
import numpy as np

# Adım 1: Veri Hazırlığı
# Modelimize öğreteceğimiz cümle
sentence = "kedi koltuğun üzerine zıpladı"

# Cümleyi kelimelere ayırıyoruz (tokenization)
tokens = sentence.split(' ')

# Benzersiz kelimelerden bir sözlük oluşturuyoruz
# Bu, her kelimeye benzersiz bir sayı (ID) atamamızı sağlar
word_list = sorted(list(set(tokens)))
word_to_ix = {word: i for i, word in enumerate(word_list)}
ix_to_word = {i: word for i, word in enumerate(word_list)}

# Sözlüğümüzdeki toplam kelime sayısı (modelin boyutlarını belirlemede kullanılacak)
vocab_size = len(word_list)

print("Kelimeden Sayıya Sözlük (word_to_ix):")
print(word_to_ix)
print(f"\nToplam Benzersiz Kelime Sayısı: {vocab_size}")



# Adım 2: Girdi (Inputs) ve Hedef (Targets) Çiftlerini Oluşturma

inputs = []
targets = []

for i in range(len(tokens) - 1):
    # Girdi: Mevcut kelimenin ID'si
    inputs.append(word_to_ix[tokens[i]])
    # Hedef: Bir sonraki kelimenin ID'si
    targets.append(word_to_ix[tokens[i+1]])

# Listeleri PyTorch tensörlerine dönüştürüyoruz
inputs = torch.LongTensor(inputs)
targets = torch.LongTensor(targets)

print("\nModelin Girdileri (Sayısal):", inputs)
print("Modelin Hedefleri (Sayısal):", targets)



# Adım 3: RNN Model Mimarisi

class WordRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(WordRNN, self).__init__()
        
        # 1. Embedding Katmanı: Kelime ID'lerini yoğun vektörlere dönüştürür.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. RNN Katmanı: Sıralı veriyi işler ve hafızayı taşır.
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        
        # 3. Fully Connected (Linear) Katman: RNN çıktısını nihai tahmine dönüştürür.
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        # x'in boyutu: (batch_size, sequence_length)
        
        # Girdileri embedding'den geçir.
        # Çıktı boyutu: (batch_size, sequence_length, embedding_dim)
        x = self.embedding(x)
        
        # RNN'den geçir.
        # outputs boyutu: (batch_size, sequence_length, hidden_size)
        # hidden_state boyutu: (num_layers, batch_size, hidden_size)
        outputs, hidden_state = self.rnn(x, hidden_state)
        
        # Son katmana göndererek skoru hesapla.
        # Çıktı boyutu: (batch_size, sequence_length, vocab_size)
        outputs = self.fc(outputs)
        
        return outputs, hidden_state

# Model parametreleri
embedding_dim = 10 # Her kelimeyi temsil eden vektörün boyutu
hidden_size = 12   # RNN hafızasının boyutu

# Modeli oluştur
model = WordRNN(vocab_size, embedding_dim, hidden_size)
print("\nModel Mimarisi:\n", model)



# Adım 4: Kayıp Fonksiyonu ve Optimize Edici

# Kayıp fonksiyonu: Tahmin ile gerçek hedef arasındaki farkı ölçer.
criterion = nn.CrossEntropyLoss()
# Optimize edici: Modelin parametrelerini kaybı azaltacak yönde günceller.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



# Adım 5: Modelin Eğitimi

epochs = 500

for epoch in range(epochs):
    # Giriş tensörünün boyutunu modele uygun hale getiriyoruz (batch_size=1)
    # [3] -> [[3]]
    model_inputs = inputs.unsqueeze(0)
    
    # İleri yayılım: Tahminleri al
    outputs, _ = model(model_inputs)
    
    # Çıktıların boyutunu kayıp fonksiyonuna uygun hale getiriyoruz
    # (batch_size, sequence_length, vocab_size) -> (batch_size * sequence_length, vocab_size)
    loss = criterion(outputs.view(-1, vocab_size), targets)
    
    # Geri yayılım ve optimizasyon
    optimizer.zero_grad() # Gradyanları sıfırla
    loss.backward()       # Yeni gradyanları hesapla
    optimizer.step()        # Parametreleri güncelle
    
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')



# Adım 6: Test Etme ve Cümle Üretimi

# Modeli değerlendirme moduna alıyoruz (eğitim dışı)
model.eval()

# Başlangıç kelimesi
start_word = "kedi"
generated_sentence = [start_word]

# Başlangıç kelimesinin ID'sini alıp tensöre çeviriyoruz
current_input = torch.LongTensor([[word_to_ix[start_word]]])
hidden_state = None # Başlangıçta hafıza boş

# Cümlenin geri kalanını üretmek için döngü
for _ in range(len(tokens) - 1):
    with torch.no_grad(): # Gradyan hesaplamasını kapat
        output, hidden_state = model(current_input, hidden_state)
        
        # Çıktıdaki en yüksek skorlu kelimenin ID'sini alıyoruz
        _, predicted_ix = torch.max(output, 2)
        
        # Tahmin edilen ID'yi bir sonraki döngünün girdisi yap
        current_input = predicted_ix
        
        # Tahmin edilen kelimeyi cümleye ekle
        predicted_word = ix_to_word[predicted_ix.item()]
        generated_sentence.append(predicted_word)

print("\nOrijinal Cümle:", sentence)
print("Modelin Ürettiği Cümle:", ' '.join(generated_sentence))