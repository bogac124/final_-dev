import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

i = 0
j = 1
x = 0
y = 0

veri = {
    'Yoğunluk': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150],
    'Sayi': [12, 18, 32, 48, 52, 65, 55, 42, 32, 16, 10, 5, 18, 25, 32, 40, 65, 43, 32, 20, 10, 4]
}

df = pd.DataFrame(veri)


min_yogunluk = df['Yoğunluk'][0]
max_yogunluk = df['Yoğunluk'][0]

while i < len(df['Yoğunluk']):
    if df['Yoğunluk'][i] < min_yogunluk:
        min_yogunluk = df['Yoğunluk'][i]
    if df['Yoğunluk'][i] > max_yogunluk:
        max_yogunluk = df['Yoğunluk'][i]
    i += 1

  

pikseller = []
for index, row in df.iterrows():
    pikseller.extend([row['Yoğunluk']] * row['Sayi'])
pikseller = np.array(pikseller)
T0 = np.mean(pikseller)
esik = 0.5
while True:
    G1 = pikseller[pikseller > T0]
    G2 = pikseller[pikseller <= T0]
    m1 = np.mean(G1) if len(G1) > 0 else 0
    m2 = np.mean(G2) if len(G2) > 0 else 0
    T1 = (m1 + m2) / 2
    if abs(T1 - T0) < esik:
        break

    T0 = T1

print(f"Optimum Eşik Değeri: {T0}")

G1_sayi = len(pikseller[pikseller > T0])
G2_sayi = len(pikseller[pikseller <= T0])

print(f"Eşik üstündeki piksel sayisi: {G1_sayi}")
print(f"Eşik altindaki veya eşit piksel sayisi: {G2_sayi}")

plt.hist(pikseller, bins=range(min_yogunluk,max_yogunluk), edgecolor='blue')
plt.axvline(T0, color='blue', linestyle='dashed', linewidth=1)
plt.title('Eşikli Piksel Yoğunluğu Histogrami')
plt.xlabel('Yoğunluk')
plt.ylabel('Frekans')
plt.show()
