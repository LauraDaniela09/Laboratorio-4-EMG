# Laboratorio-4-EMG

<h1 align="center"><i><b>𝐏𝐚𝐫𝐭𝐞 A 𝐝𝐞𝐥 𝐥𝐚𝐛𝐨𝐫𝐚𝐭𝐨𝐫𝐢𝐨</b></i></h1>

**visualizacion de la señal**
```python
# === Cargar señal ===
fs = 2000  # Frecuencia de muestreo
archivo = "/content/emg_data1.csv"

df = pd.read_csv(archivo)
t = df.iloc[:, 0].values
emg = df.iloc[:, 1].values

# === Graficar ===
plt.figure(figsize=(12,4))
plt.plot(t, emg, color='black')
plt.title("Señal EMG completa")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid(True)
plt.tight_layout()
plt.show()
```
## resultado
<p align="center">
<img width="700" height="390" alt="image" src="https://github.com/user-attachments/assets/c482b7ea-04d6-4140-8e1a-0904544b336d" />
</p>

**Detección y visualización de contracciones**
```python
df = pd.read_csv(archivo)
t = df.iloc[:, 0].values
emg = df.iloc[:, 1].values

# === Envolvente ===
analytic = signal.hilbert(emg)
envelope = np.abs(analytic)
ventana = int(0.02 * fs)
envelope_smooth = np.convolve(envelope, np.ones(ventana)/ventana, mode='same')

# === Detección de contracciones ===
umbral = np.mean(envelope_smooth) + 0.5 * np.std(envelope_smooth)
activo = envelope_smooth > umbral
min_duracion = int(0.1 * fs)

regiones = []
en_region = False
for i in range(len(activo)):
    if activo[i] and not en_region:
        inicio = i
        en_region = True
    if not activo[i] and en_region:
        fin = i
        en_region = False
        if fin - inicio >= min_duracion:
            regiones.append((inicio, fin))
if en_region:
    fin = len(activo) - 1
    if fin - inicio >= min_duracion:
        regiones.append((inicio, fin))

# === Graficar señal con contracciones ===
plt.figure(figsize=(12,4))
plt.plot(t, emg, color='black', label='Señal EMG')
plt.plot(t, envelope_smooth, color='#E50063', alpha=0.6, label='Envolvente suavizada')
for (s, e) in regiones:
    plt.axvspan(t[s], t[e], color='#FFB6C1', alpha=0.3)
plt.title("Contracciones detectadas")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.tight_layout()
plt.show()
```
## resultado
<p align="center">
<img width="900" height="390" alt="image" src="https://github.com/user-attachments/assets/117e246d-5e3f-4edf-abdf-436c13b51e53" />
</p>



<h1 align="center"><i><b>𝐏𝐚𝐫𝐭𝐞 B 𝐝𝐞𝐥 𝐥𝐚𝐛𝐨𝐫𝐚𝐭𝐨𝐫𝐢𝐨</b></i></h1>

<h1 align="center"><i><b>𝐏𝐚𝐫𝐭𝐞 C 𝐝𝐞𝐥 𝐥𝐚𝐛𝐨𝐫𝐚𝐭𝐨𝐫𝐢𝐨</b></i></h1>

```python
# Cargar datos
data = pd.read_csv("emg_data1.csv")

# Convertir columnas a valores numéricos
data["Tiempo [s]"] = pd.to_numeric(data["Tiempo [s]"], errors='coerce')
data["Voltaje [V]"] = pd.to_numeric(data["Voltaje [V]"], errors='coerce')

t = data["Tiempo [s]"].values
emg = data["Voltaje [V]"].values

# Frecuencia de muestreo estimada
fs = 1 / np.mean(np.diff(t))
N = len(emg)
print(f"Frecuencia de muestreo estimada: {fs:.2f} Hz")

# FFT
frecuencias = fftfreq(N, 1/fs)
fft_emg = fft(emg)
amplitud = np.abs(fft_emg) / N

# Gráfica señal original
plt.figure(figsize=(10,3))
plt.plot(t, emg, color='blue')
plt.title("Señal EMG original en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.grid(True)
plt.show()

```
<img width="844" height="316" alt="image" src="https://github.com/user-attachments/assets/8e581ef1-051e-4037-a31d-4e4bf310b28b" />

Frecuencia de muestreo estimada: 2000.00 Hz

```python
# --- b) Gráfica del espectro de amplitud ---

plt.figure(figsize=(10,4))
plt.plot(frecuencias[:N//2], amplitud[:N//2])
plt.title("Espectro de amplitud (Frecuencia vs Magnitud)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.show()
```

<img width="855" height="393" alt="image" src="https://github.com/user-attachments/assets/5076dbf2-41a4-47bc-967a-12ad0284695e" />

```python
mitad = N // 2
segmento1 = emg[:mitad]
segmento2 = emg[mitad:]

fft_seg1 = np.abs(fft(segmento1)) / len(segmento1)
fft_seg2 = np.abs(fft(segmento2)) / len(segmento2)
f_seg = fftfreq(len(segmento1), 1/fs)

plt.figure(figsize=(10,4))
plt.plot(f_seg[:len(f_seg)//2], fft_seg1[:len(f_seg)//2], label="Primeras contracciones", color='blue')
plt.plot(f_seg[:len(f_seg)//2], fft_seg2[:len(f_seg)//2], label="Últimas contracciones", color='orange', alpha=0.7)
plt.title("Comparación: primeras vs últimas contracciones")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()
```
<img width="855" height="394" alt="image" src="https://github.com/user-attachments/assets/ba531b9c-5150-4fb1-9a13-340b40544f6f" />

```python
energia_alta_1 = np.sum(fft_seg1[int(len(f_seg)*0.25):])
energia_alta_2 = np.sum(fft_seg2[int(len(f_seg)*0.25):])

plt.figure(figsize=(6,4))
plt.bar(["Inicio", "Final"], [energia_alta_1, energia_alta_2], color=['blue','orange'])
plt.title("Energía en altas frecuencias")
plt.ylabel("Energía relativa")
plt.grid(axis='y')
plt.show()

print(f"Energía altas frecuencias (inicio): {energia_alta_1:.4f}")
print(f"Energía altas frecuencias (final): {energia_alta_2:.4f}")

if energia_alta_2 < energia_alta_1:
    print("Se reduce el contenido de alta frecuencia → posible fatiga muscular.")
else:
    print("No hay reducción significativa en altas frecuencias.")
```

<img width="523" height="374" alt="image" src="https://github.com/user-attachments/assets/9568fa76-e26c-4b87-b810-beb09d3935b8" />


Energía altas frecuencias (inicio): 7.7270
Energía altas frecuencias (final): 7.7534
No hay reducción significativa en altas frecuencias.

```python
pico_inicial = f_seg[np.argmax(fft_seg1[:len(f_seg)//2])]
pico_final = f_seg[np.argmax(fft_seg2[:len(f_seg)//2])]

plt.figure(figsize=(10,4))
plt.plot(f_seg[:len(f_seg)//2], fft_seg1[:len(f_seg)//2], label=f"Inicio (pico {pico_inicial:.1f} Hz)", color='blue')
plt.plot(f_seg[:len(f_seg)//2], fft_seg2[:len(f_seg)//2], label=f"Final (pico {pico_final:.1f} Hz)", color='orange', alpha=0.7)
plt.axvline(pico_inicial, color='blue', linestyle='--')
plt.axvline(pico_final, color='orange', linestyle='--')
plt.title("Desplazamiento del pico espectral")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud")
plt.legend()
plt.grid(True)
plt.show()

print(f"Pico espectral inicial: {pico_inicial:.2f} Hz")
print(f"Pico espectral final: {pico_final:.2f} Hz")

if pico_final < pico_inicial:
    print("El pico espectral se desplazó hacia frecuencias bajas → esfuerzo sostenido.")
else:
    print("No se observa desplazamiento hacia bajas frecuencias.")
```

<img width="855" height="393" alt="image" src="https://github.com/user-attachments/assets/73c6db60-ab93-429e-b883-daafbd5ce442" />


Pico espectral inicial: 4.00 Hz
Pico espectral final: 4.00 Hz
No se observa desplazamiento hacia bajas frecuencias.


Conclusiones:

El análisis espectral mediante la Transformada Rápida de Fourier (FFT)
permite observar cómo el contenido en frecuencia de la señal EMG cambia
durante el esfuerzo muscular. 

Una disminución de energía en altas frecuencias y el desplazamiento del pico
hacia frecuencias más bajas son indicadores de fatiga muscular. 

Por tanto, la FFT es una herramienta diagnóstica útil en electromiografía
para evaluar la condición del músculo y su comportamiento ante esfuerzos sostenidos.
