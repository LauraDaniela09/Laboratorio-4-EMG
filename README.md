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
<img width="900" height="390" alt="image" src="https://github.com/user-attachments/assets/c482b7ea-04d6-4140-8e1a-0904544b336d" />
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


