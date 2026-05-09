import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import datetime

class MetadataBearingGenerator:
    def __init__(self, T=200, K=512):
        self.T, self.K = T, K
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.4)

        # Слайдери (Стария интерфейс)
        self.s_noise = Slider(plt.axes([0.2, 0.32, 0.6, 0.02]), 'Noise Floor', 0.1, 3.0, valinit=1.0)
        self.s_sev   = Slider(plt.axes([0.2, 0.28, 0.6, 0.02]), 'Signal Amp', 0.0, 20.0, valinit=8.0)
        self.s_harm  = Slider(plt.axes([0.2, 0.24, 0.6, 0.02]), 'Harmonics', 1, 5, valinit=3, valstep=1)
        self.s_wob   = Slider(plt.axes([0.2, 0.20, 0.6, 0.02]), 'Wobble', 0.0, 10.0, valinit=3.0)
        self.s_freq  = Slider(plt.axes([0.2, 0.16, 0.6, 0.02]), 'Impact Freq', 2.0, 40.0, valinit=15.0)

        for s in [self.s_noise, self.s_sev, self.s_harm, self.s_wob, self.s_freq]:
            s.on_changed(self.update_callback)

        self.btn = Button(plt.axes([0.4, 0.02, 0.2, 0.05]), 'Export Series + Log', color='lime')
        self.btn.on_clicked(self.export_data)

        self.im = None
        self.update_plot()

    def calculate_snr(self):
        """Изчислява теоретичното SNR в dB"""
        s_amp = self.s_sev.val
        n_std = self.s_noise.val
        if n_std == 0: return 0
        snr_db = 20 * np.log10(s_amp / n_std)
        return snr_db

    def generate_data(self):
        T, K = self.T, self.K
        amp_matrix = np.random.normal(0, self.s_noise.val, (T, K)).astype(np.float32)
        
        num_h = int(self.s_harm.val)
        # Генерираме уникална начална фаза за всяка линия
        initial_phases = np.random.uniform(0, 2*np.pi, num_h)
        
        base_f0 = K // 6
        for t in range(T):
            if t > T // 5:
                wobble = self.s_wob.val * np.sin(2 * np.pi * t / 20.0)
                impact = np.abs(np.sin(np.pi * t / self.s_freq.val))**12
                
                for h in range(1, num_h + 1):
                    center = h * base_f0 + wobble * h
                    # Фазово отместване, специфично за хармоника
                    p_offset = initial_phases[h-1] 
                    
                    if 0 < center < K:
                        width = 4.0 + 0.5 * h
                        dist = np.arange(K) - center
                        # Тук добавяме фазовото влияние в комплексния домейн по-късно
                        line = (self.s_sev.val * impact / h) * np.exp(-0.5 * (dist/width)**2)
                        amp_matrix[t, :] += line
        
        # Генерираме комплексния сигнал с разминаващи се начални фази
        final_phase = np.random.uniform(-np.pi, np.pi, (T, K)) 
        # (Забележка: В истински PR тест тук може да се вкара и p_offset в основната фаза)
        complex_data = amp_matrix * np.exp(1j * final_phase)
        return complex_data

    def update_plot(self):
        z = self.generate_data()
        snr = self.calculate_snr()
        if self.im is None:
            self.im = self.ax.imshow(np.abs(z), aspect='auto', cmap='magma')
        else:
            self.im.set_data(np.abs(z))
            self.im.set_clim(np.min(np.abs(z)), np.max(np.abs(z)))
        
        self.ax.set_title(f"Z-Complex Gen | SNR: {snr:.2f} dB | Harmonics: {int(self.s_harm.val)}")
        self.fig.canvas.draw_idle()

    def update_callback(self, val): self.update_plot()

    def export_data(self, event):
        out_dir = f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(out_dir, exist_ok=True)
        
        z_data = self.generate_data()
        snr = self.calculate_snr()
        
        # Запис на метаданни
        with open(os.path.join(out_dir, "metadata.txt"), "w") as f:
            f.write(f"Timestamp: {datetime.datetime.now()}\n")
            f.write(f"SNR (theoretical): {snr:.2f} dB\n")
            f.write(f"Noise Floor: {self.s_noise.val}\n")
            f.write(f"Signal Amplitude: {self.s_sev.val}\n")
            f.write(f"Harmonics: {int(self.s_harm.val)}\n")
            f.write(f"Wobble Amp: {self.s_wob.val}\n")
            f.write(f"Impact Frequency: {self.s_freq.val}\n")

        for t in range(self.T):
            with h5py.File(f"{out_dir}/frame_{t:04d}.h5", "w") as f:
                f.create_dataset("entry/instrument/spectrometer/data/signal", data=z_data[t,:])
        print(f"Exported to {out_dir} with metadata.txt")

if __name__ == "__main__":
    gen = MetadataBearingGenerator()
    plt.show()
