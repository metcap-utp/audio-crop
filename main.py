import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment


def load_audios_from_folder(folder_path):
    audio_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            audio, sr = librosa.load(file_path, sr=None)
            audio_files.append((audio, sr, file_name))
    return audio_files


def detect_welding_segments(
    audio_data, sr, freq_min=200, freq_max=5000, threshold=0.15
):
    """
    Detectar segmentos que tienen alta energía en la banda de frecuencia de soldadura.
    freq_min, freq_max: rango de frecuencia (Hz) típico para el sonido de soldadura
    threshold: cuadros por encima de esta energía normalizada se consideran "soldadura"
    """
    # Calcular STFT
    stft = np.abs(librosa.stft(audio_data, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # Encontrar índices para el rango de frecuencia deseado
    band_indices = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]

    # Sumar energía en la banda
    band_energy = stft[band_indices, :].sum(axis=0)

    # Normalizar y aplicar umbral
    band_energy /= band_energy.max()
    welding_frames = band_energy > threshold

    # Convertir índices de cuadros a tiempo (ms) para recorte posterior
    frame_times = (
        librosa.frames_to_time(
            np.arange(len(welding_frames)), sr=sr, hop_length=512
        )
        * 1000
    )

    # Identificar segmentos contiguos por encima del umbral
    segmentos = []
    is_active = False
    segment_start = 0
    for i, active in enumerate(welding_frames):
        if active and not is_active:
            is_active = True
            segment_start = frame_times[i]
        elif not active and is_active:
            is_active = False
            segmentos.append((segment_start, frame_times[i]))
    # Si termina mientras aún está activo
    if is_active:
        segmentos.append((segment_start, frame_times[-1]))

    return segmentos


def crop_welding_audio(
    file_path, freq_min=500, freq_max=5000, threshold=0.15, crop_start_end_s=0
):
    """
    Recortar solo los segmentos de soldadura del audio y recortar una parte (crop_start_end_s)
    al inicio y al final del audio completo.
    """
    audio_data, sr = librosa.load(file_path, sr=None)

    # Paso 1: Detectar segmentos y unirlos
    segmentos = detect_welding_segments(
        audio_data, sr, freq_min, freq_max, threshold
    )

    rangos_unidos = []
    if segmentos:
        current_start, current_end = segmentos[0]
        for start, end in segmentos[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                rangos_unidos.append((current_start, current_end))
                current_start, current_end = start, end
        rangos_unidos.append((current_start, current_end))

    # Paso 2: Ordenar y volver a unir los segmentos
    rangos_unidos.sort(key=lambda r: r[0])
    completamente_unidos = []
    if rangos_unidos:
        current_start, current_end = rangos_unidos[0]
        for start, end in rangos_unidos[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                completamente_unidos.append((current_start, current_end))
                current_start, current_end = start, end
        completamente_unidos.append((current_start, current_end))

    # Paso 3: Concatenar los rangos finales unidos
    audio_segment = AudioSegment.from_wav(file_path)
    cropped_audio = AudioSegment.empty()
    for start_ms, end_ms in completamente_unidos:
        cropped_audio += audio_segment[start_ms:end_ms]

    # Recortar los primeros y últimos segundos del audio completo
    crop_start_end_ms = crop_start_end_s * 1000
    cropped_audio = cropped_audio[crop_start_end_ms:-crop_start_end_ms]

    return cropped_audio


def plot_waveforms(original_audio, original_sr, cropped_audio, cropped_sr, file_name, output_path):
    plt.figure(figsize=(14, 12))

    plt.subplot(2, 1, 1)
    times = np.arange(len(original_audio)) / original_sr
    plt.plot(times, original_audio)
    plt.title("Original: " + file_name)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")

    plt.subplot(2, 1, 2)
    cropped_times = np.arange(len(cropped_audio)) / cropped_sr
    plt.plot(cropped_times, cropped_audio)
    plt.title("Cropped: " + file_name)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    folder_path = "audio"
    cropped_folder_path = "cropped"
    graph_folder_path = "graph"
    enable_graphs = True  # False para deshabilitar gráficos
    crop_start_end_s = 3  # Segundos a recortar del inicio y final del audio completo
    os.makedirs(cropped_folder_path, exist_ok=True)
    os.makedirs(graph_folder_path, exist_ok=True)

    audio_files = load_audios_from_folder(folder_path)
    for audio, sr, file_name in audio_files:
        original_length_ms = int((len(audio) / sr) * 1000)

        cropped_audio = crop_welding_audio(
            os.path.join(folder_path, file_name), crop_start_end_s=crop_start_end_s
        )

        cropped_audio.export(
            os.path.join(cropped_folder_path, "cropped_" + file_name),
            format="wav",
        )

        cropped_length_ms = len(cropped_audio)
        time_removed_ms = original_length_ms - cropped_length_ms
        time_removed_s = time_removed_ms / 1000.0
        print(f"Se removieron {time_removed_s} segundos de {file_name}")

        if enable_graphs:
            cropped_audio_data, cropped_sr = librosa.load(
                os.path.join(cropped_folder_path, "cropped_" + file_name),
                sr=None,
            )
            plot_waveforms(audio, sr, cropped_audio_data, cropped_sr, file_name, os.path.join(graph_folder_path, f"{file_name}.png"))


if __name__ == "__main__":
    main()
