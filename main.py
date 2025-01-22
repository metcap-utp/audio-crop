import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def load_audios_from_folder(folder_path):
    audio_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(folder_path, file_name)
            audio, sr = librosa.load(file_path, sr=None)
            audio_files.append((audio, sr, file_name))
    return audio_files


def plot_waveform(audio, sr, title="Waveform"):
    plt.figure(figsize=(14, 6))
    times = np.arange(len(audio)) / sr
    plt.plot(times, audio)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def crop_audio(
    file_path, silence_thresh=-20, min_silence_len=500, buffer_ms=3000
):
    audio = AudioSegment.from_wav(file_path)
    nonsilent_ranges = detect_nonsilent(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh
    )

    # Ordenar rangos por tiempo de inicio
    nonsilent_ranges.sort(key=lambda x: x[0])

    # Unir rangos que se superponen
    merged_ranges = []
    if nonsilent_ranges:
        current_start, current_end = nonsilent_ranges[0]
        for start, end in nonsilent_ranges[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                # Remover búfer desde el inicio y final de las partes activas
                adjusted_start = current_start + buffer_ms
                adjusted_end = current_end - buffer_ms
                if adjusted_start < adjusted_end:  # Solo agregar si queda audio después de quitar el búfer
                    merged_ranges.append((adjusted_start, adjusted_end))
                current_start, current_end = start, end

        # Agregar el último rango con búfer ajustado
        adjusted_start = current_start + buffer_ms
        adjusted_end = current_end - buffer_ms
        if adjusted_start < adjusted_end:
            merged_ranges.append((adjusted_start, adjusted_end))

    cropped_audio = AudioSegment.empty()
    for start, end in merged_ranges:
        cropped_audio += audio[start:end]

    return cropped_audio


if __name__ == "__main__":
    folder_path = "audio"
    cropped_folder_path = "cropped"
    graph_folder_path = "graph"
    enable_graphs = True  # Set this to False to disable graphs
    os.makedirs(cropped_folder_path, exist_ok=True)
    os.makedirs(graph_folder_path, exist_ok=True)

    audio_files = load_audios_from_folder(folder_path)
    for audio, sr, file_name in audio_files:
        original_length_ms = int((len(audio) / sr) * 1000)
        if enable_graphs:
            plt.figure(figsize=(14, 6))
            plt.subplot(2, 1, 1)
            times = np.arange(len(audio)) / sr
            plt.plot(times, audio)
            plt.title("Original: " + file_name)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")

        cropped_audio = crop_audio(os.path.join(folder_path, file_name))
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
            plt.subplot(2, 1, 2)
            cropped_times = np.arange(len(cropped_audio_data)) / cropped_sr
            plt.plot(cropped_times, cropped_audio_data)
            plt.title("Cropped: " + file_name)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")

            plt.tight_layout()
            plt.savefig(os.path.join(graph_folder_path, f"{file_name}.png"))
            plt.close()
