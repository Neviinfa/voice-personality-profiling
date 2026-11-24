import os
import warnings

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# 1. PATHS
# ============================================================

AUDIO_DIR = r"D:\VoicepersonalityProject\data\AudioWAV_Clean"

OUTPUT_FILE = (
    r"D:\VoicepersonalityProject\features"
    r"\enhanced_vad_audio_features.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256

warnings.filterwarnings("ignore")


# ============================================================
# 3. STATISTICS HELPER
# ============================================================

def calculate_statistics(values, prefix):

    values = np.asarray(values, dtype=np.float32)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_median": 0.0
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_median": float(np.median(values))
    }


# ============================================================
# 4. EXTRACT FEATURES FROM ONE AUDIO FILE
# ============================================================

def extract_features(file_path):

    features = {}

    try:

        # ----------------------------------------------------
        # LOAD AUDIO
        # ----------------------------------------------------

        y, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(y) == 0:
            return None

        y = np.nan_to_num(y)

        duration = len(y) / sr

        features["duration"] = float(duration)


        # ====================================================
        # 1. RMS / ENERGY
        # ====================================================

        rms = librosa.feature.rms(
            y=y,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH
        )[0]

        features.update(
            calculate_statistics(
                rms,
                "rms"
            )
        )


        # ====================================================
        # 2. ZERO CROSSING RATE
        # ====================================================

        zcr = librosa.feature.zero_crossing_rate(
            y,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH
        )[0]

        features.update(
            calculate_statistics(
                zcr,
                "zcr"
            )
        )


        # ====================================================
        # 3. SPECTRAL CENTROID
        # ====================================================

        spectral_centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )[0]

        features.update(
            calculate_statistics(
                spectral_centroid,
                "spectral_centroid"
            )
        )


        # ====================================================
        # 4. SPECTRAL BANDWIDTH
        # ====================================================

        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )[0]

        features.update(
            calculate_statistics(
                spectral_bandwidth,
                "spectral_bandwidth"
            )
        )


        # ====================================================
        # 5. SPECTRAL ROLLOFF
        # ====================================================

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )[0]

        features.update(
            calculate_statistics(
                spectral_rolloff,
                "spectral_rolloff"
            )
        )


        # ====================================================
        # 6. SPECTRAL FLATNESS
        # ====================================================

        spectral_flatness = librosa.feature.spectral_flatness(
            y=y,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )[0]

        features.update(
            calculate_statistics(
                spectral_flatness,
                "spectral_flatness"
            )
        )


        # ====================================================
        # 7. SPECTRAL CONTRAST
        # ====================================================

        spectral_contrast = librosa.feature.spectral_contrast(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        for i in range(spectral_contrast.shape[0]):

            features.update(
                calculate_statistics(
                    spectral_contrast[i],
                    f"spectral_contrast_{i + 1}"
                )
            )


        # ====================================================
        # 8. MFCC 1-13
        # ====================================================

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        for i in range(13):

            features.update(
                calculate_statistics(
                    mfcc[i],
                    f"mfcc_{i + 1}"
                )
            )


        # ====================================================
        # 9. MFCC DELTA
        # ====================================================

        mfcc_delta = librosa.feature.delta(
            mfcc
        )

        for i in range(13):

            features.update(
                calculate_statistics(
                    mfcc_delta[i],
                    f"mfcc_delta_{i + 1}"
                )
            )


        # ====================================================
        # 10. MFCC DELTA-DELTA
        # ====================================================

        mfcc_delta2 = librosa.feature.delta(
            mfcc,
            order=2
        )

        for i in range(13):

            features.update(
                calculate_statistics(
                    mfcc_delta2[i],
                    f"mfcc_delta2_{i + 1}"
                )
            )


        # ====================================================
        # 11. CHROMA
        # ====================================================

        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        for i in range(12):

            features.update(
                calculate_statistics(
                    chroma[i],
                    f"chroma_{i + 1}"
                )
            )


        # ====================================================
        # 12. PITCH / F0
        # ====================================================

        try:

            pitch = librosa.yin(
                y,
                fmin=70,
                fmax=500,
                sr=sr,
                frame_length=2048,
                hop_length=HOP_LENGTH
            )

            pitch = pitch[
                np.isfinite(pitch)
            ]

            features.update(
                calculate_statistics(
                    pitch,
                    "pitch"
                )
            )

        except Exception:

            features.update(
                calculate_statistics(
                    [],
                    "pitch"
                )
            )


        # ====================================================
        # 13. VOICED / UNVOICED RATIO
        # ====================================================

        try:

            voiced_flag, _ = librosa.pyin(
                y,
                fmin=70,
                fmax=500,
                sr=sr,
                frame_length=2048,
                hop_length=HOP_LENGTH
            )

            voiced_flag = np.asarray(
                voiced_flag
            )

            valid_frames = len(
                voiced_flag
            )

            voiced_frames = np.sum(
                ~np.isnan(voiced_flag)
            )

            if valid_frames > 0:

                voiced_ratio = (
                    voiced_frames / valid_frames
                )

            else:

                voiced_ratio = 0.0

            features["voiced_ratio"] = float(
                voiced_ratio
            )

            features["unvoiced_ratio"] = float(
                1.0 - voiced_ratio
            )

        except Exception:

            features["voiced_ratio"] = 0.0
            features["unvoiced_ratio"] = 1.0


        # ====================================================
        # 14. ENERGY CHANGE
        # ====================================================

        if len(rms) > 1:

            energy_change = np.diff(rms)

            features.update(
                calculate_statistics(
                    np.abs(energy_change),
                    "energy_change"
                )
            )

        else:

            features.update(
                calculate_statistics(
                    [],
                    "energy_change"
                )
            )


        # ====================================================
        # 15. PITCH CHANGE
        # ====================================================

        try:

            pitch_values = librosa.yin(
                y,
                fmin=70,
                fmax=500,
                sr=sr,
                frame_length=2048,
                hop_length=HOP_LENGTH
            )

            pitch_values = pitch_values[
                np.isfinite(pitch_values)
            ]

            if len(pitch_values) > 1:

                pitch_change = np.diff(
                    pitch_values
                )

                features.update(
                    calculate_statistics(
                        np.abs(pitch_change),
                        "pitch_change"
                    )
                )

            else:

                features.update(
                    calculate_statistics(
                        [],
                        "pitch_change"
                    )
                )

        except Exception:

            features.update(
                calculate_statistics(
                    [],
                    "pitch_change"
                )
            )


        # ====================================================
        # 16. AMPLITUDE
        # ====================================================

        amplitude = np.abs(y)

        features.update(
            calculate_statistics(
                amplitude,
                "amplitude"
            )
        )


        return features


    except Exception as e:

        print(
            f"\nError processing "
            f"{os.path.basename(file_path)}: {e}"
        )

        return None


# ============================================================
# 5. MAIN FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("ENHANCED VAD FEATURE EXTRACTION")
    print("=" * 70)

    print("\nAudio directory:")
    print(AUDIO_DIR)

    print("\nOutput file:")
    print(OUTPUT_FILE)


    # --------------------------------------------------------
    # CHECK DIRECTORY
    # --------------------------------------------------------

    if not os.path.exists(AUDIO_DIR):

        raise FileNotFoundError(
            f"\nAudio directory not found:\n{AUDIO_DIR}"
        )


    # --------------------------------------------------------
    # FIND WAV FILES
    # --------------------------------------------------------

    audio_files = [
        file
        for file in os.listdir(AUDIO_DIR)
        if file.lower().endswith(".wav")
    ]

    audio_files.sort()

    print(
        f"\nTotal WAV files found: "
        f"{len(audio_files)}"
    )


    if len(audio_files) == 0:

        raise ValueError(
            "No WAV files found."
        )


    # --------------------------------------------------------
    # PROCESS AUDIO
    # --------------------------------------------------------

    all_features = []
    failed_files = []

    print("\nExtracting enhanced features...\n")


    for filename in tqdm(
        audio_files,
        desc="Processing audio"
    ):

        file_path = os.path.join(
            AUDIO_DIR,
            filename
        )

        result = extract_features(
            file_path
        )

        if result is not None:

            result["filename"] = filename


            # ------------------------------------------------
            # EXTRACT EMOTION FROM CREMA-D FILENAME
            # ------------------------------------------------

            parts = filename.split("_")

            if len(parts) >= 3:

                emotion_code = parts[2]

                emotion_map = {
                    "ANG": "anger",
                    "DIS": "disgust",
                    "FEA": "fear",
                    "HAP": "happy",
                    "NEU": "neutral",
                    "SAD": "sad"
                }

                result["emotion"] = emotion_map.get(
                    emotion_code,
                    emotion_code
                )

            else:

                result["emotion"] = "unknown"


            all_features.append(
                result
            )

        else:

            failed_files.append(
                filename
            )


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    print("\nCreating feature dataframe...")

    feature_df = pd.DataFrame(
        all_features
    )


    # --------------------------------------------------------
    # REORDER COLUMNS
    # --------------------------------------------------------

    first_columns = [
        "filename",
        "emotion"
    ]

    remaining_columns = [
        column
        for column in feature_df.columns
        if column not in first_columns
    ]

    feature_df = feature_df[
        first_columns + remaining_columns
    ]


    # --------------------------------------------------------
    # CLEAN NUMERIC VALUES
    # --------------------------------------------------------

    numeric_columns = feature_df.select_dtypes(
        include=[np.number]
    ).columns

    feature_df[numeric_columns] = (
        feature_df[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )


    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )


    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    feature_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n" + "=" * 70)
    print("ENHANCED FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        f"\nSuccessful files : "
        f"{len(feature_df)}"
    )

    print(
        f"Failed files     : "
        f"{len(failed_files)}"
    )

    print(
        f"Feature columns   : "
        f"{len(feature_df.columns)}"
    )

    print(
        f"\nOutput file:\n"
        f"{OUTPUT_FILE}"
    )


    # --------------------------------------------------------
    # EMOTION DISTRIBUTION
    # --------------------------------------------------------

    print("\nEmotion distribution:")

    print(
        feature_df["emotion"].value_counts()
    )


    # --------------------------------------------------------
    # MISSING VALUE CHECK
    # --------------------------------------------------------

    total_missing = (
        feature_df.isnull()
        .sum()
        .sum()
    )

    print(
        f"\nTotal missing values: "
        f"{total_missing}"
    )


    # --------------------------------------------------------
    # FEATURE PREVIEW
    # --------------------------------------------------------

    print("\nFirst 20 columns:")

    print(
        feature_df.columns[
            :20
        ].tolist()
    )


    # --------------------------------------------------------
    # FAILED FILES
    # --------------------------------------------------------

    if failed_files:

        print("\nFirst failed files:")

        for filename in failed_files[:20]:

            print(filename)

        if len(failed_files) > 20:

            print(
                f"... and "
                f"{len(failed_files) - 20} more"
            )


# ============================================================
# 6. RUN
# ============================================================

if __name__ == "__main__":
    main()