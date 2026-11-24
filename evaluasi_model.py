import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Urutan kelas sesuai baris dan kolom pada Gambar 4.5 (Confusion Matrix)
classes = ['falling', 'sitting', 'standing', 'walking']

# Matriks Confusion Matrix persis dari hasil evaluasi Random Forest & SVM
cm_rf = np.array([
    [1749,    8,    4,   27],  # True: falling
    [   8,  615,    0,    1],  # True: sitting
    [   1,    0,  615,    2],  # True: standing
    [  43,    0,    6,  592]   # True: walking
])

cm_svm = np.array([
    [1669,   11,   15,   93],  # True: falling
    [  14,  608,    1,    1],  # True: sitting
    [  14,    0,  593,   11],  # True: standing
    [  69,    0,   38,  534]   # True: walking
])

def evaluate_from_cm(cm, model_name):
    # Rekonstruksi array y_true dan y_pred dari matriks
    y_true, y_pred = [], []
    for true_idx in range(len(classes)):
        for pred_idx in range(len(classes)):
            count = cm[true_idx, pred_idx]
            y_true.extend([classes[true_idx]] * count)
            y_pred.extend([classes[pred_idx]] * count)

    acc = accuracy_score(y_true, y_pred)
    print(f"============================================================")
    print(f"HASIL EVALUASI MODEL: {model_name}")
    print(f"Overall Accuracy: {acc * 100:.2f}% ({np.trace(cm)}/{np.sum(cm)} sampel)")
    print(f"============================================================")
    
    # Mencetak Classification Report standar Scikit-Learn
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(report)

    # Menyusun tabel rapi sesuai format Tabel 4.3 di laporan
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    rows = []
    for cls in ['sitting', 'standing', 'walking', 'falling']:
        rows.append({
            'Kelas Aktivitas': cls.capitalize(),
            'Precision (%)': round(report_dict[cls]['precision'] * 100, 2),
            'Recall (%)': round(report_dict[cls]['recall'] * 100, 2),
            'F1-Score (%)': round(report_dict[cls]['f1-score'] * 100, 2),
            'Sampel Uji': int(report_dict[cls]['support'])
        })
    rows.append({
        'Kelas Aktivitas': 'Rata-rata / Total (Macro)',
        'Precision (%)': round(report_dict['macro avg']['precision'] * 100, 2),
        'Recall (%)': round(report_dict['macro avg']['recall'] * 100, 2),
        'F1-Score (%)': round(report_dict['macro avg']['f1-score'] * 100, 2),
        'Sampel Uji': int(report_dict['macro avg']['support'])
    })
    df_table = pd.DataFrame(rows)
    print("\n--- Format Tabel Siap Salin ke Laporan ---")
    print(df_table.to_string(index=False))
    print("\n")

# Jalankan evaluasi untuk kedua model
evaluate_from_cm(cm_rf, "RANDOM FOREST")
evaluate_from_cm(cm_svm, "SUPPORT VECTOR MACHINE (SVM)")