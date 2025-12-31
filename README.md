⚠️ **پروژه تمرینی - نه برای استفاده علمی**

این یک **مقاله تحقیقی تمرینی** است که دارای معایبی از جمله:
- نشت داده احتمالی  
- نتایج متناقض (CV۷۷.۷٪ vs p=۰.۸۰۳)
- فرض‌های ضعیف
- فقدان منابع
می باشد.

**صرفاً برای آموزش است. نتایج قابل اعتماد نیستند.**



## 📁 ساختار پروژه:

- **`src/`** - کد اصلی Python (proof of concept)
- **`technical_report.pdf`** - گزارش فنی تفصیلی (معماری، پیاده‌سازی، پارامترها)
- **`research.md`** - مقاله تحقیقی (مقدمه، روش‌شناسی، نتایج، بحث)

برای درک کامل معایب و محدودیت‌ها، لطفاً هر دو فایل را بخوانید.


## 🚀 نحوه اجرا (صرفاً برای تست):

**توجه**: این کد صرفاً برای آموزش است و نتایج قابل اعتماد نیستند.

### پیش‌نیازها:
```bash
pip install numpy scipy scikit-learn pandas
```

### دانلود مجموعه داده:
مجموعه داده DEAP را از Kaggle دانلود کنید:
- **لینک**: [https://www.kaggle.com/datasets/manh123df/deap-dataset](https://www.kaggle.com/datasets/manh123df/deap-dataset)
- فایل‌ها را در مسیر `../data/deap-dataset/` قرار دهید

ساختار انتظاری:

```
├── data
│   └── deap-dataset
│       ├── EDA_DEAP.ipynb
│       ├── Metadata
│       │   ├── online_ratings.xls
│       │   ├── participant_questionnaire.xls
│       │   ├── participant_ratings.xls
│       │   ├── video_list.xlsx
│       │   └── video_list_fixed.xlsx
│       ├── audio_stimuli_MIDI
│       │   ├── exp_id_1.mid
│       │   ├── ...
│       │   └── exp_id_40.mid
│       ├── audio_stimuli_MIDI_tempo24
│       │   ├── exp_id_1_tempo24.mid
│       │   ├── ...
│       │   └── exp_id_40_tempo24.mid
│       ├── data_preprocessed_python
│       │   ├── s01.dat
│       │   ├── ...
│       │   └── s32.dat
│       └── metadata_xls
│           ├── online_ratings.xls
│           ├── participant_questionnaire.xls
│           ├── participant_ratings.xls
│           └── video_list.xls
```

### نحوه تست:
```bash
cd src/
python main.py \
    --dataset_root ../data/deap-dataset \
    --output_dir results_test_5subjects \
    --scheme binary \
    --target valence \
    --skip_artifact_rejection \
    --max_subjects 5 \
    --pca_components 10 \
    --min_samples_per_state 10 \
    --log_level INFO

```
