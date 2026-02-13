# MLOps Assignment #01

**Student:** Ayaan Khan  
**University ID:** 22I-2066  
**Course:** MLOps (BS DS)  
**Submission Date:** 13-Feb-2026  

---

## ✅ EC2 Instance Details

- **Instance Type:** t2.micro
- **Operating System:** Ubuntu 22.04 LTS  
- **EBS Attached:** 10 GB volume mounted at /mnt/ml-data  
- **IAM Role:** MLopsEC2S3Role  

---

## ✅ S3 Bucket Details

- **Bucket Name:** mlops-ayaan  
- **Versioning:** Enabled  
- **Access:** Private (IAM role required)  
- **Stored Files:**  
  /datasets/raw.csv  
  /datasets/processed.csv

---

## ✅ Directory Structure on EBS

/mnt/ml-data/  
├── datasets/         # Raw and processed datasets  
├── features/         # Feature files (optional)  
├── models/           # Saved models and scaler  
├── logs/             # Metrics logs  
├── venv/             # Python virtual environment  
├── setup_ml_env.sh   # Bootstrap environment script  
└── train_pipeline.py # ML pipeline script  

---

## ✅ Scripts

1. **setup_ml_env.sh**  
   - Installs Python3, pip, virtualenv  
   - Creates virtual environment at /mnt/ml-data/venv  
   - Installs required ML libraries: pandas, numpy, scikit-learn, joblib  
   - Idempotent: can be re-run after EC2 restart without errors  

2. **train_pipeline.py**  
   - Loads processed dataset from /mnt/ml-data/datasets/processed.csv  
   - Performs feature scaling  
   - Trains RandomForest and LogisticRegression models  
   - Logs metrics to /mnt/ml-data/logs/metrics.log  
   - Saves best model and scaler to /mnt/ml-data/models/  

---

## ✅ Execution Steps

1. **Set up ML environment:**

chmod +x /mnt/ml-data/setup_ml_env.sh  
/mnt/ml-data/setup_ml_env.sh  

2. **Activate virtual environment:**

source /mnt/ml-data/venv/bin/activate  

3. **Run ML pipeline:**

python /mnt/ml-data/train_pipeline.py  

- Logs metrics automatically to /mnt/ml-data/logs/metrics.log  
- Saves best model and scaler to /mnt/ml-data/models/  

4. **Upload / Sync datasets (if needed):**

aws s3 sync /mnt/ml-data/datasets s3://mlops-ayaan/datasets/  
aws s3 sync s3://mlops-ayaan/datasets /mnt/ml-data/datasets/  

5. **Auto-Shutdown (Task 5):**

crontab -l  
# 59 23 * * * /sbin/shutdown -h now  

- After restart, all data on /mnt/ml-data persists  

---

## ✅ Notes

- All outputs (models, logs, scaler, datasets) are stored on EBS (/mnt/ml-data)  
- IAM role MLopsEC2S3Role provides secure S3 access; no access keys used  
- Scripts are idempotent — safe to rerun after reboot  
- Ensure EC2 instance has EBS attached and mounted at /mnt/ml-data  

---

**Prepared by:** Ayaan Khan
EOF
