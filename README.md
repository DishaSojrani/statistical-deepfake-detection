Statistical Deepfake Detection
---------------
This project uses statistical image analysis techniques to detect deepfake images by extracting texture-based and noise-based statistical features. The model classifies real vs fake images using supervised machine learning.
--------------
📌 Project Overview
---
1.Detect deepfake images using statistical image features.

2.Extract features like entropy, variance, pixel distribution, noise residuals.

3.Apply ML models (SVM / PCA / CNN).

4.Visualize real vs fake patterns.

5.Evaluate results using accuracy, confusion matrix, ROC curve.

-------------
🧠 Techniques Used
---
1.Python

2.OpenCV

3.NumPy, Pandas

4.Scikit-Learn

5.Statistical Image Features

6.Machine Learning Classification

--------------
📂 Folder Structure
---
data/

 └── sample_images/
 
notebooks/

scripts/

images/

results/

README.md

-------------
🚀 How to Run
---
pip install -r requirements.txt

python detect_deepfake.py

-------------
📊 Results
---
1.Accuracy: SVM:-95.46%; CNN:-88.33%; PCA:-80.13%

2.Precision & Recall: SVM:-94.02% & 97.10%; CNN:-84.19% & 94.41%; PCA:-80.27% & 79.96%

3.F1 Score: SVM:-95.54%; CNN:-89.01%; PCA:-80.11%

4.AUC: SVM:-98.87%; CNN:-95.90%; PCA:-80.11%

5.Train Time: SVM:-315.47s; CNN:-178.00s; PCA:-0.0312s

Some results are shown in files.

-------------
📎 Dataset
---
You may use sample datasets like:

1.FaceForensics++

2.Deepfake Detection Challenge (DFDC)

3.kaggle

Some sample images are uploded in files.


------------
👩‍💻 Author
---
Ms Disha Rajkumar Sojrani
---
M.Sc. Statistics | Data Science & ML Enthusiast
---

