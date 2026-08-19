VACCINE REACTION PREDICTION
FINAL SHAREABLE PACKAGE

QUICK START - WINDOWS
---------------------
FIRST TIME:
1. Double-click setup.bat.
2. Wait for installation to finish.

NORMAL LOCAL USE:
1. Double-click run_app.bat.
2. The GUI opens in your browser.

HIDDEN LOCAL USE:
1. Double-click Open App Hidden.vbs.
2. The GUI opens without an obvious terminal window.

SAME WI-FI NETWORK:
1. Double-click run_app_network.bat.
2. Use the laptop's IPv4 address with port 8501 on the other device.
   Example: http://192.168.1.25:8501
3. The laptop must stay ON.

PUBLIC ONLINE ACCESS:
A permanent public link requires deployment to a cloud service.
See DEPLOYMENT_GUIDE.txt.

VS CODE:
.venv\Scripts\activate
python -m streamlit run app.py

MODEL:
The trained CatBoost model is already included.
Normal users do NOT need train_model.py.

PATIENT-FACING RESULT:
The GUI excludes the large "OTHER REPORTED REACTION" catch-all class from
the displayed result and selects the highest-scoring specific reaction
category. Accuracy and model probability scores are intentionally hidden
from the patient-facing interface.

IMPORTANT:
This is a dataset-based prediction tool, not a diagnostic system.
It cannot establish causation and must not replace professional medical care.

PRIVACY:
The shareable package does not include the patient-level Excel dataset.
