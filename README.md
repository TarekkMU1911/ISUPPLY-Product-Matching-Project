# User Guide for Making Predictions

## Installation Requirements

Before running the project, ensure you have installed all the necessary libraries. You can install them using the following command:

```bash
pip install numpy pandas xgboost scikit-learn tensorflow scipy nltk matplotlib seaborn arabic-reshaper python-bidi wordcloud
```

Additionally, download the NLTK stopwords:

```python
import nltk
nltk.download('stopwords')
```

## Steps to Use the Model:

1. **Load the Dataset** for training in the **Reading Training Dataset** section **(cell number 2)**.
2. **Select the font path** based on your OS and **comment the other option** in the **Visualizing Most Repeated Products** section **(cell number 13)**.
3. **Load the test dataset** in the **Predictions For Test Set** section **(cell number 30)**.
4. **This Python Notebook will generate:**
   - **The trained model** (saved as `.h5` file)
   - **Model predictions** on the validation dataset
   - **Model predictions** on the test dataset

By following these steps, users can efficiently utilize the trained model for **accurate product matching**.

## Required Libraries

Ensure you have the following libraries installed in your environment. You can install them using pip as mentioned above.

```python
import numpy as np
import pandas as pd
import re
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from scipy.sparse import csr_matrix
import scipy
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import arabic_reshaper
from bidi.algorithm import get_display
from wordcloud import WordCloud
from collections import Counter
from matplotlib import font_manager
import warnings
warnings.filterwarnings("ignore")
```

## How It Works?

This notebook processes product matching using a deep learning model. It leverages NLP techniques for text vectorization, feature extraction, and classification. Follow the steps above to ensure proper execution.






## And you can take a look through our power point presentation from the following link to get a better understanding of the project and how it works:
https://docs.google.com/presentation/d/1uvlB1GkI0fr79_Zyu52x0CJWHS8nvzi2/edit?usp=sharing&ouid=117402692951599971689&rtpof=true&sd=true
