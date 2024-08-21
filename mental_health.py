import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
import warnings
import pickle

warnings.filterwarnings("ignore")

data = pd.read_csv("encoded_data (1).csv", encoding="latin-1")


data = np.array(data)

X = data[1:, 1:-1].astype(int)
y = data[1:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()

stack = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)
stack.fit(X_train, y_train)

# Save the model to disk
pickle.dump(stack, open('model.pkl', 'wb'))

# Load the model from disk
model = pickle.load(open('model.pkl', 'rb'))

# Example: Use the loaded model for prediction
example_input = np.array([[1, 0, 1, 0, 0, 1, 1,]])  # Example input data for prediction
predicted_class = model.predict(example_input)
print("Predicted class:", predicted_class)
