# compare_classifiers.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap

# --- Load dataset ---
df = pd.read_csv('dataset.csv')

# Target column
target_col = 'type'

# Encode target labels
label_encoder = LabelEncoder()
df[target_col] = label_encoder.fit_transform(df[target_col])

# Features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Normalize features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

results = {}
os.makedirs("visualizations", exist_ok=True)

# Train & evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = accuracy_score(y_test, preds)

# --- Visualization 1: Accuracy bar chart ---
plt.figure()
plt.bar(results.keys(), results.values(), color='skyblue')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/accuracy_bar.png")

# --- Visualization 2: Confusion matrix (Random Forest) ---
rf_model = models["Random Forest"]
rf_preds = rf_model.predict(X_test)
cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.savefig("visualizations/confusion_matrix_rf.png")

# --- Visualization 3: Decision tree plot ---
plt.figure(figsize=(12, 8))
plot_tree(
    models["Decision Tree"],
    feature_names=X.columns,
    class_names=label_encoder.classes_,
    filled=True
)
plt.title("Decision Tree Visualization")
plt.savefig("visualizations/decision_tree.png")

# --- Visualization 4: Decision boundary (KNN on 2 features) ---
X2 = X_scaled[['height', 'width']]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, random_state=42
)

knn_model = KNeighborsClassifier()
knn_model.fit(X2_train, y2_train)

h = 0.02
x_min, x_max = X2.iloc[:, 0].min() - 1, X2.iloc[:, 0].max() + 1
y_min, y_max = X2.iloc[:, 1].min() - 1, X2.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])

plt.figure()
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
plt.scatter(X2.iloc[:, 0], X2.iloc[:, 1], c=y, edgecolor="k",
            cmap=ListedColormap(["red", "green", "blue"]))
plt.title("Decision Boundary (KNN)")
plt.xlabel("height")
plt.ylabel("width")
plt.savefig("visualizations/decision_boundary_knn.png")

# --- Visualization 5: Decision boundary (Logistic Regression on 2 features) ---
log_model = LogisticRegression(max_iter=200)
log_model.fit(X2_train, y2_train)

Z_log = log_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_log = Z_log.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, Z_log, cmap=cmap_light, alpha=0.8)
plt.scatter(X2.iloc[:, 0], X2.iloc[:, 1], c=y, edgecolor="k",
            cmap=ListedColormap(["red", "green", "blue"]))
plt.title("Decision Boundary (Logistic Regression)")
plt.xlabel("height")
plt.ylabel("width")
plt.savefig("visualizations/decision_boundary_logreg.png")

print("✅ All plots saved in 'visualizations/' folder")
