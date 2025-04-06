import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer #
import joblib

data = pd.read_excel("/content/data.xlsx")

X = data.drop(["Defect_level", "Quality_score"], axis=1)
y = data[["Defect_level", "Quality_score"]]

numeric_cols = ["Length", "Width", "Quality", "Size_metric"]
categorical_cols = ["Type"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

X_processed = preprocessor.fit_transform(X)

joblib.dump(preprocessor, "/content/preprocessor.pkl")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_processed.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_processed, y, epochs=50, batch_size=8, validation_split=0.2) # батч

model.save("/content/display_predictor.keras")