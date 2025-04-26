from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import pickle
import pandas as pd

instrument_df=pd.read_csv("src/models/ins_data.csv")
instrument_df.head(5)
X=instrument_df.drop(["instrument"],axis=1)
Y=instrument_df["instrument"]

X_train1, X_test1, Y_train1, Y_test1=train_test_split(X, Y, test_size=0.3, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train1, Y_train1)

y_pred = model.predict(X_test1)
print(y_pred)

df = pd.DataFrame({'y_pred': y_pred,
                   'Y_test1': Y_test1})
print(df)

print(f'accuracy: {accuracy_score(Y_test1, y_pred) :.3}')

with open('instrument_pickle_file.pkl','wb') as pkl:
    pickle.dump(model, pkl)