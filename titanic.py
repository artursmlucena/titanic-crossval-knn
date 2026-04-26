import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score

# polishing data
file = pd.read_csv('titanic.csv', encoding= 'latin1', decimal = '.', sep = ',')
file_clean = file[["Survived", "Pclass", "Sex", "Age"]]
file_clean["Age"] = file_clean["Age"].fillna(file_clean["Age"].mean())
file_clean["Sex"] = file_clean["Sex"].map({"male": 0, "female" : 1})


predict = file_clean[["Age", "Pclass", "Sex"]]
results = file_clean["Survived"]

# tuning the model using cross valodation and plotting the graph
import matplotlib.pyplot as plt
scores = []
k_s = []
for k in range(1, 6, 1):
    knn = KNN(n_neighbors = k)
    score = cross_val_score(knn, predict, results, cv=5)
    scores.append(score.mean())
    k_s.append(k)

plt.xlabel("K (vizinhos)")
plt.ylabel("Accuracy")
plt.title("KNN - Cross Validation")
plt.plot(k_s, scores)
plt.savefig('grafico.png')








