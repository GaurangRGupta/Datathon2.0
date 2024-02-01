data = pd.read_csv(r"C:\Users\gaura\PycharmProjects\FloodPredictor\dataset.csv")
data.head()

data.apply(lambda x: sum(x.isnull()), axis=0)

data['FLOODS'].replace(['YES', 'NO'], [1, 0], inplace=True)

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
c = data[['JUN', 'JUL', 'AUG', 'SEP']]

x = data[c.columns].values
y = data['ANNUAL RAINFALL'].values

from sklearn import preprocessing

minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
minmax.fit(x).transform(x)

from sklearn import model_selection, neighbors
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train_std = minmax.fit_transform(x_train)
x_test_std = minmax.transform(x_test)

print(y_train.shape)
print(y_test.shape)
print(x_train_std.shape)

num_bins = 5
bins = np.linspace(y_train.min(), y_train.max(), num_bins + 1)
y_train_discrete = pd.cut(y_train, bins=bins, labels=False, include_lowest=True)
print("Bin Edges:", bins)
print("Discrete Values:\n", y_train_discrete)

num_bins = 5
bins = np.linspace(y_test.min(), y_test.max(), num_bins + 1)
y_test_discrete = pd.cut(y_test, bins=bins, labels=False, include_lowest=True)
print("Bin Edges:", bins)
print("Discrete Values:\n", y_test_discrete)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(multi_class='ovr')
lr_clf = lr.fit(x_train_std, y_train_discrete)

lr_accuracy = cross_val_score(lr_clf, x_test_std, y_test_discrete, cv=3, scoring='accuracy', n_jobs=-1)

lr_accuracy.mean()

y_predict = lr_clf.predict(x_test_std)
print('Predicted chances of flood')
print(y_predict)

from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

y_predict = lr_clf.predict(x_test_std)
y_predict_discrete = pd.cut(y_predict, bins=bins, labels=False, include_lowest=True)
print("\naccuracy score: %f" % (accuracy_score(y_test_discrete, y_predict) * 100))
print("recall score: %f" % (recall_score(y_test_discrete, y_predict, average='weighted') * 100))
print("roc score: %f" % (roc_auc_score(y_test_discrete, y_predict, average='weighted') * 100))

import seaborn as sns

axis = sns.barplot(x='Name', y='Score', data=tr_split)
axis.set(xlabel='Classifier', ylabel='Accuracy')
for p in axis.patches:
    height = p.get_height()

    axis.text(p.get_x() + p.get_width() / 2, height + 0.005, '{:1.4f}'.format(height), ha="center")
plt.show()
