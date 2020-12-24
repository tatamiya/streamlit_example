from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import streamlit as st


@st.cache
def get_iris_data():
    return datasets.load_iris()


st.title("Iris with Random Forest")

iris = get_iris_data()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17)

max_depth = 5

n_estimators = st.slider("n_estimators", min_value=1, max_value=1000, value=100)
max_depth = st.slider("max_depth", min_value=1, max_value=20, value=3)
min_samples_split = st.slider("min_samples_split", min_value=2, max_value=100, value=2)

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    random_state=43,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = metrics.f1_score(y_test, y_pred, average="micro")

st.write("f1-score: ", score)


st.dataframe(metrics.classification_report(y_test, y_pred, output_dict=True))
