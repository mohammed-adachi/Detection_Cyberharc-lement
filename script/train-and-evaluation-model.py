import joblib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, chi2

# Load your data here.  Replace 'X_train' and 'y_train' with your actual data.
# Example:
# from sklearn.datasets import fetch_20newsgroups
# newsgroups_train = fetch_20newsgroups(subset='train')
# X_train = newsgroups_train.data
# y_train = newsgroups_train.target


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)),
    ('feature_selection', SelectKBest(chi2, k=5000)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

param_grid = {
    'feature_selection__k': [3000, 5000],
    'clf__C': [0.1, 1.0, 10.0],
}

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Early stopping
best_score = grid_search.best_score_
val_score = best_model.score(X_val, y_val)
if val_score < best_score * 0.95:
    print("Warning: Possible overfitting. Consider reducing model complexity.")

model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"model_{model_version}.joblib"
joblib.dump(best_model, model_filename)
print(f"Model saved as {model_filename}")

