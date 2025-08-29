# train.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def main():
    data = load_iris()
    X, y = data.data, data.target
    target_names = list(data.target_names)

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=200)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    print(f"Test accuracy: {acc:.3f}")

    # Save pipeline + labels together
    artifact = {"pipeline": pipe, "labels": target_names}
    joblib.dump(artifact, "model.joblib")
    print("Saved model to model.joblib")

if __name__ == "__main__":
    main()
