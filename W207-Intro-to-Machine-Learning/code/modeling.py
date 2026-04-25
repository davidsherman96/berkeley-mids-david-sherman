from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def create_models(random_state=1234):
    baseline = DummyClassifier()
    lr_model = LogisticRegressionCV(random_state=random_state)
    knn_model = KNeighborsClassifier()
    dt_model = DecisionTreeClassifier(random_state=random_state)
    random_forest_model = ensemble.RandomForestClassifier(
        random_state=random_state
    )
    adaboost_model = ensemble.AdaBoostClassifier(
        random_state=random_state
    )
    gradient_boosting_model = ensemble.GradientBoostingClassifier(
        random_state=random_state
    )
    estimators = [
        ('lr_model', lr_model),
        ('knn_model', knn_model),
        ('dt_model', dt_model),
        ('random_forest_model', random_forest_model),
        ('adaboost_model', adaboost_model),
        ('gradient_boosting_model', gradient_boosting_model)
    ]
    voting_hard_model = ensemble.VotingClassifier(
        estimators=estimators,
        voting='hard'
    )
    voting_soft_model = ensemble.VotingClassifier(
        estimators=estimators,
        voting='soft'
    )

    return {
        'baseline': baseline,
        'lr_model': lr_model,
        'knn_model': knn_model,
        'dt_model': dt_model,
        'random_forest_model': random_forest_model,
        'adaboost_model': adaboost_model,
        'gradient_boosting_model': gradient_boosting_model,
        'voting_hard_model': voting_hard_model,
        'voting_soft_model': voting_soft_model
    }


def run_experiments(models, X_train, y_train, X_val, y_val):
    report = {}
    for name, model in models.items():
        print(f"\n\nFitting {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        print(classification_report(y_val, predictions))
        report[name] = classification_report(y_val, predictions, output_dict=True)
    return report


def evaluate_model(model, X, y):
    predictions = model.predict(X)

    print(classification_report(y, predictions))

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true=y, y_pred=predictions),
        display_labels=["Negative", "Positive"]
    ).plot()

    return X[predictions != y]
