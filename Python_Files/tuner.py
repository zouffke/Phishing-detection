from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Tuner:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.best_model_prediction = None
        self.default_model_prediction = None
        self.best_model = None
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.default_model = model.fit(X_train, y_train)

        super().__init__()

    def tune(self, param_grid, cv=5, scoring='accuracy', n_jobs=-1, **kwargs):
        grid = GridSearchCV(self.model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, **kwargs)
        grid.fit(self.X_train, self.y_train)
        self.best_model = grid.best_estimator_

    def __print_evaluate(self, model_name: str, model, model_type, prediction):
        print(f'{model_type} {model_name} accuracy:\n\t{accuracy_score(self.y_test, prediction)}')
        print(f'{model_type} {model_name} Report:\n{classification_report(self.y_test, prediction)}')

    def evaluate(self, model: str = 'default', model_name='model'):
        if model == 'default':
            self.default_model_prediction = self.default_model.predict(self.X_test)
            self.__print_evaluate(model_name, self.default_model, 'Default', self.default_model_prediction)
        elif model == 'best' and self.best_model is not None:
            self.best_model_prediction = self.best_model.predict(self.X_test)
            self.__print_evaluate(model_name, self.best_model, 'Best', self.best_model_prediction)
        else:
            raise Exception

    def __print_cross_validation(self, model, cv, scoring, n_jobs, model_name, model_type):
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scoring, n_jobs=n_jobs)
        print(f'{model_type} {model_name} Cross-Validation accuracy:\n\t{scores.mean()}')

    def cross_validation(self, model='default', cv=5, scoring='accuracy', n_jobs=-1, model_name='model'):
        if model == 'default':
            self.__print_cross_validation(model=self.default_model,
                                          cv=cv,
                                          scoring=scoring,
                                          n_jobs=n_jobs,
                                          model_name=model_name,
                                          model_type='Default')
        elif model == 'best' and self.best_model is not None:
            self.__print_cross_validation(model=self.best_model,
                                          cv=cv, scoring=scoring,
                                          n_jobs=n_jobs,
                                          model_name=model_name,
                                          model_type='Best')
        else:
            raise Exception

    def __show_confusion_matrix(self, prediction, model_name, model_type):
        default_cm = confusion_matrix(self.y_test, prediction)
        plt.figure(figsize=(10, 7))
        sns.heatmap(default_cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_type} {model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def confusion_matrix(self, model='default', model_name='model'):
        if model == 'default':
            self.__show_confusion_matrix(self.default_model_prediction, model_name, "Default")
        elif model == 'best' and self.best_model is not None:
            self.__show_confusion_matrix(self.best_model_prediction, model_name, "Best")
        else:
            raise Exception

    def save_best_model(self, model_path):
        joblib.dump(self.best_model, model_path)
