import neptune
from joblib import dump
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

neptune.init(api_token='ANONYMOUS',
             project_qualified_name='shared/onboarding')

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                    test_size=0.4, random_state=1234)

params = {'n_estimators': 10,
          'max_depth': 3,
          'min_samples_leaf': 1,
          'min_samples_split': 2,
          'max_features': 3,
          'random_state': 1234
          }

neptune.create_experiment(name='great-idea', # name experiment
                          params=params,  # log parameters
                          upload_source_files=['*.py', 'requirements.txt']  # log source and environment
                          )
neptune.append_tag(['experiment-organization', 'me'])  # organize things

clf = RandomForestClassifier(**params)
clf.fit(X_train, y_train)
y_train_pred = clf.predict_proba(X_train)
y_test_pred = clf.predict_proba(X_test)

train_f1 = f1_score(y_train, y_train_pred.argmax(axis=1), average='macro')
test_f1 = f1_score(y_test, y_test_pred.argmax(axis=1), average='macro')
print(f'Train f1:{train_f1} | Test f1:{test_f1}')

neptune.log_metric('train_f1', train_f1)  # log metrics
neptune.log_metric('test_f1', test_f1)  # log metrics

dump(clf, 'model.pkl')
neptune.log_artifact('model.pkl')  # log files
