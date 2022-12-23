import pandas as pd
from hw3.model import Model

DATA_PATH = "postgres/insurance.csv"
MODEL_NAMES = ['linear', 'gradboost']

# проверяет обучение модели
def test_train():
    for model_name in MODEL_NAMES:
        model = Model(model_name)
        model.fit(DATA_PATH)
        assert model.train_score > 0.75

# проверяет сохранение и загрузку модели
def test_dump_load():
    for model_name in MODEL_NAMES:
        model = Model(model_name)
        model.fit(DATA_PATH)
        train_score = model.train_score
        model.save_model('temp.pickle')
        model = Model(model_name)
        model.load_model('temp.pickle')
        model.test(DATA_PATH)
        assert model.test_score == train_score

def test_predict():
    for model_name in MODEL_NAMES:
        model = Model(model_name)
        model.fit(DATA_PATH)
        pred = model.predict({'age': 23, 'bmi': 28.5, 'children': 0, 'sex': 'm', 'smoker': 'false', 'region': 'northwest'})
        assert pred > 0
        assert not pd.isna(pred)
