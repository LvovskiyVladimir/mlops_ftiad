import pandas as pd


DATA_PATH = "postgres/insurance.csv"
COLUMN_NAMES = {'age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'}

# проверяет, что на вход подается именно .csv файл
def test_file_format():
    assert DATA_PATH.endswith('.csv')

# проверяет корректность имён столбцов в датасете
def test_column_names():
    with open(DATA_PATH, 'r') as f:
        lines = f.read().split('\n')
    column_names = set(lines[0].split(','))
    assert column_names.intersection(COLUMN_NAMES) == COLUMN_NAMES

# проверяет, что датасет не пустой
def test_not_empty():
    with open(DATA_PATH, 'r') as f:
        lines = f.read().split('\n')
    assert len(lines) > 1

# проверяет, что в датасете нет пропущенных значений
def test_nan():
    df = pd.read_csv(DATA_PATH)
    assert df.isna().sum().sum() == 0

# проверяет корректность типов данных в датасете
def test_data_types():
    df = pd.read_csv(DATA_PATH)
    assert df.dtypes['sex'] == 'object'
    assert df.dtypes['bmi'] == 'float64'
    assert df.dtypes['children'] == 'int64'
    assert df.dtypes['smoker'] == 'object'
    assert df.dtypes['region'] == 'object'
    assert df.dtypes['charges'] == 'float64'
