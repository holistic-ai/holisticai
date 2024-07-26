from holisticai.datasets import load_dataset

def test_load_dataset():
    SHARD_SIZE = 50
    dataset = load_dataset("adult", protected_attribute='sex').sample(1000)
    dataset = dataset.map(lambda x: {'new': str(x['group_a'])  + str(x['y'])}, vectorized=False)
    dataset = dataset.groupby(['y','group_a']).head(SHARD_SIZE)
    dataset = dataset.train_test_split(test_size=0.2, stratify=dataset['y'], random_state=0)


def test_map():
    dataset = load_dataset('us_crime')
    dataset = dataset.train_test_split(test_size=0.2, random_state=42)
    test = dataset['test']
    new_test = test.map(lambda sample: {'group': sample['y']}, vectorized=False)
    assert list(new_test['X'].columns) == list(test['X'].columns)