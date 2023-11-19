"""Generate artificial data and get random datasets."""
from data_generation.RandomKaggleDataset import RandomKaggleDataset
from data_generation.FakeDataGenerator import FakeDataGenerator


def generate_data(
        save_dir_path='data',
        fake_data_count=100,
        random_queris=10,
        table_per_query=5
        ):
    """Generate artificial data and get random datasets."""
    fakegenerator_ru = FakeDataGenerator(locale='ru_RU',
                                         fake_data_count=fake_data_count//2
                                         )
    fakegenerator_ru.create_random_data(row_max=300)

    fakegenerator_us = FakeDataGenerator(locale='en_US',
                                         fake_data_count=fake_data_count//2
                                         )
    fakegenerator_us.create_random_data(row_max=300)

    rnd_kag_dat = RandomKaggleDataset(random_queris,
                                      table_per_query,
                                      save_dir_path)
    rnd_kag_dat.pipeline()
