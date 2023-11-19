"""Data generator class."""
from faker import Faker
import pandas as pd
import random
from typing import Optional, Union, Sequence, Dict
import inspect
import os


class FakeDataGenerator:
    """Class for generate artificial DWH data."""

    def __init__(self,
                 fake_data_count=100,
                 save_dir_path='data',
                 seed=None,
                 locale: Optional[
                     Union[str, Sequence[str],
                           Dict[str, Union[int, float]]]
                     ] = 'ru_RU',
                 ) -> None:
        """Init faker."""
        self.fake_data_count = fake_data_count
        self.save_dir_path = save_dir_path
        self.locale = locale
        self.fake = Faker(
            locale=locale,
            seed=seed
            )

    def create_random_data(self,
                           column_min=3,
                           column_max=10,
                           row_min=300,
                           row_max=1000):
        """Create and save fake data using Faker."""
        # Get all methods that we can use to fake
        all_providers = []
        for attr in dir(self.fake):
            if 'local' in attr or '__' in attr or 'random' in attr or\
                attr in ['tsv', 'csv', 'pcv',
                         'fixed_width', 'json', 'dsv'] or \
                ('ru_RU' in self.fake.locales and
                    attr in ['suffix_male', 'suffix_female',
                             'suffix_nonbinary', 'suffix']):
                continue
            try:
                if inspect.ismethod(getattr(self.fake, attr)) and (
                            isinstance(getattr(self.fake, attr)(), str)
                            ):
                    all_providers.append(attr)
            except Exception:
                #  Calling `attr` on instances maybe deprecated.
                #  In ismethod we can take some bad methods.
                continue
        for table_index in range(0, self.fake_data_count):
            # Get random columns, rows count
            num_columns = random.randint(column_min, column_max)
            num_rows = random.randint(row_min, row_max)
            table_structure = random.sample(all_providers, num_columns)

            # Generate data for every column
            # TODO: encode columns
            data = {column: [getattr(self.fake, column)()
                             for _ in range(num_rows)]
                    for column in table_structure}

            df = pd.DataFrame(data)

            # Save to csv file
            # TODO: encode file_name
            file_name = f"random_fake_table_{table_index}_{self.locale}.csv"
            save_pth = os.path.join(self.save_dir_path, file_name)
            df.to_csv(save_pth, index=False)

            print(f"Table {table_index} saved to file: {save_pth}")


if __name__ == '__main__':
    dg = FakeDataGenerator(real_data_count=50, fake_data_count=10)
    dg.create_fake_data()
