# Удаляет из test-data те датасеты, которые есть в data.
import os
import shutil


def disintersect_folders(*, check_intersectoin_path, source_path) -> None:
    """
    Функция удаляет из директории source_path папки, которые есть в
    check_intersectoin_path.
    """
    folders_in_path1 = set(os.listdir(check_intersectoin_path))
    folders_in_path2 = set(os.listdir(source_path))

    common_folders = folders_in_path1.intersection(folders_in_path2)

    for folder in common_folders:
        folder_path = os.path.join(source_path, folder)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            print(f'Deleted: {folder_path}')
