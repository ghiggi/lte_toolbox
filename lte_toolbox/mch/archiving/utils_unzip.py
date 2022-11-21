import pathlib
import datetime
from zipfile import ZipFile


def unzip_mch(input_dir_path: pathlib.Path,
              output_dir_path: pathlib.Path,
              data_start_date: datetime.datetime,
              data_end_date: datetime.datetime = None,
              product: str = "RZC"):
    """Unzip all RZC .zip files for all years starting 2016 and save them
    in an output folder.

    Parameters
    ----------
    input_dir_path : pathlib.Path
        Path to input folder
    output_dir_path : pathlib.Path
        Path to folder where unzipped files will be saved
    data_start_year: int, optional
        Year starting which the data should be unzipped
    """
    if not data_end_date:
        data_end_date = datetime.datetime.today()
    folders = input_dir_path.glob("*")
    for folder_year in sorted(folders):
        year = int(folder_year.name)
        if year >= data_start_date.year:
            print(f"{year}.. ", end="")
            output_year_path = output_dir_path / str(year)
            output_year_path.mkdir(exist_ok=True)
            days = folder_year.glob("*")
            for folder_day in days:
                folder_datetime = datetime.datetime.strptime(folder_day.name, "%y%j")
                if folder_datetime >= data_start_date and folder_datetime <= data_end_date:
                    output_day_path = output_year_path / folder_day.name
                    output_day_path.mkdir(parents=True, exist_ok=True)
                    zip_path = list(folder_day.glob(f"{product}*.zip"))[0]
                    unzip_file(zip_path, output_day_path)
            print("done.")


def unzip_file(zip_path: pathlib.Path, output_path: pathlib.Path):
    """Unzip .zip file and save the foldeer in output_path.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to input folder
    output_path : pathlib.Path
        Path to folder where unzipped files will be saved
    """
    with ZipFile(zip_path, 'r') as zip_ref:
        output_zip_path = output_path / zip_path.stem
        output_zip_path.mkdir(exist_ok=True)
        zip_ref.extractall(output_zip_path)


def unzip_files(input_path: pathlib.Path, output_path: pathlib.Path):
    """Unzip .zip files in input_path and save them in output_path.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to input folder
    output_path : pathlib.Path
        Path to folder where unzipped files will be saved
    """
    for p in sorted(input_path.glob("*.zip")):
        with ZipFile(p, 'r') as zip_ref:
            zip_name = p.name[:-4]
            output_zip_path = output_path / zip_name
            output_zip_path.mkdir(exist_ok=True)
            zip_ref.extractall(output_zip_path)
