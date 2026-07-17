from datetime import datetime
from pathlib import Path
import psycopg2
import pandas as pd


def calc_duration(func):
    """
    Декоратор для измерения времени выполнения функции

    :param func: функция, время выполнения которой нужно измерить
    :return: обертка над функцией с выводом времени выполнения
    """
    def wrapper(*args, **kwargs):
        time_before = datetime.now()
        result = func(*args, **kwargs)
        time_after = datetime.now()

        print(
            f"Время выполнения: "
            f"{str(time_after - time_before).split('.')[0]}"
        )

        return result

    return wrapper


class PostgresDB:
    """
    Класс для работы с базой данных PostgreSQL

    Предоставляет методы для:
    - подключения к базе данных
    - закрытия соединения
    - выполнения SQL-запросов
    - чтения данных в pandas DataFrame
    - загрузки CSV-файлов в таблицы PostgreSQL
    """

    def __init__(self, db_args: dict):
        """
        Инициализация параметров подключения.

        :param db_args: словарь параметров подключения к БД
                        (database, host, port, user, password)
        """
        self.db_args = db_args
        self.conn = None
        self.cursor = None

    def connect(self):
        """
        Устанавливает подключение к PostgreSQL и создает курсор

        :raises Exception: при ошибке подключения
        """
        try:
            self.conn = psycopg2.connect(**self.db_args)
            self.cursor = self.conn.cursor()
            print("Подключение к БД успешно установлено")

        except Exception as error:
            print(f"Ошибка подключения к PostgreSQL: {error}")
            raise

    def close(self):
        """
        Закрывает курсор и соединение с базой данных

        После закрытия ссылки на соединение и курсор
        сбрасываются в None
        """
        try:
            if self.cursor:
                self.cursor.close()
                self.cursor = None

            if self.conn:
                self.conn.close()
                self.conn = None

            print("Подключение к БД закрыто")

        except Exception as error:
            print(f"Ошибка при закрытии подключения: {error}")

    @calc_duration
    def execute_query(self, query: str, params=None):
        """
        Выполняет SQL-запрос без возврата результата

        Используется для:
        - CREATE TABLE
        - ALTER TABLE
        - INSERT / UPDATE / DELETE
        - CREATE INDEX

        :param query: SQL-запрос
        :param params: параметры для параметризованного запроса
        :raises Exception: при ошибке выполнения
        """
        try:
            if self.conn is None or self.cursor is None:
                self.connect()

            self.cursor.execute(query, params)
            self.conn.commit()

            print("Запрос выполнен успешно")

        except Exception as error:
            if self.conn:
                self.conn.rollback()

            print(f"Ошибка при выполнении запроса: {error}")
            raise

    @calc_duration
    def get_df(self, query: str, params=None) -> pd.DataFrame:
        """
        Выполняет SQL-запрос и возвращает результат
        в виде pandas DataFrame

        :param query: SQL-запрос SELECT
        :param params: параметры для параметризованного запроса
        :return: DataFrame с результатом запроса
        :raises Exception: при ошибке чтения данных
        """
        try:
            if self.conn is None:
                self.connect()

            return pd.read_sql(query, self.conn, params=params)

        except Exception as error:
            print(f"Ошибка при чтении данных: {error}")
            raise

    @calc_duration
    def copy_from_csv(
        self,
        table_name: str,
        file_path: str,
        columns: list[str]
    ):
        """
        Загружает данные из CSV-файла в таблицу PostgreSQL

        Используется механизм COPY FROM STDIN,
        который работает значительно быстрее INSERT

        :param table_name: имя таблицы в БД
        :param file_path: путь до CSV-файла
        :param columns: список столбцов таблицы
        :raises Exception: при ошибке загрузки
        """
        try:
            if self.conn is None or self.cursor is None:
                self.connect()

            file_path = Path(file_path)
            columns_sql = ", ".join(columns)

            copy_sql = f"""
                COPY {table_name} ({columns_sql})
                FROM STDIN
                WITH CSV HEADER DELIMITER ','
            """

            with open(file_path, "r", encoding="utf-8") as f:
                self.cursor.copy_expert(copy_sql, f)

            self.conn.commit()

            print(
                f"Данные из {file_path.name} "
                f"успешно загружены в {table_name}"
            )

        except Exception as error:
            if self.conn:
                self.conn.rollback()

            print(f"Ошибка при загрузке CSV: {error}")
            raise