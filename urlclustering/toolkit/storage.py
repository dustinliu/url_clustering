import sqlite3

raw_data_create = '''
create table raw_data
(
  id        integer constraint raw_data_pk primary key autoincrement,
  word      text,
  url       text,
  frequency real,
  amount    real,
  label     int default 0
);
'''
raw_data_insert = '''
insert into raw_data (word, url, frequency, amount) values (?, ?, ?, ?)
'''


class Storage:
    def __init__(self, db_dir='./data'):
        self._db_dir = db_dir
        self._conn = sqlite3.connect(f'{db_dir}/urlclustering.db')

    def init_tables(self):
        cursor = self._conn.cursor()
        cursor.execute(raw_data_create)

    def write_raw_data(self, word, url, frequency, amount):
        cursor = self._conn.cursor()
        cursor.execute(raw_data_create, word, url, frequency, amount)

    def batch_write_raw_data(self, raw_data):
        cursor = self._conn.cursor()
        cursor.executemany(raw_data_insert, raw_data)
        self._conn.commit()


if __name__ == '__main__':
    Storage().init_tables()
