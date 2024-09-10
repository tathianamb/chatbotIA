import mysql.connector
from langchain.schema import Document
import logging


class MySQLLoader:
    def __init__(self, query, host, port, user, password, database):
        self.query = query
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

    def _connect(self):
        return mysql.connector.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )

    def load(self):
        logging.info("Connecting to MySQL database.")
        connection = self._connect()
        cursor = connection.cursor(dictionary=True)

        try:
            logging.info("Executing query.")
            cursor.execute(self.query)
            rows = cursor.fetchall()

            logging.info(f"Fetched {len(rows)} rows from the database.")
            documents = []
            for row in rows:
                content = " ".join([f"{key}: {value}" for key, value in row.items()])
                documents.append(Document(page_content=content))

            return documents

        finally:
            cursor.close()
            connection.close()
            logging.info("MySQL connection closed.")
