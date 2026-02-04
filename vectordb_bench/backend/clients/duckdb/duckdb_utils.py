import duckdb


class SetDuckDBThreadsTo:
    """Temporarily set the number of DuckDB threads."""

    def __init__(self, conn: duckdb.DuckDBPyConnection, new_threads: int | None):
        self.conn = conn
        self.active = False

        if new_threads is not None:
            self.active = True
            self.new_threads = new_threads
            # Retrieve the maximum number of threads DuckDB is currently allowed to use.
            self.old_threads = conn.execute("SELECT current_setting('threads')").fetchone()[0]

    def __enter__(self):
        if self.active:
            self.conn.execute(f"SET threads = {self.new_threads}")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.active:
            self.conn.execute(f"SET threads = {self.old_threads}")
