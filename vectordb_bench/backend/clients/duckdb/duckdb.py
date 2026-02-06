from contextlib import contextmanager
from typing import Generator, Any

import pandas as pd

from ..api import VectorDB, FilterOp, Filter, IndexType
import duckdb
from .config import CreateIndex
from .duckdb_utils import SetDuckDBThreadsTo
from .config import DuckDBConnectionConfigDict, DuckDBCaseConfig


class DuckDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: DuckDBConnectionConfigDict,
        db_case_config: DuckDBCaseConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "DuckDB"

        self.dims = dim
        self.conn_config = db_config
        self.case_config = db_case_config
        self.drop_old = drop_old
        self.with_predicate_column = with_scalar_labels
        self.kwargs = kwargs

        self.id_column_name = "id"
        self.embedding_column_name = "embedding"
        self.embedding_column_element_type = "FLOAT"
        self.predicate_column_name = "label"
        self.where_clause = ""

        self.conn = self._create_connection(read_only=False)

        if self.drop_old:
            if self.case_config.create_index != CreateIndex.NEVER:
                self._drop_index()
            self._drop_table()
            self._create_table(self.dims)
            if (
                self.case_config.create_index == CreateIndex.BEFORE_INSERT
                and self.case_config.index != IndexType.PDXEARCH
            ):
                with SetDuckDBThreadsTo(self.conn, self.case_config.duckdb_threads_during_index_creation):
                    self._drop_index()
                    self._create_index()

        self.conn.close()
        self.conn = None

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        self.conn = self._create_connection(read_only=False)
        self.conn.execute("PRAGMA disable_progress_bar;")

        if self.case_config.index == IndexType.PDXEARCH or self.case_config.index == IndexType.HNSW:
            # Disable DuckDB's late materialization query optimization as the
            # PDXearch and VSS extensions do not handle it yet, leading to
            # suboptimal query plans when K <= 50.
            self.conn.execute("SET late_materialization_max_rows = 0;")

        # Load extension
        if self.case_config.index == IndexType.PDXEARCH:
            self._load_pdxearch_extension(self.case_config.extension_path)

            # Temporary, while index persistence does not work for PDXearch.
            with SetDuckDBThreadsTo(self.conn, self.case_config.duckdb_threads_during_index_creation):
                self._drop_index()
                self._create_index()

        if self.case_config.index == IndexType.HNSW:
            self._load_vss_extension()

        # Warmup query to ensure the table and extension index are loaded.
        self.prepare_filter(Filter(type=FilterOp.NonFilter))
        self.search_embedding([0.0] * self.dims)

        try:
            yield
        finally:
            self.conn.close()
            self.conn = None

    def _create_connection(self, read_only: bool):
        # Note: We don't support concurrent connections currently.
        config = {"allow_unsigned_extensions": "true"}
        if self.conn_config["duckdb_threads"] is not None:
            config["threads"] = self.conn_config["duckdb_threads"]

        return duckdb.connect(
            config=config,
            database=self.conn_config["database_name"],
            read_only=read_only,
        )

    def _load_pdxearch_extension(self, extension_path: str):
        self.conn.execute(f"LOAD '{extension_path}'")
        self.conn.commit()

    def _load_vss_extension(self):
        self.conn.execute("INSTALL vss")
        self.conn.execute("LOAD vss")
        self.conn.commit()

    # ----------------------------------
    # Table creation
    # ----------------------------------
    def _create_table(self, dims: int):
        assert self.conn is not None
        assert self.dims == dims

        create_table_sql = (
            f"""CREATE TABLE {self.conn_config['table_name']}
                ({self.id_column_name} INTEGER,
                {self.embedding_column_name} {self.embedding_column_element_type}[{self.dims}],
                {self.predicate_column_name} VARCHAR(64));"""
            if self.with_predicate_column
            else f"""CREATE TABLE {self.conn_config['table_name']}
                ({self.id_column_name} INTEGER,
                {self.embedding_column_name} {self.embedding_column_element_type}[{self.dims}]);"""
        )

        self.conn.execute(create_table_sql)
        self.conn.commit()

    def _drop_table(self):
        assert self.conn is not None

        self.conn.execute(f"DROP TABLE IF EXISTS {self.conn_config['table_name']}")
        self.conn.commit()

    # ----------------------------------
    # Index creation
    # ----------------------------------
    def _drop_index(self):
        assert self.conn is not None

        self.conn.execute(f"DROP INDEX IF EXISTS {self.case_config.index_name}")
        self.conn.commit()

    def _format_sql_value(self, value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return f"'{value}'"
        return str(value)

    def _create_index(self):
        assert self.conn is not None

        # Temporary workaround for PDXearch to make _create_index a noop if the table has not been populated yet.
        # This can be removed and the other code refactored once PDXearch has implemented index persistence.
        if self.case_config.index == IndexType.PDXEARCH:
            res = self.conn.execute(f"SELECT COUNT(*) FROM {self.conn_config['table_name']}")
            if res.fetchone()[0] == 0:
                return

        # https://duckdb.org/docs/stable/core_extensions/vss#persistence
        if self.case_config.index == IndexType.HNSW:
            self.conn.execute(f"SET hnsw_enable_experimental_persistence = true;")

        index_param = self.case_config.index_param()
        index_options = ", ".join(
            [
                f"{k}={self._format_sql_value(v)}"
                for opt_dict in index_param["index_options"]
                for k, v in opt_dict.items()
            ]
        )
        with_clause = f"WITH ({index_options})" if index_options else ""
        create_index_sql = f"""CREATE INDEX {index_param['index_name']} ON {self.conn_config['table_name']}
                USING {index_param['index_type']} ({self.embedding_column_name}) {with_clause}"""
        self.conn.execute(create_index_sql)
        self.conn.commit()

    def optimize(self, data_size: int | None = None):
        if self.case_config.create_index == CreateIndex.AFTER_INSERT and self.case_config.index != IndexType.PDXEARCH:
            with SetDuckDBThreadsTo(self.conn, self.case_config.duckdb_threads_during_index_creation):
                # For PDXearch this is useless, as the index is not persisted, so we have to drop and recreate it before
                # the search occurs when it calls init() again (as in between it will drop the connection). Before
                # running this optimize it will already call init() so we skip this for PDXearch and it will be timed
                # correctly anyway.
                self._drop_index()
                self._create_index()

    # ----------------------------------
    # Populate table
    # ----------------------------------
    def insert_embeddings(
        self, embeddings: list[list[float]], metadata: list[int], labels_data: list[str] | None = None, **kwargs: Any
    ) -> tuple[int, Exception | None]:
        assert self.conn is not None
        if self.with_predicate_column:
            assert labels_data is not None

        try:
            df = pd.DataFrame(
                {
                    self.id_column_name: metadata,
                    self.embedding_column_name: embeddings,
                }
            )
            if self.with_predicate_column:
                df[self.predicate_column_name] = labels_data

            # https://duckdb.org/docs/stable/clients/python/data_ingestion#directly-accessing-dataframes-and-arrow-objects
            temp_view_name = "temp_view"
            self.conn.register(temp_view_name, df)
            self.conn.execute(f"INSERT INTO {self.conn_config['table_name']} FROM {temp_view_name}")
            self.conn.commit()

            return len(metadata), None
        except Exception as e:
            return 0, e

    # ----------------------------------
    # Filtered and non-filtered search
    # ----------------------------------
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.where_clause = ""
        elif filters.type == FilterOp.NumGE:
            self.where_clause = f"WHERE {self.id_column_name} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.where_clause = f"WHERE {self.predicate_column_name} = '{filters.label_value}'"
        else:
            raise ValueError(f"Unsupported filter type: {filters.type}")
        return

    def search_embedding(self, query: list[float], K: int = 100, **kwargs):
        array_function_name = self.case_config._metric_type_to_function_name()

        # The WHERE clause is set by the `prepare_filter` method above.
        result = self.conn.execute(
            f"""SELECT {self.id_column_name} FROM {self.conn_config['table_name']} {self.where_clause}
                ORDER BY {array_function_name}({self.embedding_column_name},{query}::{self.embedding_column_element_type}[{self.dims}])
                LIMIT {K};"""
        )

        return [int(i[0]) for i in result.fetchall()]
