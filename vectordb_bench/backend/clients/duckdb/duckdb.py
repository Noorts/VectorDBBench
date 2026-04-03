from contextlib import contextmanager
from typing import Generator, Any

import logging

import pandas as pd

from ..api import VectorDB, FilterOp, Filter, IndexType
import duckdb
from .config import CreateIndex
from .duckdb_utils import SetDuckDBThreadsTo
from .config import DuckDBConnectionConfigDict, DuckDBCaseConfig

log = logging.getLogger(__name__)


class DuckDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: DuckDBConnectionConfigDict,
        db_case_config: DuckDBCaseConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        predicate_column_type: str = "VARCHAR(64)",
        **kwargs,
    ):
        self.name = "DuckDB"

        self.dims = dim
        self.conn_config = db_config
        self.case_config = db_case_config
        self.drop_old = drop_old
        self.with_predicate_column = with_scalar_labels
        self.predicate_column_type = predicate_column_type
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

        # Set session parameters
        session_statements = [
            f"SET {k} = {self._format_sql_value(v)};"
            for opt_dict in self.case_config.session_param()["session_options"]
            for k, v in opt_dict.items()
        ]
        if session_statements:
            log.debug(f"Setting runtime session parameters:")
            for statement in session_statements:
                log.debug(f"  {statement}")
                self.conn.execute(statement)
            self.conn.commit()

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
        load_extension_sql = f"LOAD '{extension_path}'"
        log.debug(f"Loading PDXearch extension from {extension_path}")
        self.conn.execute(load_extension_sql)
        self.conn.commit()

    def _load_vss_extension(self):
        extension_path = getattr(self.case_config, "extension_path", None)
        if extension_path:
            log.debug(f"Loading VSS extension from {extension_path}")
            self.conn.execute(f"LOAD '{extension_path}'")
        else:
            log.debug("Installing and loading VSS extension")
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
            f"CREATE TABLE {self.conn_config['table_name']} "
            + f"({self.id_column_name} INTEGER, "
            + f"{self.embedding_column_name} {self.embedding_column_element_type}[{self.dims}], "
            + f"{self.predicate_column_name} {self.predicate_column_type});"
            if self.with_predicate_column
            else f"CREATE TABLE {self.conn_config['table_name']} "
            + f"({self.id_column_name} INTEGER, "
            + f"{self.embedding_column_name} {self.embedding_column_element_type}[{self.dims}]);"
        )
        log.debug(f"Creating table: {create_table_sql}")
        self.conn.execute(create_table_sql)
        self.conn.commit()

    def _drop_table(self):
        assert self.conn is not None

        drop_table_sql = f"DROP TABLE IF EXISTS {self.conn_config['table_name']}"
        log.debug(f"Dropping table: {drop_table_sql}")
        self.conn.execute(drop_table_sql)
        self.conn.commit()

    # ----------------------------------
    # Index creation
    # ----------------------------------
    def _drop_index(self):
        assert self.conn is not None

        drop_index_sql = f"DROP INDEX IF EXISTS {self.case_config.index_name}"
        log.debug(f"Dropping index: {drop_index_sql}")
        self.conn.execute(drop_index_sql)
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
                log.debug("Table has not been populated yet, skipping index creation")
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
        create_index_sql = (
            f"CREATE INDEX {index_param['index_name']} ON {self.conn_config['table_name']}"
            + f" USING {index_param['index_type']} ({self.embedding_column_name}) {with_clause}"
        )
        log.debug(f"Creating index: {create_index_sql}")
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

        log.debug(f"Inserting {len(embeddings)} embeddings into table {self.conn_config['table_name']}")

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
        FilterOp.ExactMatchInt,
        FilterOp.RangeInt,
        FilterOp.ExactMatchInSet,
    ]

    def prepare_filter(self, filters: Filter, attrs: dict | None = None):
        if filters.type == FilterOp.NonFilter:
            self.where_clause = ""
        elif filters.type == FilterOp.NumGE:
            self.where_clause = f"WHERE {self.id_column_name} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.where_clause = f"WHERE {self.predicate_column_name} = '{filters.label_value}'"
        elif filters.type == FilterOp.ExactMatchInt:
            self.where_clause = f"WHERE {self.predicate_column_name} = {attrs['label']}"
        elif filters.type == FilterOp.RangeInt:
            self.where_clause = (
                f"WHERE {self.predicate_column_name} BETWEEN {attrs['range_start']} AND {attrs['range_end']}"
            )
        elif filters.type == FilterOp.ExactMatchInSet:
            self.where_clause = f"WHERE list_contains({self.predicate_column_name}, '{attrs['label']}')"
        else:
            raise ValueError(f"Unsupported filter type: {filters.type}")
        return

    def search_embedding(self, query: list[float], K: int = 100, **kwargs):
        array_function_name = self.case_config._metric_type_to_function_name()

        if self.case_config.use_blob_interface:
            if self.case_config.index == IndexType.HNSW:
                blob_func = "vss_base64_to_blob"
            else:
                blob_func = "pdxearch_base64_to_blob"
            query_vec_literal = f"{blob_func}('{_encode_query_blob_base64(query)}')"
        else:
            query_vec_literal = f"{query}::{self.embedding_column_element_type}[{self.dims}]"

        # The WHERE clause is set by the `prepare_filter` method above.
        result = self.conn.execute(
            f"""SELECT {self.id_column_name} FROM {self.conn_config['table_name']} {self.where_clause}
                ORDER BY {array_function_name}({self.embedding_column_name}, {query_vec_literal})
                LIMIT {K};"""
        )

        return [int(i[0]) for i in result.fetchall()]


_BLOB_INV_SCALE = [100000.0, 10000.0, 1000.0, 100.0]


def _encode_float_to_int16(value: float) -> int:
    """Encode a float to a signed int16 using multi-scale quantization (mirrors pdxearch_blob_codec.hpp)."""
    for x in range(4):
        q = round(value * _BLOB_INV_SCALE[x])
        if -8192 <= q <= 8191:
            return (q << 2) | x
    q = max(-8192, min(8191, round(value * _BLOB_INV_SCALE[3])))
    return (q << 2) | 3


def _encode_query_blob_base64(query_vec: list[float]) -> str:
    """Encode a float vector to a base64 string of quantized int16 values."""
    import struct
    import base64

    encoded = struct.pack(f"<{len(query_vec)}h", *(_encode_float_to_int16(float(v)) for v in query_vec))
    return base64.b64encode(encoded).decode("ascii")
