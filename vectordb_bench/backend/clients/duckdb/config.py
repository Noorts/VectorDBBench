from ..api import DBConfig, DBCaseConfig, MetricType, IndexType
from typing import TypedDict, Sequence, Any
from pydantic import BaseModel
from abc import abstractmethod
from enum import Enum


class CreateIndex(str, Enum):
    BEFORE_INSERT = "BEFORE_INSERT"
    AFTER_INSERT = "AFTER_INSERT"
    NEVER = "NEVER"


class DuckDBConnectionConfigDict(TypedDict):
    database_name: str
    table_name: str
    duckdb_threads: int | None


class DuckDBConnectionConfig(DBConfig):
    """
    Connection information for DuckDB.
    """

    database_name: str
    table_name: str
    duckdb_threads: int | None = None

    def to_dict(self) -> DuckDBConnectionConfigDict:
        return {
            "database_name": self.database_name,
            "table_name": self.table_name,
            "duckdb_threads": self.duckdb_threads,
        }


class DuckDBIndexParam(TypedDict):
    index_type: str  # Flat, PDXearch, ...
    extension_path: str
    index_name: str
    index_options: Sequence[dict[str, Any]]
    duckdb_threads_during_index_creation: int | None


class DuckDBSearchParam(TypedDict):
    array_distance_function_name: str


class DuckDBSessionCommands(TypedDict):
    session_options: Sequence[dict[str, Any]]


class DuckDBCaseConfig(BaseModel, DBCaseConfig):
    index: IndexType = IndexType.NONE
    create_index: CreateIndex = CreateIndex.NEVER
    # The backend sets the metric_type based on the dataset's metric type.
    metric_type: MetricType | None = None

    def _metric_type_to_function_name(self) -> str:
        # https://duckdb.org/docs/stable/sql/functions/array
        if self.metric_type == MetricType.L2:
            return "array_distance"
        if self.metric_type == MetricType.IP:
            return "array_inner_product"
        if self.metric_type == MetricType.COSINE:
            return "array_cosine_distance"
        raise ValueError(f"Unsupported metric type: {self.metric_type}")

    @abstractmethod
    def index_param(self) -> DuckDBIndexParam: ...

    @abstractmethod
    def search_param(self) -> DuckDBSearchParam: ...

    @abstractmethod
    def session_param(self) -> DuckDBSessionCommands: ...


class DuckDBCasePlainConfig(DuckDBCaseConfig):
    index: IndexType = IndexType.Flat
    create_index: CreateIndex = CreateIndex.NEVER

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {
            "array_distance_function_name": self._metric_type_to_function_name(),
        }

    def session_param(self) -> dict:
        return {}


class DuckDBCasePDXearchConfig(DuckDBCaseConfig):
    index: IndexType = IndexType.PDXEARCH
    create_index: CreateIndex = CreateIndex.AFTER_INSERT

    index_name: str
    extension_path: str
    duckdb_threads_during_index_creation: int | None = None

    # Index creation options
    # Except metric. Set automatically based on dataset.
    index_quantization_type: str | None = None
    index_n_probe: int | None = None
    index_normalize: bool | None = None
    index_seed: int | None = None
    runtime_n_probe: int | None = None

    def _metric_type_to_index_metric_type(self) -> str:
        if self.metric_type == MetricType.L2:
            return "l2sq"
        raise ValueError(f"Unsupported metric type: {self.metric_type}")

    def index_param(self) -> DuckDBIndexParam:
        index_options = [
            {"index_metric_type": self._metric_type_to_index_metric_type()}
        ]
        if self.index_quantization_type is not None:
            index_options.append({
                "index_quantization_type": self.index_quantization_type,
            })
        if self.index_n_probe is not None:
            index_options.append({
                "index_n_probe": self.index_n_probe,
            })
        if self.index_normalize is not None:
            index_options.append({
                "index_normalize": self.index_normalize,
            })
        if self.index_seed is not None:
            index_options.append({
                "index_seed": self.index_seed,
            })

        return {
            "index_type": self.index.value,
            "extension_path": self.extension_path,
            "index_name": self.index_name,
            "index_options": index_options,
            "duckdb_threads_during_index_creation": self.duckdb_threads_during_index_creation,
        }

    def search_param(self) -> DuckDBSearchParam:
        return {
            "array_distance_function_name": self._metric_type_to_function_name(),
        }

    def session_param(self) -> DuckDBSessionCommands:
        session_options = []
        if self.runtime_n_probe is not None:
            session_options.append({
                "name": "runtime_n_probe",
                "value": self.runtime_n_probe,
            })
        return {
            "session_options": session_options,
        }


_duckdb_case_config = {
    IndexType.Flat: DuckDBCasePlainConfig,
    IndexType.PDXEARCH: DuckDBCasePDXearchConfig,
}
