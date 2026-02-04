from typing import Annotated, Unpack

import click

from vectordb_bench.cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from vectordb_bench.backend.clients import DB


class DuckDBTypedDict(CommonTypedDict):
    database_name: Annotated[
        str,
        click.option("--database-name", type=str,
                     help="Name of a DuckDB database file. It will be created if it doesn't exist.",
                     default="test.db", show_default=True, required=False),
    ]
    duckdb_threads: Annotated[
        int | None,
        click.option("--duckdb-threads", type=int,
                     help="Number of threads to use for DuckDB. Affects both index creation and search. Can be overwritten by duckdb_threads_during_index_creation.",
                     required=False),
    ]
    table_name: Annotated[
        str,
        click.option("--table-name", type=str,
                     help="Name of the table to create.",
                     default="test_table", show_default=True, required=False),
    ]


class DuckDBWithExtensionTypedDict(DuckDBTypedDict):
    extension_path: Annotated[
        str,
        click.option("--extension-path", type=str,
                     help="Path to the DuckDB extension.",
                     required=True),
    ]
    index_name: Annotated[
        str,
        click.option("--index-name", type=str,
                     help="Name of the extension's index to create.",
                     default="test_idx", show_default=True, required=False),
    ]
    duckdb_threads_during_index_creation: Annotated[
        int | None,
        click.option("--duckdb-threads-during-index-creation", type=int,
                     help="Number of threads to use during index creation. Overwrites the duckdb_threads parameter temporarily.",
                     required=False),
    ]


class DuckDBPDXearchTypedDict(DuckDBWithExtensionTypedDict):
    quantization_type: Annotated[
        str,
        click.option("--quantization-type", type=click.Choice(["F32", "SQ8"]),
                     help="[Index creation parameter] Quantization to use for the embeddings in the PDXearch index. F32 means no quantization.",
                     default="F32", show_default=True, required=False),
    ]
    # TODO: Improve description.
    n_probe: Annotated[
        int | None,
        click.option("--n-probe", type=int,
                     help="[Index creation parameter] Number of partitions to probe in (the first iteration of) the PDXearch index search. Stored as the default n_probe value in the index. Can be overwritten by the runtime_n_probe parameter.",
                     required=False),
    ]
    # TODO: Improve description.
    normalize: Annotated[
        bool | None,
        click.option("--normalize", type=bool,
                     help="[Index creation parameter] Whether to normalize the vectors before storing them in the index.",
                     required=False),
    ]
    seed: Annotated[
        int | None,
        click.option("--seed", type=int,
                     help="[Index creation parameter] Seed to use for the index.",
                     required=False),
    ]
    runtime_n_probe: Annotated[
        int | None,
        click.option("--runtime-n-probe", type=int,
                     help="[Runtime parameter] Number of partitions to probe in (the first iteration of) the PDXearch index search. Overwrites the index_n_probe parameter.",
                     required=False),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(DuckDBTypedDict)
def DuckDB(**parameters: Unpack[DuckDBTypedDict]):
    from .config import DuckDBConnectionConfig, DuckDBCasePlainConfig

    print(parameters["num_concurrency"])
    print(parameters["search_concurrent"])

    assert parameters["num_concurrency"] == [1] or parameters[
        "search_concurrent"] == False, "The DuckDB client does not yet support concurrent search."

    run(
        db=DB.DuckDB,
        db_config=DuckDBConnectionConfig(
            database_name=parameters["database_name"], duckdb_threads=parameters["duckdb_threads"],
            table_name=parameters["table_name"]),
        db_case_config=DuckDBCasePlainConfig(**parameters),
        ** parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(DuckDBPDXearchTypedDict)
def DuckDBPDXearch(**parameters: Unpack[DuckDBPDXearchTypedDict]):
    from .config import DuckDBConnectionConfig, DuckDBCasePDXearchConfig

    assert parameters["num_concurrency"] == [1] or parameters[
        "search_concurrent"] == False, "The DuckDB PDXearch client does not yet support concurrent search."

    run(
        db=DB.DuckDB,
        db_config=DuckDBConnectionConfig(
            database_name=parameters["database_name"],
            table_name=parameters["table_name"],
            duckdb_threads=parameters["duckdb_threads"],
        ),
        db_case_config=DuckDBCasePDXearchConfig(
            extension_path=parameters["extension_path"],
            index_name=parameters["index_name"],
            duckdb_threads_during_index_creation=parameters["duckdb_threads_during_index_creation"],
            index_quantization_type=parameters["quantization_type"],
            index_n_probe=parameters["n_probe"],
            index_normalize=parameters["normalize"],
            index_seed=parameters["seed"],
            runtime_n_probe=parameters["runtime_n_probe"],
        ),
        **parameters,
    )
