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
        click.option(
            "--database-name",
            type=str,
            help="Name of a DuckDB database file. It will be created if it doesn't exist.",
            default="test.db",
            show_default=True,
            required=False,
        ),
    ]
    duckdb_threads: Annotated[
        int | None,
        click.option(
            "--duckdb-threads",
            type=int,
            help="Number of threads to use for DuckDB. Affects both index creation and search. Can be overwritten by duckdb_threads_during_index_creation.",
            required=False,
        ),
    ]
    table_name: Annotated[
        str,
        click.option(
            "--table-name",
            type=str,
            help="Name of the table to create.",
            default="test_table",
            show_default=True,
            required=False,
        ),
    ]
    use_blob_interface: Annotated[
        bool,
        click.option(
            "--use-blob-interface/--no-use-blob-interface",
            help="Use the PDXearch blob-based query interface instead of FLOAT[] literals.",
            default=True,
            show_default=True,
        ),
    ]


class DuckDBWithExtensionTypedDict(DuckDBTypedDict):
    index_name: Annotated[
        str,
        click.option(
            "--index-name",
            type=str,
            help="Name of the extension's index to create.",
            default="test_idx",
            show_default=True,
            required=False,
        ),
    ]
    duckdb_threads_during_index_creation: Annotated[
        int | None,
        click.option(
            "--duckdb-threads-during-index-creation",
            type=int,
            help="Number of threads to use during index creation. Overwrites the duckdb_threads parameter temporarily.",
            required=False,
        ),
    ]


class DuckDBPDXearchTypedDict(DuckDBWithExtensionTypedDict):
    extension_path: Annotated[
        str,
        click.option("--extension-path", type=str, help="Path to the DuckDB PDXearchextension.", required=True),
    ]
    quantization_type: Annotated[
        str,
        click.option(
            "--quantization-type",
            type=click.Choice(["f32", "u8"]),
            help="[Index creation parameter] Quantization to use for the embeddings in the PDXearch index. f32 means no quantization.",
            default="u8",
            show_default=True,
            required=False,
        ),
    ]
    # TODO: Improve description.
    n_probe: Annotated[
        int | None,
        click.option(
            "--n-probe",
            type=int,
            help="[Index creation parameter] Number of partitions to probe in (the first iteration of) the PDXearch index search. Stored as the default n_probe value in the index. Can be overwritten by the runtime_n_probe parameter.",
            required=False,
        ),
    ]
    seed: Annotated[
        int | None,
        click.option("--seed", type=int, help="[Index creation parameter] Seed to use for the index.", required=False),
    ]
    runtime_n_probe: Annotated[
        int | None,
        click.option(
            "--runtime-n-probe",
            type=int,
            help="[Runtime parameter] Number of partitions to probe in (the first iteration of) the PDXearch index search. Overwrites the index_n_probe parameter.",
            required=False,
        ),
    ]


class DuckDBVSSTypedDict(DuckDBWithExtensionTypedDict):
    # https://duckdb.org/docs/stable/core_extensions/vss#index-options
    ef_construction: Annotated[
        int,
        click.option(
            "--ef-construction",
            type=int,
            help="[Index creation parameter] The number of candidate vertices to consider during the construction of the index. A higher value will result in a more accurate index, but will also increase the time it takes to build the index.",
            required=False,
            default=128,
            show_default=True,
        ),
    ]
    ef_search: Annotated[
        int,
        click.option(
            "--ef-search",
            type=int,
            help="[Index creation parameter] The number of candidate vertices to consider during the search phase of the index. A higher value will result in a more accurate index, but will also increase the time it takes to perform a search. Overwritten by the runtime_ef_search parameter.",
            required=False,
            default=64,
            show_default=True,
        ),
    ]
    M: Annotated[
        int,
        click.option(
            "--M",
            type=int,
            help="[Index creation parameter] The maximum number of neighbors to keep for each vertex in the graph. A higher value will result in a more accurate index, but will also increase the time it takes to build the index.",
            required=False,
            default=16,
            show_default=True,
        ),
    ]
    M0: Annotated[
        int,
        click.option(
            "--M0",
            type=int,
            help="[Index creation parameter] The base connectivity, or the number of neighbors to keep for each vertex in the zero-th level of the graph. A higher value will result in a more accurate index, but will also increase the time it takes to build the index. Typically 2 * m.",
            required=False,
            default=32,
            show_default=True,
        ),
    ]
    runtime_ef_search: Annotated[
        int | None,
        click.option(
            "--runtime-ef-search",
            type=int,
            help="[Runtime parameter] The number of candidate vertices to consider during the search phase of the index. A higher value will result in a more accurate index, but will also increase the time it takes to perform a search. Overwrites the ef_search parameter.",
            required=False,
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(DuckDBTypedDict)
def DuckDB(**parameters: Unpack[DuckDBTypedDict]):
    from .config import DuckDBConnectionConfig, DuckDBCasePlainConfig

    assert (
        parameters["num_concurrency"] == [1] or parameters["search_concurrent"] == False
    ), "The DuckDB client does not yet support concurrent search."

    run(
        db=DB.DuckDB,
        db_config=DuckDBConnectionConfig(
            database_name=parameters["database_name"],
            duckdb_threads=parameters["duckdb_threads"],
            table_name=parameters["table_name"],
            db_label=parameters["db_label"],
        ),
        db_case_config=DuckDBCasePlainConfig(
            use_blob_interface=parameters["use_blob_interface"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(DuckDBPDXearchTypedDict)
def DuckDBPDXearch(**parameters: Unpack[DuckDBPDXearchTypedDict]):
    from .config import DuckDBConnectionConfig, DuckDBCasePDXearchConfig

    assert (
        parameters["num_concurrency"] == [1] or parameters["search_concurrent"] == False
    ), "The DuckDB PDXearch client does not yet support concurrent search."

    run(
        db=DB.DuckDB,
        db_config=DuckDBConnectionConfig(
            database_name=parameters["database_name"],
            table_name=parameters["table_name"],
            duckdb_threads=parameters["duckdb_threads"],
            db_label=parameters["db_label"],
        ),
        db_case_config=DuckDBCasePDXearchConfig(
            use_blob_interface=parameters["use_blob_interface"],
            extension_path=parameters["extension_path"],
            index_name=parameters["index_name"],
            duckdb_threads_during_index_creation=parameters["duckdb_threads_during_index_creation"],
            index_quantization_type=parameters["quantization_type"],
            index_n_probe=parameters["n_probe"],
            index_seed=parameters["seed"],
            runtime_n_probe=parameters["runtime_n_probe"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(DuckDBVSSTypedDict)
def DuckDBVSS(**parameters: Unpack[DuckDBVSSTypedDict]):
    from .config import DuckDBConnectionConfig, DuckDBCaseVSSConfig

    assert (
        parameters["num_concurrency"] == [1] or parameters["search_concurrent"] == False
    ), "The DuckDB VSS client does not yet support concurrent search."

    run(
        db=DB.DuckDB,
        db_config=DuckDBConnectionConfig(
            database_name=parameters["database_name"],
            table_name=parameters["table_name"],
            duckdb_threads=parameters["duckdb_threads"],
            db_label=parameters["db_label"],
        ),
        db_case_config=DuckDBCaseVSSConfig(
            use_blob_interface=parameters["use_blob_interface"],
            index_name=parameters["index_name"],
            duckdb_threads_during_index_creation=parameters["duckdb_threads_during_index_creation"],
            index_ef_construction=parameters["ef_construction"],
            index_ef_search=parameters["ef_search"],
            index_M=parameters["m"],  # lowercase because of how click works
            index_M0=parameters["m0"],  # lowercase because of how click works
            runtime_ef_search=parameters["runtime_ef_search"],
        ),
        **parameters,
    )
