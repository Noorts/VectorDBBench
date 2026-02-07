import logging
import pathlib
import typing
from abc import ABC, abstractmethod
from enum import Enum

from tqdm import tqdm
from fsspec.callbacks import Callback

from vectordb_bench import config

logging.getLogger("s3fs").setLevel(logging.CRITICAL)

log = logging.getLogger(__name__)

DatasetReader = typing.TypeVar("DatasetReader")


class DatasetSource(Enum):
    S3 = "S3"
    AliyunOSS = "AliyunOSS"

    def reader(self, alternative_s3_bucket: bool = False) -> DatasetReader:
        if self == DatasetSource.S3:
            if alternative_s3_bucket:
                assert (
                    config.ALTERNATIVE_AWS_S3_URL
                ), "The ALTERNATIVE_AWS_S3_URL environment variable is not set. The dataset you're using is stored in a private AWS S3 bucket. Please specify the bucket details. See the README."
                assert (
                    config.ALTERNATIVE_AWS_S3_REGION
                ), "The ALTERNATIVE_AWS_S3_REGION environment variable is not set. The dataset you're using is stored in a private AWS S3 bucket. Please specify the bucket details. See the README."

                return AwsS3Reader(
                    remote_root=config.ALTERNATIVE_AWS_S3_URL, region_name=config.ALTERNATIVE_AWS_S3_REGION
                )
            return AwsS3Reader()

        if self == DatasetSource.AliyunOSS:
            return AliyunOSSReader()

        return None


class DatasetReader(ABC):
    source: DatasetSource
    remote_root: str

    @abstractmethod
    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path):
        """read dataset files from remote_root to local_ds_root,

        Args:
            dataset(str): for instance "sift_small_500k"
            files(list[str]):  all filenames of the dataset
            local_ds_root(pathlib.Path): whether to write the remote data.
        """

    @abstractmethod
    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        pass


# TODO: Separate S3 tracking (relies on base class) from AliyunOSS tracking (relies on special callback method).
# TODO: Reduce code duplication?
class ProgressUpdateCallback(Callback):
    """Tracks download progress at byte granularity across multiple files. Integrates with a tqdm progress bar."""

    def __init__(self, tqdm_progress_bar, total_bytes_offset):
        super().__init__()
        self.tqdm_progress_bar = tqdm_progress_bar
        self.total_bytes_offset = total_bytes_offset
        self.file_bytes = 0

    def relative_update(self, inc):
        """For fsspec: called with incremental byte count."""
        self.file_bytes += inc
        self.tqdm_progress_bar.n = self.total_bytes_offset + self.file_bytes
        self.tqdm_progress_bar.refresh()

    def absolute_update(self, bytes_consumed):
        """For oss2: called with absolute byte count for current file."""
        self.file_bytes = bytes_consumed
        self.tqdm_progress_bar.n = self.total_bytes_offset + self.file_bytes
        self.tqdm_progress_bar.refresh()

    def as_oss2_callback(self):
        """Returns a callback function compatible with oss2's progress_callback signature."""

        def oss2_progress(bytes_consumed, total_bytes):
            self.absolute_update(bytes_consumed)

        return oss2_progress


class AliyunOSSReader(DatasetReader):
    source: DatasetSource = DatasetSource.AliyunOSS
    remote_root: str = config.ALIYUN_OSS_URL

    def __init__(self):
        import oss2

        self.bucket = oss2.Bucket(oss2.AnonymousAuth(), self.remote_root, "benchmark", True)

    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        info = self.bucket.get_object_meta(remote.as_posix())

        # check size equal
        remote_size, local_size = info.content_length, local.stat().st_size
        if remote_size != local_size:
            log.info(f"local file: {local} size[{local_size}] not match with remote size[{remote_size}]")
            return False

        return True

    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path):
        downloads = []
        if not local_ds_root.exists():
            log.info(f"local dataset root path not exist, creating it: {local_ds_root}")
            local_ds_root.mkdir(parents=True)
            downloads = [
                (
                    pathlib.PurePosixPath("benchmark", dataset, f),
                    local_ds_root.joinpath(f),
                )
                for f in files
            ]

        else:
            for file in files:
                remote_file = pathlib.PurePosixPath("benchmark", dataset, file)
                local_file = local_ds_root.joinpath(file)

                if (not local_file.exists()) or (not self.validate_file(remote_file, local_file)):
                    log.info(f"local file: {local_file} not match with remote: {remote_file}; add to downloading list")
                    downloads.append((remote_file, local_file))

        if len(downloads) == 0:
            return

        total_bytes_to_download = sum(
            self.bucket.get_object_meta(remote.as_posix()).content_length for remote, _ in downloads
        )
        log.info(
            f"Started downloading files from AliyunOSS, total count: {len(downloads)}, total size: {total_bytes_to_download / (1024**3):.2f} GB"
        )

        total_files = len(downloads)
        tqdm_progress_bar = tqdm(
            total=total_bytes_to_download,
            desc=f"Downloaded (0/{total_files} files)",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )
        bytes_downloaded = 0
        files_completed = 0

        for remote_file, local_file in downloads:
            log.debug(f"Downloading file '{remote_file}' to '{local_file}'")
            file_size = self.bucket.get_object_meta(remote_file.as_posix()).content_length

            callback = ProgressUpdateCallback(tqdm_progress_bar, bytes_downloaded)
            self.bucket.get_object_to_file(
                remote_file.as_posix(),
                local_file.absolute(),
                progress_callback=callback.as_oss2_callback(),
            )
            bytes_downloaded += file_size
            files_completed += 1
            tqdm_progress_bar.set_description(f"Downloaded ({files_completed}/{total_files} files)")

        tqdm_progress_bar.close()
        log.info(f"Completed downloading all files, downloaded file count = {len(downloads)}")


class AwsS3Reader(DatasetReader):
    source: DatasetSource = DatasetSource.S3
    remote_root: str
    region_name: str

    def __init__(self, remote_root: str = config.AWS_S3_URL, region_name: str = "us-west-2"):
        import s3fs

        self.remote_root = remote_root
        self.region_name = region_name

        self.fs = s3fs.S3FileSystem(
            anon=True,
            client_kwargs={"region_name": region_name},
        )

    def ls_all(self, dataset: str):
        dataset_root_dir = pathlib.Path(self.remote_root, dataset)
        log.info(f"listing dataset: {dataset_root_dir}")
        names = self.fs.ls(dataset_root_dir)
        for n in names:
            log.info(n)
        return names

    def read(self, dataset: str, files: list[str], local_ds_root: pathlib.Path):
        downloads = []
        if not local_ds_root.exists():
            log.info(f"local dataset root path not exist, creating it: {local_ds_root}")
            local_ds_root.mkdir(parents=True)
            downloads = [pathlib.PurePosixPath(self.remote_root, dataset, f) for f in files]

        else:
            for file in files:
                remote_file = pathlib.PurePosixPath(self.remote_root, dataset, file)
                local_file = local_ds_root.joinpath(file)

                if (not local_file.exists()) or (not self.validate_file(remote_file, local_file)):
                    log.info(f"local file: {local_file} not match with remote: {remote_file}; add to downloading list")
                    downloads.append(remote_file)

        if len(downloads) == 0:
            return

        total_bytes_to_download = sum(self.fs.info(f).get("size", 0) for f in downloads)
        log.info(
            f"Started downloading files from S3, total count: {len(downloads)}, total size: {total_bytes_to_download / (1024**3):.2f} GB"
        )

        total_files = len(downloads)
        tqdm_progress_bar = tqdm(
            total=total_bytes_to_download,
            desc=f"Downloaded (0/{total_files} files)",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )
        bytes_downloaded = 0
        files_completed = 0

        for s3_file in downloads:
            log.debug(f"Downloading file '{s3_file}' to '{local_ds_root}'")
            file_size = self.fs.info(s3_file).get("size", 0)

            callback = ProgressUpdateCallback(tqdm_progress_bar, bytes_downloaded)
            local_file = local_ds_root / s3_file.name
            self.fs.get_file(s3_file.as_posix(), local_file.as_posix(), callback=callback)
            bytes_downloaded += file_size
            files_completed += 1
            tqdm_progress_bar.set_description(f"Downloaded ({files_completed}/{total_files} files)")

        tqdm_progress_bar.close()
        log.info(f"Completed downloading all files, downloaded file count = {len(downloads)}")

    def validate_file(self, remote: pathlib.Path, local: pathlib.Path) -> bool:
        # info() uses ls() inside, maybe we only need to ls once
        info = self.fs.info(remote)

        # check size equal
        remote_size, local_size = info.get("size"), local.stat().st_size
        if remote_size != local_size:
            log.info(f"local file: {local} size[{local_size}] not match with remote size[{remote_size}]")
            return False

        return True
