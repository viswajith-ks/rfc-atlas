"""RFC Atlas: Kaggle Vector Forge Orchestrator.

Handles multi-tenant identity hydration, ephemeral staging, and robust execution
tracking for remote GPU-accelerated embedding generation.
"""

import argparse
import builtins
import fnmatch
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.cli import ApiException
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelSessionStatusResponse,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CHUNKS_DIR = _PROJECT_ROOT / "data" / "chunks"
LOCAL_EMBEDDINGS_DIR = _PROJECT_ROOT / "data" / "embeddings"
STAGING_DIR = _PROJECT_ROOT / ".scratch" / "kaggle_deploy"
LOCAL_LOG_DIR = _PROJECT_ROOT / "data" / "logs"
REMOTE_ENTRY_MODULE = "rfc_atlas.vector_store.kaggle_embedder"
REMOTE_ENTRY_SCRIPT = "src/rfc_atlas/vector_store/kaggle_embedder.py"

IGNORE_PATTERNS = {
    ".git",
    ".venv",
    ".scratch",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".ruff_cache",
    "tests",
    "notebooks",
    "data/raw",
    "data/logs",
    "data/lancedb",
    "data/graph",
    "data/embeddings",
    "data/normalized",
    "rfc_atlas.egg-info",
}


def _generate_dataset_meta(ds_slug: str, dataset_id: str) -> dict[str, Any]:
    """Generates Kaggle dataset staging metadata.

    Args:
        ds_slug (str): The combined Kaggle user/dataset routing slug.
        dataset_id (str): The unique ID of the target dataset.

    Returns:
        dict[str, Any]: The Kaggle-compliant dataset metadata payload.
    """
    return {
        "title": f"RFC Atlas Temporary Chunks {dataset_id}",
        "id": ds_slug,
        "licenses": [{"name": "CC0-1.0"}],
    }


def _generate_kernel_meta(kernel_slug: str, ds_slug: str) -> dict[str, Any]:
    """Generates Kaggle kernel staging metadata for Dual-T4 execution.

    Args:
        kernel_slug (str): The combined Kaggle user/kernel routing slug.
        ds_slug (str): The bound target dataset slug providing the payload.

    Returns:
        dict[str, Any]: The Kaggle-compliant kernel metadata payload.
    """
    return {
        "id": kernel_slug,
        "title": "RFC Atlas Forge Engine",
        "code_file": "kaggle_entrypoint.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [ds_slug],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def _generate_notebook_payload(ds_slug: str) -> dict[str, Any]:
    """Generates the Kaggle Jupyter Notebook payload for execution.

    Args:
        ds_slug (str): The bound target dataset slug providing the payload.

    Returns:
        dict[str, Any]: The Kaggle-compliant ipynb structured dictionary.
    """
    source_cells = [
        "import os, subprocess, sys, zipfile, shutil\n",
        "from pathlib import Path\n\n",
        "try:\n",
        "    print('==== KAGGLE BOOT SEQUENCE ====')\n",
        "    import torch\n",
        "    gpu_count = torch.cuda.device_count()\n",
        "    if gpu_count < 2:\n",
        "        print(f'🛑 FATAL: Expected Dual T4s, but Kaggle '\n",
        "              f'allocated {gpu_count} GPU(s).')\n",
        "        sys.exit(1)\n\n",
        "    print(f'[*] Hardware Discovery: Found {gpu_count} '\n",
        "          'GPU(s). Proceeding with execution.')\n",
        "    bundle_path = next(Path('/kaggle/input').rglob('source_bundle.pack'))\n",
        "    workdir = Path('/tmp/project')\n",
        "    workdir.mkdir(parents=True, exist_ok=True)\n\n",
        "    print(f'[*] Extracting secure payload to {workdir}...')\n",
        "    with zipfile.ZipFile(bundle_path, 'r') as z:\n",
        "        z.extractall(workdir)\n\n",
        "    resume_dir = workdir / 'resume_state' / 'parquet_vectors'\n",
        "    target_out = Path('/kaggle/working/parquet_vectors')\n",
        "    if resume_dir.exists():\n",
        "        parquets = list(resume_dir.rglob('*.parquet'))\n",
        "        if parquets:\n",
        "            print(f'[*] Cloud Resume: Restoring '\n",
        "                  f'{len(parquets)} previous shards to '\n",
        "                  'Kaggle output directory...')\n",
        "            target_out.mkdir(parents=True, exist_ok=True)\n",
        "            for pq in parquets:\n",
        "                shutil.copy2(pq, target_out / pq.name)\n\n",
        "    print('[*] Installing project dependencies...')\n",
        "    subprocess.run(\n",
        "        [sys.executable, '-m', 'pip', 'install', '-e', '.[vector-store]'],\n",
        "        cwd=workdir, check=True\n",
        "    )\n\n",
        "    gpus_available = [f'cuda:{i}' for i in range(gpu_count)]\n",
        "    print('[*] Patching embedder dynamically for hardware...')\n",
        f"    embedder_script = workdir / '{REMOTE_ENTRY_SCRIPT}'\n",
        "    code = embedder_script.read_text()\n",
        "    code = code.replace(\n",
        '        \'target_devices=["cuda:0", "cuda:1"]\',\n',
        "        f'target_devices={gpus_available}'\n",
        "    )\n",
        "    embedder_script.write_text(code)\n\n",
        "    print('[*] Igniting Kaggle Embedder Pipeline...')\n",
        "    subprocess.run(\n",
        f"        [sys.executable, '-m', '{REMOTE_ENTRY_MODULE}',\n",
        "         '--in-dir', 'data/chunks'], cwd=workdir, check=True\n",
        "    )\n",
        "    print('[+] Kaggle Vector Forge completed successfully.')\n",
        "except SystemExit:\n",
        "    pass\n",
        "except Exception as e:\n",
        "    import traceback\n",
        "    traceback.print_exc()\n",
        "    Path('/kaggle/working/fatal_error.log').write_text(\n",
        "        traceback.format_exc()\n",
        "    )\n",
        "    sys.exit(1)\n",
    ]

    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source_cells,
            }
        ],
        "metadata": {
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "dataSources": [{"sourceId": ds_slug, "sourceType": "dataset"}],
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


class KaggleOrchestrator:
    """Manages ephemeral Kaggle kernels for high-throughput GPU vectorization."""

    def __init__(self) -> None:
        """Initializes the orchestrator, hydrates credentials, and generates slugs."""
        self.api = KaggleApi()
        self._authenticate_client()
        self.username = self._resolve_username()
        self.dataset_id = f"lance-chunks-{int(time.time())}"
        self.ds_slug = f"{self.username}/{self.dataset_id}"
        self.kernel_id = "rfc-atlas-forge-engine"
        self.kernel_slug = f"{self.username}/{self.kernel_id}"

    def _authenticate_client(self) -> None:
        """Authenticates the Kaggle API client via local tokens."""
        try:
            self.api.authenticate()
        except (OSError, ValueError):
            logger.exception("Kaggle Authentication Failed! Details")
            sys.exit(1)

    def _resolve_username(self) -> str:
        """Extracts the verified Kaggle username from the loaded configuration.

        Returns:
            str: The authenticated Kaggle username string.
        """
        username = self.api.config_values.get("username") or os.environ.get(
            "KAGGLE_USERNAME"
        )
        if not username:
            logger.error("Could not resolve Kaggle username.")
            sys.exit(1)
        return str(username)

    @staticmethod
    def _should_ignore(path: Path) -> bool:
        """Evaluates if a path should be excluded from the source bundle upload.

        Args:
            path (Path): The local filesystem path to evaluate.

        Returns:
            bool: True if the path matches an exclusion pattern, False otherwise.
        """
        rel = path.relative_to(_PROJECT_ROOT).as_posix()
        for pat in IGNORE_PATTERNS:
            if fnmatch.fnmatch(rel, pat) or rel.startswith(pat.rstrip("/") + "/"):
                return True
        return False

    def build_pack_bundle(self, bundle_path: Path) -> None:
        """Compiles project source code and JSONL chunks into a secure ZIP bundle.

        Args:
            bundle_path (Path): Target path for the output ZIP file.
        """
        logger.info(
            "[*] Compiling source code and chunks into secure bundle: %s",
            bundle_path.name,
        )
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for path in _PROJECT_ROOT.rglob("*"):
                if path.is_dir() or self._should_ignore(path):
                    continue
                z.write(path, arcname=path.relative_to(_PROJECT_ROOT))

            if LOCAL_EMBEDDINGS_DIR.exists():
                parquets = list(LOCAL_EMBEDDINGS_DIR.rglob("*.parquet"))
                if parquets:
                    logger.info(
                        "[*] Cloud Resume: Bundling %d existing Parquet shards...",
                        len(parquets),
                    )
                    for pq in parquets:
                        z.write(pq, arcname=f"resume_state/parquet_vectors/{pq.name}")

    def prepare_dataset_staging(self) -> Path:
        """Prepares the ephemeral Kaggle Dataset workspace containing the payload.

        Returns:
            Path: The directory containing the dataset staging files.
        """
        if STAGING_DIR.exists():
            shutil.rmtree(STAGING_DIR)

        dataset_stage = STAGING_DIR / "dataset"
        dataset_stage.mkdir(parents=True, exist_ok=True)

        ds_meta = _generate_dataset_meta(self.ds_slug, self.dataset_id)
        with (dataset_stage / "dataset-metadata.json").open("w", encoding="utf-8") as f:
            json.dump(ds_meta, f, indent=2)

        self.build_pack_bundle(dataset_stage / "source_bundle.pack")
        return dataset_stage

    def prepare_kernel_staging(self) -> None:
        """Generates the ephemeral Kaggle Notebook execution environment metadata."""
        kernel_stage = STAGING_DIR / "kernel"
        kernel_stage.mkdir(parents=True, exist_ok=True)

        kernel_meta = _generate_kernel_meta(self.kernel_slug, self.ds_slug)
        with (kernel_stage / "kernel-metadata.json").open("w", encoding="utf-8") as f:
            json.dump(kernel_meta, f, indent=2)

        notebook_payload = _generate_notebook_payload(self.ds_slug)
        with (kernel_stage / "kaggle_entrypoint.ipynb").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(notebook_payload, f, indent=2)

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=15),
        retry=retry_if_exception_type((
            ApiException,
            ConnectionError,
            TimeoutError,
            OSError,
            ValueError,
        )),
    )
    def push_dataset_with_retry(self, dataset_stage: Path) -> None:
        """Uploads the ephemeral chunk dataset to Kaggle with exponential backoff.

        Args:
            dataset_stage (Path): Path to the compiled dataset staging directory.
        """
        logger.info("[*] Initializing new ephemeral dataset: %s...", self.ds_slug)
        self.api.dataset_create_new(str(dataset_stage), quiet=True)
        logger.info("[+] Successfully pushed ephemeral dataset.")

    def push_pipeline(self) -> None:
        """Coordinates the full cloud-bridge deployment and starts the remote engine."""
        logger.info(
            "[*] Starting Cloud-Bridge Push Sequence for tenant: %s", self.username
        )
        dataset_stage = self.prepare_dataset_staging()
        self.prepare_kernel_staging()

        self.push_dataset_with_retry(dataset_stage)

        logger.info("[*] Waiting for Kaggle backend to mount ephemeral dataset...")
        while True:
            try:
                status_raw = self.api.dataset_status(  # pyright: ignore[reportUnknownMemberType]
                    self.ds_slug
                )
            except (
                ApiException,
                OSError,
                ValueError,
                ConnectionError,
                TimeoutError,
            ) as e:
                logger.warning("[*] Transient error tracking dataset status: %s", e)
                time.sleep(15)
                continue

            status = str(status_raw).lower()
            if status == "ready":
                logger.info("[+] Dataset target status flipped to READY.")
                break
            if status == "error":
                logger.error("Kaggle storage processing failed.")
                sys.exit(1)

            time.sleep(15)

        logger.info(
            "[*] Dataset ready. Allowing 30 seconds for IAM backend propagation..."
        )
        time.sleep(30)

        logger.info(
            "[*] Igniting compute execution pool for kernel: %s", self.kernel_slug
        )
        self.api.kernels_push(str(STAGING_DIR / "kernel"))
        logger.info("[+++] Headless Dual-T4 Forge Engine successfully activated.")

    def _fetch_and_print_crash_logs(self) -> None:
        """Attempts to retrieve the traceback of a fatally crashed remote execution."""
        LOCAL_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            self.api.kernels_output(self.kernel_slug, path=str(LOCAL_EMBEDDINGS_DIR))
        except (ApiException, OSError, ValueError, ConnectionError, TimeoutError) as e:
            logger.warning("[-] Could not fetch crash logs: %s", e)
            return

        error_log = LOCAL_EMBEDDINGS_DIR / "fatal_error.log"
        if error_log.exists():
            logger.error("\n\n==================================================")
            logger.error("💥 FATAL EXCEPTION DETECTED ON KAGGLE SERVERS 💥")
            logger.error("==================================================\n")
            logger.error(error_log.read_text())
            logger.error("==================================================\n\n")
            error_log.unlink()
        else:
            logger.info(
                "[+] Downloaded the crashed notebook. Check %s", LOCAL_EMBEDDINGS_DIR
            )

    @retry(
        reraise=True,
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception_type((
            ApiException,
            ConnectionError,
            TimeoutError,
            OSError,
            ValueError,
        )),
    )
    def _check_status(self) -> ApiGetKernelSessionStatusResponse:
        """Pings the Kaggle API for the latest kernel execution status.

        Returns:
            ApiGetKernelSessionStatusResponse: The kernel status payload.
        """
        return self.api.kernels_status(  # pyright: ignore[reportUnknownMemberType]
            self.kernel_slug
        )

    def _archive_cloud_logs(self) -> None:
        """Moves telemetry and logs from the payload into the local log directory."""
        LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        downloaded_log = LOCAL_EMBEDDINGS_DIR / "embedder_telemetry.log"

        if downloaded_log.exists():
            target_log = LOCAL_LOG_DIR / f"cloud_forge_{int(time.time())}.log"
            shutil.move(str(downloaded_log), str(target_log))
            logger.info("[*] Cloud python telemetry recovered and routed to logs.")
        else:
            fallback_log = (
                LOCAL_EMBEDDINGS_DIR / "parquet_vectors" / "embedder_telemetry.log"
            )
            if fallback_log.exists():
                target_log = LOCAL_LOG_DIR / f"cloud_forge_{int(time.time())}.log"
                shutil.move(str(fallback_log), str(target_log))
                logger.info(
                    "[*] Cloud telemetry recovered from fallback and routed to logs."
                )

        kernel_log = LOCAL_EMBEDDINGS_DIR / f"{self.kernel_id}.log"
        if kernel_log.exists():
            target_kernel_log = (
                LOCAL_LOG_DIR / f"kaggle_kernel_stdout_{int(time.time())}.log"
            )
            shutil.move(str(kernel_log), str(target_kernel_log))
            logger.info("[*] Kaggle native stdout log recovered and routed to logs.")

    def poll_and_fetch_pipeline(self) -> None:
        """Monitors kernel execution state and pulls completed Parquet vectors."""
        logger.info(
            "[*] Initiating Stakeout Sequence for remote engine: %s", self.kernel_slug
        )

        while True:
            try:
                response = self._check_status()
                status = str(response.status).lower()
            except (
                ApiException,
                ConnectionError,
                TimeoutError,
                OSError,
                ValueError,
            ) as e:
                logger.warning(
                    "[!] Warning: Status tracking transient error: "
                    "%s. Retrying loop...",
                    type(e).__name__,
                )
                time.sleep(30)
                continue

            logger.info("[*] Current Cloud Status: %s", status.upper())

            if "complete" in status:
                LOCAL_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

                logger.info(
                    "[*] 📥 Initiating massive payload download. This may take "
                    "several minutes depending on your network speed..."
                )
                self.api.kernels_output(
                    self.kernel_slug, path=str(LOCAL_EMBEDDINGS_DIR)
                )

                parquets = list(LOCAL_EMBEDDINGS_DIR.rglob("*.parquet"))
                if not parquets:
                    logger.error(
                        "Execution marked complete, but no parquet "
                        "files were generated."
                    )
                    logger.info("[*] Checking for crash logs...")
                    self._fetch_and_print_crash_logs()
                    self._cleanup_ephemeral_assets()
                    sys.exit(1)

                logger.info(
                    "[+] Valid Parquet artifacts detected! Dual-T4 run "
                    "successfully finished."
                )
                break

            if "error" in status or "cancel" in status:
                logger.error(
                    "Remote GPU execution died with terminating state: %s",
                    status.upper(),
                )
                logger.info("[*] Attempting to pull crash logs from Kaggle...")
                self._fetch_and_print_crash_logs()
                self._cleanup_ephemeral_assets()
                sys.exit(1)

            time.sleep(60)

        logger.info(
            "[+++] Transfer sequence complete. Local embeddings space is synchronized."
        )
        self._archive_cloud_logs()

    def _cleanup_ephemeral_assets(self) -> None:
        """Automates the destruction of ephemeral cloud datasets and kernels."""
        logger.info("[*] Initiating ephemeral cloud asset cleanup...")

        original_input = builtins.input
        builtins.input = lambda _="": "y"

        try:
            try:
                if hasattr(self.api, "dataset_delete"):
                    self.api.dataset_delete(self.username, self.dataset_id)
                    logger.info(
                        "[+] Ephemeral dataset %s wiped from Kaggle.", self.ds_slug
                    )
            except (
                ApiException,
                OSError,
                ValueError,
                ConnectionError,
                TimeoutError,
            ) as e:
                logger.warning(
                    "[-] Non-fatal warning: Could not cleanly delete dataset "
                    "%s. Reason: %s",
                    self.ds_slug,
                    e,
                )

            try:
                if hasattr(self.api, "kernels_delete"):
                    self.api.kernels_delete(self.kernel_slug, no_confirm=True)
                    logger.info(
                        "[+] Ephemeral kernel %s wiped from Kaggle.", self.kernel_slug
                    )
            except (
                ApiException,
                OSError,
                ValueError,
                ConnectionError,
                TimeoutError,
            ) as e:
                logger.warning(
                    "[-] Non-fatal warning: Could not cleanly delete kernel "
                    "%s. Reason: %s",
                    self.kernel_slug,
                    e,
                )
        finally:
            builtins.input = original_input

        if STAGING_DIR.exists():
            shutil.rmtree(STAGING_DIR)


def main() -> None:
    """Parses arguments and orchestrates the Kaggle ephemeral compute pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="RFC Atlas Cloud Sync Orchestrator")
    parser.add_argument(
        "--phase",
        choices=["push", "poll", "full"],
        required=True,
        help=(
            "Execution strategy. 'push' ignites the run. "
            "'poll' fetches output. 'full' does both."
        ),
    )
    args = parser.parse_args()

    try:
        orchestrator = KaggleOrchestrator()
        if args.phase in {"push", "full"}:
            orchestrator.push_pipeline()
        if args.phase in {"poll", "full"}:
            orchestrator.poll_and_fetch_pipeline()
    except Exception:
        logger.exception("CRITICAL FAILURE: Kaggle Cloud Sync aborted abnormally.")
        sys.exit(1)


if __name__ == "__main__":
    main()
