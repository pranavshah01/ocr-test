
import sys
import shutil
import logging
from pathlib import Path
from typing import List, Set
import argparse
from datetime import datetime


sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_default_config

logger = logging.getLogger(__name__)

def setup_cleanup_logging() -> logging.Logger:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)

def find_pycache_directories(start_path: Path) -> List[Path]:
    pycache_dirs = []

    if not start_path.exists():
        return pycache_dirs

    for item in start_path.rglob("__pycache__"):
        if item.is_dir():
            pycache_dirs.append(item)

    return pycache_dirs

def calculate_directory_size(directory: Path) -> int:
    total_size = 0

    if not directory.exists():
        return 0

    for item in directory.rglob("*"):
        if item.is_file():
            try:
            total_size += item.stat().st_size
        except (OSError, FileNotFoundError):
            # File may have been deleted or become inaccessible
            continue

    return total_size

def format_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f} {size_names[i]}"

def cleanup_directory(directory: Path, dry_run: bool = False) -> tuple[bool, int]:
    if not directory.exists():
        return True, 0

    try:
        size_before = calculate_directory_size(directory)

        if dry_run:
            logger.info(f"Would remove: {directory} ({format_size(size_before)})")
            return True, size_before
        else:
            shutil.rmtree(directory)
            logger.info(f"Removed: {directory} ({format_size(size_before)})")
            return True, size_before

    except Exception as e:
        logger.error(f"Failed to remove %s: %s", directory, str(e).replace('\n', ' ').replace('\r', ''))
        return False, 0

def cleanup_pycache_directories(start_path: Path, dry_run: bool = False) -> tuple[int, int]:
    pycache_dirs = find_pycache_directories(start_path)

    if not pycache_dirs:
        logger.info("No __pycache__ directories found")
        return 0, 0

    logger.info(f"Found {len(pycache_dirs)} __pycache__ directories")

    total_size_freed = 0
    directories_removed = 0

    for pycache_dir in pycache_dirs:
        success, size_freed = cleanup_directory(pycache_dir, dry_run)
        if success:
            directories_removed += 1
            total_size_freed += size_freed

    return directories_removed, total_size_freed

def cleanup_logs(logs_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    if not logs_dir.exists():
        logger.info("No logs directory found")
        return 0, 0

    log_files = list(logs_dir.glob("*.log"))

    if not log_files:
        logger.info("No log files found")
        return 0, 0

    logger.info(f"Found {len(log_files)} log files")

    total_size_freed = 0
    files_removed = 0

    for log_file in log_files:
        try:
            size_before = log_file.stat().st_size

            if dry_run:
                logger.info(f"Would remove: {log_file} ({format_size(size_before)})")
                total_size_freed += size_before
                files_removed += 1
            else:
                log_file.unlink()
                logger.info(f"Removed: {log_file} ({format_size(size_before)})")
                total_size_freed += size_before
                files_removed += 1

        except Exception as e:
            logger.error(f"Failed to remove %s: %s", log_file, str(e).replace('\n', ' ').replace('\r', ''))

    return files_removed, total_size_freed

def cleanup_reports(reports_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    if not reports_dir.exists():
        logger.info("No reports directory found")
        return 0, 0


    report_files = list(reports_dir.glob("*.json")) + list(reports_dir.glob("*.html"))

    if not report_files:
        logger.info("No report files found")
        return 0, 0

    logger.info(f"Found {len(report_files)} report files")

    total_size_freed = 0
    files_removed = 0

    for report_file in report_files:
        try:
            size_before = report_file.stat().st_size

            if dry_run:
                logger.info(f"Would remove: {report_file} ({format_size(size_before)})")
                total_size_freed += size_before
                files_removed += 1
            else:
                report_file.unlink()
                logger.info(f"Removed: {report_file} ({format_size(size_before)})")
                total_size_freed += size_before
                files_removed += 1

        except Exception as e:
            logger.error(f"Failed to remove %s: %s", report_file, str(e).replace('\n', ' ').replace('\r', ''))

    return files_removed, total_size_freed

def create_cleanup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cleanup utility for Document Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m app.cleanup --dry-run --all          # Show what would be cleaned
  python3 -m app.cleanup --logs --reports         # Clean logs and reports only
  python3 -m app.cleanup --cache                  # Clean __pycache__ only
  python3 -m app.cleanup --all                    # Clean everything
        """
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )

    parser.add_argument(
        "--logs",
        action="store_true",
        help="Clean up log files"
    )

    parser.add_argument(
        "--reports",
        action="store_true",
        help="Clean up report files (JSON and HTML)"
    )

    parser.add_argument(
        "--cache",
        action="store_true",
        help="Clean up __pycache__ directories"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Clean up everything (logs, reports, and cache)"
    )

    return parser

def main():
    try:

        logger = setup_cleanup_logging()


        parser = create_cleanup_argument_parser()
        args = parser.parse_args()


        config = get_default_config()


        clean_logs = args.logs or args.all
        clean_reports = args.reports or args.all
        clean_cache = args.cache or args.all

        if not any([clean_logs, clean_reports, clean_cache]):
            logger.error("No cleanup targets specified. Use --logs, --reports, --cache, or --all")
            parser.print_help()
            sys.exit(1)


        mode = "DRY RUN" if args.dry_run else "CLEANUP"
        logger.info(f"Starting {mode} operation")

        total_files_removed = 0
        total_size_freed = 0


        if clean_logs:
            logger.info("Cleaning up log files...")
            files_removed, size_freed = cleanup_logs(config.logs_dir, args.dry_run)
            total_files_removed += files_removed
            total_size_freed += size_freed


        if clean_reports:
            logger.info("Cleaning up report files...")
            files_removed, size_freed = cleanup_reports(config.reports_dir, args.dry_run)
            total_files_removed += files_removed
            total_size_freed += size_freed


        if clean_cache:
            logger.info("Cleaning up __pycache__ directories...")
            dirs_removed, size_freed = cleanup_pycache_directories(Path.cwd(), args.dry_run)
            total_files_removed += dirs_removed
            total_size_freed += size_freed


        logger.info("=" * 60)
        logger.info("CLEANUP SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total items {'that would be ' if args.dry_run else ''}removed: {total_files_removed}")
        logger.info(f"Total space {'that would be ' if args.dry_run else ''}freed: {format_size(total_size_freed)}")

        if args.dry_run:
            logger.info("This was a dry run. Use --all to perform actual cleanup.")
        else:
            logger.info("Cleanup completed successfully!")

    except KeyboardInterrupt:
        logger.info("Cleanup interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()