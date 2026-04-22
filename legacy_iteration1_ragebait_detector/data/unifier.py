from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import pandas as pd

from ragebait_detector.config import Settings
from ragebait_detector.utils.io import dump_json, write_csv

FINAL_COLUMNS = [
    "post_id",
    "author_id",
    "created_at",
    "language",
    "text",
    "source",
]
SUPPORTED_SUFFIXES = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".txt": "sql",
    ".sql": "sql",
}
CREATE_TABLE_PATTERN = re.compile(
    r"create\s+table\s+(?:if\s+not\s+exists\s+)?(?P<name>[^\s(]+)\s*\((?P<body>.*?)\);",
    flags=re.IGNORECASE | re.DOTALL,
)
SQL_DATA_PATTERN = re.compile(
    r"\b(insert\s+into|copy\s+.+?\s+from\s+stdin|values\s*\()",
    flags=re.IGNORECASE | re.DOTALL,
)
NON_COLUMN_SQL_PREFIXES = (
    "constraint",
    "primary key",
    "foreign key",
    "unique",
    "check",
)


@dataclass
class SqlTableDefinition:
    name: str
    columns: list[str]


@dataclass
class FileInspection:
    path: Path
    kind: str
    total_rows: int = 0
    columns: list[str] = field(default_factory=list)
    preview_rows: list[dict[str, str]] = field(default_factory=list)
    dataframe: pd.DataFrame | None = None
    sql_tables: list[SqlTableDefinition] = field(default_factory=list)
    has_row_data: bool = True
    note: str = ""


@dataclass
class CompiledBatch:
    rows: list[dict[str, str]]
    skipped_empty_text: int
    selected_rows: int


def discover_input_files(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    files = [
        path
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return files


def inspect_input_file(path: str | Path, preview_size: int = 5) -> FileInspection:
    source_path = Path(path)
    kind = SUPPORTED_SUFFIXES.get(source_path.suffix.lower())
    if kind is None:
        raise ValueError(f"Unsupported file type: {source_path.suffix}")

    if kind == "sql":
        text = source_path.read_text(encoding="utf-8", errors="replace")
        tables = extract_sql_table_definitions(text)
        has_row_data = bool(SQL_DATA_PATTERN.search(text))
        note = (
            "Schema file with row-loading statements detected."
            if has_row_data
            else "Schema-only SQL file detected. No tweet rows will be imported from this file."
        )
        return FileInspection(
            path=source_path,
            kind=kind,
            sql_tables=tables,
            has_row_data=has_row_data,
            note=note,
        )

    dataframe = load_tabular_file(source_path)
    preview_rows = [
        {column: stringify_value(value) for column, value in row.items()}
        for row in dataframe.head(preview_size).to_dict(orient="records")
    ]
    return FileInspection(
        path=source_path,
        kind=kind,
        total_rows=len(dataframe),
        columns=[str(column) for column in dataframe.columns],
        preview_rows=preview_rows,
        dataframe=dataframe,
        note=f"Loaded {len(dataframe)} rows from {source_path.name}.",
    )


def load_tabular_file(path: str | Path) -> pd.DataFrame:
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        dataframe = pd.read_csv(
            source_path,
            sep=None,
            engine="python",
            dtype=object,
            encoding_errors="replace",
        )
    elif suffix in {".parquet", ".pq"}:
        dataframe = pd.read_parquet(source_path)
    else:
        raise ValueError(f"Unsupported tabular file type: {suffix}")

    dataframe.columns = [str(column) for column in dataframe.columns]
    return dataframe.reset_index(drop=True)


def extract_sql_table_definitions(sql_text: str) -> list[SqlTableDefinition]:
    tables: list[SqlTableDefinition] = []
    for match in CREATE_TABLE_PATTERN.finditer(sql_text):
        name = match.group("name").strip().strip('"')
        body = match.group("body")
        columns: list[str] = []
        for raw_line in body.splitlines():
            line = raw_line.strip().rstrip(",")
            lowered = line.lower()
            if not line or lowered.startswith(NON_COLUMN_SQL_PREFIXES):
                continue
            column_name = line.split()[0].strip('"')
            columns.append(column_name)
        tables.append(SqlTableDefinition(name=name, columns=columns))
    return tables


def parse_numeric_selection(expression: str, upper_bound: int) -> list[int]:
    normalized = expression.strip().lower()
    if normalized in {"", "all"}:
        return list(range(upper_bound))

    selected: set[int] = set()
    for token in normalized.split(","):
        part = token.strip()
        if not part:
            continue
        if ":" in part or "-" in part:
            separator = ":" if ":" in part else "-"
            start_raw, end_raw = part.split(separator, 1)
            start = 1 if start_raw == "" else int(start_raw)
            end = upper_bound if end_raw == "" else int(end_raw)
            if start < 1 or end < 1 or start > upper_bound or end > upper_bound:
                raise ValueError("Selection is out of range.")
            if end < start:
                raise ValueError("Range end must be greater than or equal to range start.")
            selected.update(range(start - 1, end))
            continue

        index = int(part)
        if index < 1 or index > upper_bound:
            raise ValueError("Selection is out of range.")
        selected.add(index - 1)

    if not selected:
        raise ValueError("No rows or files were selected.")
    return sorted(selected)


def select_rows(dataframe: pd.DataFrame, expression: str) -> pd.DataFrame:
    indices = parse_numeric_selection(expression, upper_bound=len(dataframe))
    return dataframe.iloc[indices].reset_index(drop=True)


def prompt(message: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{message}{suffix}: ").strip()
    if value:
        return value
    return default or ""


def prompt_yes_no(message: str, default: bool = True) -> bool:
    default_token = "Y/n" if default else "y/N"
    value = input(f"{message} [{default_token}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def prompt_file_selection(files: list[Path]) -> list[Path]:
    while True:
        print("\nAvailable input files:")
        for index, path in enumerate(files, start=1):
            print(f"  {index}. {path.name}")
        selection = prompt("Select files by number (e.g. 1,3-5 or all)", default="all")
        try:
            indices = parse_numeric_selection(selection, upper_bound=len(files))
            return [files[index] for index in indices]
        except ValueError as exc:
            print(f"Invalid selection: {exc}")


def print_file_preview(inspection: FileInspection) -> None:
    print(f"\nFile: {inspection.path.name}")
    print(f"Type: {inspection.kind}")
    print(inspection.note)

    if inspection.kind == "sql":
        if inspection.sql_tables:
            print("Tables found:")
            for table in inspection.sql_tables:
                joined_columns = ", ".join(table.columns) if table.columns else "(no columns parsed)"
                print(f"  - {table.name}: {joined_columns}")
        else:
            print("No CREATE TABLE statements were parsed from this SQL file.")
        return

    print(f"Rows available: {inspection.total_rows}")
    print("Columns:")
    for index, column in enumerate(inspection.columns, start=1):
        sample_value = first_non_empty_value(inspection.dataframe, column)
        print(f"  {index}. {column} | sample={sample_value!r}")

    if inspection.preview_rows:
        print("Preview:")
        for index, row in enumerate(inspection.preview_rows, start=1):
            print(f"  Row {index}: {row}")


def first_non_empty_value(dataframe: pd.DataFrame | None, column: str) -> str:
    if dataframe is None:
        return ""
    for value in dataframe[column].head(10).tolist():
        rendered = stringify_value(value)
        if rendered:
            return rendered[:120]
    return ""


def resolve_column_reference(columns: list[str], raw_value: str, required: bool) -> str | None:
    value = raw_value.strip()
    if not value:
        if required:
            raise ValueError("A column selection is required.")
        return None

    if value.isdigit():
        index = int(value)
        if index < 1 or index > len(columns):
            raise ValueError("Column selection is out of range.")
        return columns[index - 1]

    for column in columns:
        if column == value or column.lower() == value.lower():
            return column

    raise ValueError("Column name not found.")


def prompt_column_mapping(columns: list[str]) -> dict[str, str | None]:
    mapping: dict[str, str | None] = {}
    prompts = [
        ("text", True),
        ("author_id", False),
        ("created_at", False),
        ("language", False),
    ]
    for field_name, required in prompts:
        while True:
            raw_value = prompt(
                f"Select the column for {field_name} by number or name",
                default=None,
            )
            try:
                mapping[field_name] = resolve_column_reference(columns, raw_value, required=required)
                break
            except ValueError as exc:
                print(f"Invalid column selection: {exc}")
    return mapping


def compile_rows_from_dataframe(
    dataframe: pd.DataFrame,
    mapping: dict[str, str | None],
    source_name: str,
    starting_post_id: int,
) -> CompiledBatch:
    rows: list[dict[str, str]] = []
    skipped_empty_text = 0

    for _, row in dataframe.iterrows():
        text_value = normalize_text_value(row.get(mapping["text"])) if mapping["text"] else ""
        if not text_value:
            skipped_empty_text += 1
            continue

        rows.append(
            {
                "post_id": str(starting_post_id + len(rows)),
                "author_id": normalize_optional_value(row.get(mapping["author_id"])) if mapping["author_id"] else "",
                "created_at": normalize_optional_value(row.get(mapping["created_at"])) if mapping["created_at"] else "",
                "language": normalize_optional_value(row.get(mapping["language"])).lower() if mapping["language"] else "",
                "text": text_value,
                "source": source_name,
            }
        )

    return CompiledBatch(
        rows=rows,
        skipped_empty_text=skipped_empty_text,
        selected_rows=len(dataframe),
    )


def normalize_text_value(value: Any) -> str:
    text = normalize_optional_value(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_optional_value(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except TypeError:
            pass
    return str(value).strip()


def stringify_value(value: Any) -> str:
    rendered = normalize_optional_value(value)
    return rendered[:120]


def deduplicate_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    seen: set[tuple[str, ...]] = set()
    deduped: list[dict[str, str]] = []

    for row in rows:
        signature = (
            row["author_id"],
            row["created_at"],
            row["language"],
            row["text"],
            row["source"],
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(row)

    renumbered = [
        {
            **row,
            "post_id": str(index),
        }
        for index, row in enumerate(deduped, start=1)
    ]
    return renumbered, len(rows) - len(renumbered)


def resolve_output_path(
    raw_output: str | None,
    default_directory: str | Path,
    default_filename: str,
) -> Path:
    if raw_output:
        candidate = Path(raw_output)
    else:
        candidate = Path(prompt("Output CSV file name", default=default_filename))

    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate
    return Path(default_directory) / candidate


def run_interactive_import(
    settings: Settings,
    input_dir: str | None = None,
    output_path: str | None = None,
    manifest_path: str | None = None,
) -> dict[str, Any]:
    root = Path(input_dir or settings.paths.raw_dir)
    files = discover_input_files(root)
    if not files:
        raise FileNotFoundError(
            f"No supported files were found in {root}. Supported types: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    print(f"Place your untouched Kaggle files in: {root.resolve()}")
    print("CSV and Parquet files can be imported directly.")
    print("SQL/TXT schema files are inspected for reference and skipped if they contain no row data.")

    selected_files = prompt_file_selection(files)
    destination = resolve_output_path(
        raw_output=output_path,
        default_directory=settings.paths.unlabeled_dir,
        default_filename=Path(settings.paths.unlabeled_posts_path).name,
    )
    manifest_destination = Path(manifest_path or settings.paths.source_manifest_path)
    dedupe_rows = prompt_yes_no("Deduplicate identical compiled rows across files?", default=True)

    compiled_rows: list[dict[str, str]] = []
    manifest_entries: list[dict[str, Any]] = []
    next_post_id = 1

    for path in selected_files:
        inspection = inspect_input_file(path)
        print_file_preview(inspection)

        if inspection.kind == "sql" and not inspection.has_row_data:
            manifest_entries.append(
                {
                    "file": path.name,
                    "status": "skipped",
                    "reason": "schema_only_sql",
                    "rows_selected": 0,
                    "rows_imported": 0,
                }
            )
            continue

        if inspection.dataframe is None:
            manifest_entries.append(
                {
                    "file": path.name,
                    "status": "skipped",
                    "reason": "unsupported_sql_payload",
                    "rows_selected": 0,
                    "rows_imported": 0,
                }
            )
            continue

        source_name = prompt("Enter a display name for this data source", default=path.stem)
        row_expression = prompt(
            "Select rows to import by number (e.g. all, 1-500, 1,3,8)",
            default="all",
        )
        subset = select_rows(inspection.dataframe, row_expression)
        if subset.empty:
            manifest_entries.append(
                {
                    "file": path.name,
                    "status": "skipped",
                    "reason": "no_rows_selected",
                    "rows_selected": 0,
                    "rows_imported": 0,
                }
            )
            continue

        mapping = prompt_column_mapping(inspection.columns)
        compiled_batch = compile_rows_from_dataframe(
            dataframe=subset,
            mapping=mapping,
            source_name=source_name,
            starting_post_id=next_post_id,
        )
        compiled_rows.extend(compiled_batch.rows)
        next_post_id += len(compiled_batch.rows)

        manifest_entries.append(
            {
                "file": path.name,
                "status": "imported",
                "source_name": source_name,
                "rows_selected": compiled_batch.selected_rows,
                "rows_imported": len(compiled_batch.rows),
                "rows_skipped_empty_text": compiled_batch.skipped_empty_text,
                "mapping": mapping,
            }
        )

    deduplicated_count = 0
    if dedupe_rows:
        compiled_rows, deduplicated_count = deduplicate_rows(compiled_rows)

    write_csv(destination, compiled_rows, FINAL_COLUMNS)
    summary = {
        "input_directory": str(root),
        "output_csv": str(destination),
        "manifest_path": str(manifest_destination),
        "files_selected": [path.name for path in selected_files],
        "rows_written": len(compiled_rows),
        "duplicate_rows_removed": deduplicated_count,
        "final_columns": FINAL_COLUMNS,
        "sources": manifest_entries,
    }
    dump_json(manifest_destination, summary)
    return summary
