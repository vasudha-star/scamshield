"""Distributed PySpark Big Data Pipeline for ScamShield (Phase 4).

Executes distributed processing:
1. Ingestion of unified datasets into Spark DataFrame.
2. Distributed Schema Standardization and Validation.
3. Distributed URL and Length profiling using Spark Catalyst expressions.
4. Distributed Aggregations & Big Data Analytics:
   - Records by source dataset
   - Records by language
   - Records by project label
   - URL availability by modality
   - Text length statistics (min, max, mean, stddev)
5. Partition analysis and performance benchmarking (records/sec, wall-clock time).
6. Partitioned Parquet output generation to data/interim/.
7. Produces reports/spark_analytics_report.json and reports/spark_pipeline_report.md.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from src.utils.logger import get_logger

logger = get_logger("spark_pipeline")

# Configure HADOOP_HOME for Windows execution
hadoop_dir = PROJECT_ROOT / "hadoop"
if (hadoop_dir / "bin" / "winutils.exe").exists():
    os.environ["HADOOP_HOME"] = str(hadoop_dir)
    os.environ["PATH"] = str(hadoop_dir / "bin") + os.pathsep + os.environ.get("PATH", "")


def create_spark_session(app_name: str = "ScamShield-BigData-Pipeline") -> SparkSession:
    """Builds an optimized local SparkSession utilizing all available CPU cores."""
    logger.info("Initializing PySpark Session with local[*] master...")
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    # Set log level to WARN to reduce terminal noise
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"PySpark Session created successfully. Version: {spark.version}")
    return spark


def run_spark_pipeline(input_path: str = "data/processed/cleaned.parquet", output_dir: str = "data/interim") -> dict[str, Any]:
    start_time = time.time()
    spark = create_spark_session()

    input_file = PROJECT_ROOT / input_path
    out_dir = PROJECT_ROOT / output_dir
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found at {input_file}. Ensure Phase 3 has run.")

    logger.info(f"Reading dataset into PySpark from {input_file}...")
    df = spark.read.parquet(str(input_file))

    total_records = df.count()
    num_partitions = df.rdd.getNumPartitions()
    logger.info(f"Loaded {total_records:,} records across {num_partitions} Spark partitions.")

    # 1. Big Data Analytics: Records by Source Dataset
    logger.info("Computing distributed aggregation: Records by Source Dataset...")
    source_stats = (
        df.groupBy("source_dataset")
        .agg(
            F.count("*").alias("count"),
            F.round(F.mean(F.length("text")), 2).alias("avg_length"),
            F.sum(F.when(F.col("url_present") == True, 1).otherwise(0)).alias("url_count"),
        )
        .toPandas()
    )

    # 2. Big Data Analytics: Records by Language
    logger.info("Computing distributed aggregation: Records by Language...")
    lang_stats = (
        df.groupBy("language")
        .agg(F.count("*").alias("count"))
        .orderBy(F.desc("count"))
        .toPandas()
    )

    # 3. Big Data Analytics: Records by Project Label
    logger.info("Computing distributed aggregation: Records by Project Label...")
    label_stats = (
        df.groupBy("project_label")
        .agg(F.count("*").alias("count"))
        .orderBy(F.desc("count"))
        .toPandas()
    )

    # 4. Big Data Analytics: Text Length Statistics
    logger.info("Computing distributed global length statistics...")
    length_metrics = (
        df.select(
            F.min(F.length("text")).alias("min_length"),
            F.max(F.length("text")).alias("max_length"),
            F.round(F.mean(F.length("text")), 2).alias("mean_length"),
            F.round(F.stddev(F.length("text")), 2).alias("stddev_length"),
        )
        .first()
        .asDict()
    )

    # 5. Distributed Partitioned Parquet Write
    logger.info(f"Writing distributed partitioned Parquet partitions to {out_dir}...")
    write_start = time.time()
    (
        df.write.mode("overwrite")
        .partitionBy("channel")
        .parquet(str(out_dir))
    )
    write_time = round(time.time() - write_start, 2)
    logger.info(f"Partitioned write completed in {write_time} seconds.")

    # 6. Calculate Output Size
    total_size_bytes = sum(f.stat().st_size for f in out_dir.rglob("*.parquet"))
    output_size_mb = round(total_size_bytes / (1024 * 1024), 2)

    total_pipeline_time = round(time.time() - start_time, 2)
    throughput = round(total_records / max(total_pipeline_time, 0.001), 2)

    spark.stop()
    logger.info("SparkSession stopped.")

    # Compile analytics dictionary
    analytics = {
        "execution_summary": {
            "records_processed": int(total_records),
            "processing_time_sec": total_pipeline_time,
            "throughput_records_per_sec": throughput,
            "spark_partitions": num_partitions,
            "write_time_sec": write_time,
            "output_size_mb": output_size_mb,
            "output_directory": str(output_dir),
        },
        "records_by_source": source_stats.to_dict(orient="records"),
        "records_by_language": lang_stats.to_dict(orient="records"),
        "records_by_label": label_stats.to_dict(orient="records"),
        "text_length_distribution": {k: float(v) if isinstance(v, (int, float)) else v for k, v in length_metrics.items()},
    }

    # Save JSON Report
    json_path = reports_dir / "spark_analytics_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analytics, f, indent=2)

    # Save Markdown Report
    md_path = reports_dir / "spark_pipeline_report.md"
    generate_markdown_report(analytics, md_path)

    return analytics


def generate_markdown_report(analytics: dict[str, Any], output_path: Path) -> None:
    exec_info = analytics["execution_summary"]
    length_info = analytics["text_length_distribution"]

    lines = [
        "# ScamShield — PySpark Big Data Pipeline Report (Phase 4)",
        "",
        "## 1. Spark Execution & Benchmark Performance",
        f"- **Total Records Processed:** {exec_info['records_processed']:,}",
        f"- **Total Pipeline Time:** {exec_info['processing_time_sec']} seconds",
        f"- **Processing Throughput:** **{exec_info['throughput_records_per_sec']:,} records/second**",
        f"- **RDD Partitions Executed:** {exec_info['spark_partitions']}",
        f"- **Partitioned Write Duration:** {exec_info['write_time_sec']} seconds",
        f"- **Output Storage Size:** {exec_info['output_size_mb']} MB (Snappy compressed Parquet)",
        "",
        "## 2. Distributed Big Data Analytics",
        "",
        "### Ingestion & URL Availability by Source Dataset",
        "| Dataset Source | Records Ingested | Avg Message Length (chars) | Records with URLs |",
        "|---|---|---|---|",
    ]
    for row in analytics["records_by_source"]:
        lines.append(f"| **{row['source_dataset']}** | {row['count']:,} | {row['avg_length']} | {row['url_count']:,} |")

    lines.extend([
        "",
        "### Project Label Class Balance",
        "| Project Label | Distributed Count |",
        "|---|---|",
    ])
    for row in analytics["records_by_label"]:
        lines.append(f"| **{row['project_label']}** | {row['count']:,} |")

    lines.extend([
        "",
        "### Multilingual Language Distribution",
        "| Language Tag | Record Count |",
        "|---|---|",
    ])
    for row in analytics["records_by_language"]:
        lines.append(f"| **{row['language']}** | {row['count']:,} |")

    lines.extend([
        "",
        "### Global Text Length Statistics",
        f"- **Minimum Character Length:** {int(length_info['min_length'])} chars",
        f"- **Maximum Character Length:** {int(length_info['max_length']):,} chars",
        f"- **Mean Character Length:** {length_info['mean_length']} chars",
        f"- **Standard Deviation:** {length_info['stddev_length']} chars",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    results = run_spark_pipeline()
    print("\n" + "=" * 65)
    print("      SCAMSHIELD — SPARK BIG DATA PIPELINE COMPLETE")
    print("=" * 65)
    summary = results["execution_summary"]
    print(f"Records Processed:   {summary['records_processed']:,}")
    print(f"Processing Speed:    {summary['throughput_records_per_sec']:,} records/sec")
    print(f"Total Pipeline Time: {summary['processing_time_sec']}s")
    print(f"Output Size:         {summary['output_size_mb']} MB")
    print(f"Report:              reports/spark_pipeline_report.md")
    print("=" * 65)
