import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    session = (SparkSession.builder.master("local[2]").appName("statcompute-tests")
               .config("spark.sql.shuffle.partitions", "4").config("spark.driver.memory", "1g")
               .config("spark.ui.enabled", "true").config("spark.ui.showConsoleProgress", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    return str(tmp_path_factory.mktemp("datasets"))
