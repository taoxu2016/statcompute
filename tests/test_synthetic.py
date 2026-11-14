"""The synthetic data are deterministic, independent of partitioning, and as described."""
import pytest

from statcompute.testkit import DatasetSpec, generate, materialize


def checksum(frame):
    columns = ", ".join(frame.columns)
    row = frame.selectExpr(f"bit_xor(xxhash64({columns})) AS c", "count(*) AS n").first()
    return int(row["c"]), int(row["n"])


def test_dataset_id_is_stable_and_sensitive():
    a = DatasetSpec("cox", 1000, 10, 4)
    assert a.dataset_id() == DatasetSpec("cox", 1000, 10, 4).dataset_id()
    assert a.dataset_id() != DatasetSpec("cox", 1000, 10, 4, seed=80).dataset_id()
    assert a.dataset_id().startswith("cox-n1000-m10-p4-simple-s79-")
    assert DatasetSpec.from_dict(a.to_dict()) == a


@pytest.mark.parametrize("family", ["cox", "logistic_fe"])
@pytest.mark.parametrize("design", ["simple", "correlated"])
def test_generation_does_not_depend_on_partitioning(spark, family, design):
    spec = DatasetSpec(family, 3000, 12, 6, design=design)
    one, seven = checksum(generate(spark, spec, partitions=1)), checksum(generate(spark, spec, partitions=7))
    assert one == seven and one[1] == 3000


def test_materialize_writes_a_manifest_and_reuses_the_table(spark, data_root):
    spec = DatasetSpec("cox", 4000, 16, 5)
    first = materialize(spark, spec, data_root)
    again = materialize(spark, spec, data_root)
    assert first.manifest == again.manifest and first.rows == 4000
    assert checksum(first.frame(spark))[0] == first.manifest["checksum"]
    assert 0.05 * 4000 < first.manifest["events"] < 0.95 * 4000
    sizes = [r["count"] for r in first.frame(spark).groupBy("provider").count().collect()]
    assert len(sizes) == 16 and set(sizes) == {250}
    times = [r["time"] for r in first.frame(spark).select("time").limit(200).collect()]
    assert all(t == int(t) and 1 <= t <= 1825 for t in times)


def test_the_correlated_design_has_the_stated_margins(spark):
    spec = DatasetSpec("logistic_fe", 60000, 20, 8, design="correlated")
    frame = generate(spark, spec)
    stats = frame.selectExpr(*[f"avg(x{k}) AS m{k}" for k in range(1, 9)],
                             *[f"stddev(x{k}) AS s{k}" for k in range(1, 9)]).first()
    for k in range(1, 9):
        if spec.is_indicator(k):
            assert abs(stats[f"m{k}"] - spec.prevalence(k)) < 4 * (spec.prevalence(k) / 60000) ** 0.5 + 1e-3
        else:
            assert abs(stats[f"s{k}"] / spec.scale(k) - 1) < 0.03
