import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from agent_pipeline_benchmark.corpus import WorkItem
from agent_pipeline_benchmark.subprocesses import environment_without_virtualenv


@dataclass(frozen=True)
class TestMovements:
    progressed: int
    preserved: int
    regressed: int

    @property
    def solved(self) -> bool:
        return self.progressed > 0 and self.regressed == 0

    __test__ = False


@dataclass(frozen=True)
class TestVerdict:
    name: str
    passed: bool

    __test__ = False


def score_hidden_tests(work_item: WorkItem, working_copy: Path) -> list[TestVerdict]:
    destination = working_copy / "tests"
    copied_files = copy_hidden_tests(work_item.hidden_tests, destination)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            junit_xml = Path(tmp) / "junit.xml"
            run_full_suite(junit_xml, working_copy)
            return record_every_hidden_test(junit_test_verdicts(junit_xml), work_item)
    finally:
        remove_copied_files(copied_files)


def passing_test_ids(work_item: WorkItem, working_copy: Path) -> frozenset[str]:
    return frozenset(verdict.name for verdict in score_hidden_tests(work_item, working_copy) if verdict.passed)


def test_movements(before: frozenset[str], after: frozenset[str]) -> TestMovements:
    return TestMovements(
        progressed=len(after - before),
        preserved=len(after & before),
        regressed=len(before - after),
    )


test_movements.__test__ = False


def run_full_suite(junit_xml: Path, working_copy: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["uv", "run", "pytest", "-q", "--continue-on-collection-errors", "--junitxml", str(junit_xml)],
        cwd=working_copy,
        env=environment_without_virtualenv(),
        capture_output=True,
        check=False,
    )


def junit_test_verdicts(junit_xml: Path) -> list[TestVerdict]:
    root = ET.parse(junit_xml).getroot()
    return [
        TestVerdict(
            name=f"{testcase.attrib['classname']}::{testcase.attrib['name']}",
            passed=not any(
                testcase.find(tag) is not None for tag in ("skipped", "failure", "error")
            ),
        )
        for testcase in root.iter("testcase")
    ]


def parse_passing_test_ids(junit_xml: Path) -> frozenset[str]:
    return frozenset(verdict.name for verdict in junit_test_verdicts(junit_xml) if verdict.passed)


def record_every_hidden_test(verdicts: list[TestVerdict], work_item: WorkItem) -> list[TestVerdict]:
    recorded = {verdict.name.rsplit("::", 1)[-1] for verdict in verdicts}
    missing = [
        name for name in hidden_test_function_names(work_item.hidden_tests) if name not in recorded
    ]
    return [*verdicts, *[TestVerdict(name=name, passed=False) for name in missing]]


def hidden_test_function_names(hidden_tests: Path) -> list[str]:
    return [
        match.group(1)
        for test_file in hidden_tests.rglob("test_*.py")
        for match in re.finditer(r"def (test_\w+)", test_file.read_text())
    ]


def copy_hidden_tests(hidden_tests: Path, destination: Path) -> list[Path]:
    copied_files = []
    for source in hidden_tests.rglob("*"):
        if source.is_file():
            target = destination / source.relative_to(hidden_tests)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied_files.append(target)
    return copied_files


def remove_copied_files(copied_files: list[Path]) -> None:
    for path in copied_files:
        path.unlink(missing_ok=True)
