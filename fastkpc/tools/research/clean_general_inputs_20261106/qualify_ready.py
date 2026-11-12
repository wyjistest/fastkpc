"""Qualify completed references while the final original-R job is still running."""
import importlib.util
from pathlib import Path
import shutil
import sys
import time

TOOL = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ready_campaign", TOOL / "campaign.py")
driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
build_tool = driver.build_module
BASE, CHECKOUT = driver.BASE, driver.CHECKOUT
sha, write, read = driver.sha, driver.write, driver.read
ORDER = ["real179", "real243", "real300", "public500", "blocks256", "dense1025", "dense500"]

def main():
    tag, reference_tag = sys.argv[1:3]
    assert Path(tag).name == tag and Path(reference_tag).name == reference_tag
    out = BASE / tag; out.mkdir()
    previous = BASE / reference_tag
    reference_protocol = read(previous / "protocol.json")
    build = read(BASE / "build/build.json")
    assert build["status"] == "PASS_CLEAN_FULL_BUILD"
    assert reference_protocol["producer_commit"] == build_tool.COMMIT
    assert driver.git("rev-parse", "HEAD") == build_tool.COMMIT
    inputs = read(BASE / "inputs/manifest.json")
    files = {str(CHECKOUT/p): h for p, h in build["source_sha256"].items()}
    files.update({str(p): sha(p) for p in TOOL.iterdir() if p.is_file()})
    files.update({item["path"]: item["file_sha256"] for item in inputs.values()})
    files[build["native_path"]] = build["native_sha256"]
    files[str(previous / "protocol.json")] = sha(previous / "protocol.json")
    snapshot = out / "tools"; snapshot.mkdir()
    for p in TOOL.iterdir():
        if p.is_file(): shutil.copyfile(p, snapshot / p.name)
    def frozen():
        assert not driver.git("status", "--porcelain")
        for p, expected in files.items(): assert sha(p) == expected, p
    frozen()
    write(out / "protocol.json", dict(action="qualify", producer_commit=build_tool.COMMIT,
        native_sha256=build["native_sha256"], previous=str(previous), cases=ORDER,
        inputs=inputs, dependencies=files.copy(), order=["off", "on"], tolerance=1e-10,
        reference_binding="per-case reference-bindings JSON created before replay or GPU execution",
        scheduling="six ready inputs first; wait for final dense500 original-R reference",
        evidence_class="correctness only; CPU reference may run concurrently",
        phase10="ACTIVE", candidate="NOT_FROZEN", holdout="SEALED", recommended=False))
    env = driver.environment(); reports = {}
    for case in ORDER:
        ref = previous / case / "receipt.rds"
        completion = previous / (case + "-process.json")
        while not completion.is_file():
            print("WAIT original-R reference", case, flush=True)
            time.sleep(30)
        process = read(completion)
        report_path = previous / case / "report.json"
        assert process["exit_code"] == 0 and read(report_path)["pass"], case
        bindings = {str(p): sha(p) for p in (ref, completion, report_path)}
        files.update(bindings)
        write(out / (case + "-reference-bindings.json"), dict(case=case, dependencies=bindings))
        def command(mode, stem, reference):
            return ["taskset", "-c", "0-19", "Rscript", "--vanilla", str(TOOL / "run.R"),
                mode, inputs[case]["path"], str(out/stem), str(reference), "correctness"]
        replay = case + "-replay"
        driver.job(out, replay, command("replay", replay, ref), env, frozen, "correctness")
        reports[case] = {}
        for mode in ("off", "on"):
            stem = case + "-" + mode
            report = driver.job(out, stem, command(mode, stem, out/replay/"receipt.rds"),
                env, frozen, "correctness")
            assert report["native_sha256"] == build["native_sha256"]
            reports[case][mode] = report
    while not (previous/"summary.json").is_file(): time.sleep(5)
    prior = read(previous/"summary.json")
    assert prior["pass"] and set(prior["reports"]) == set(reports)
    frozen()
    write(out/"summary.json", dict(action="qualify", reports=reports,
        producer_commit=build_tool.COMMIT, native_sha256=build["native_sha256"],
        evidence_class="fresh original-reference correctness", **{"pass": True}))
    print("PASS qualify", len(reports), "inputs", flush=True)

if __name__ == "__main__":
    main()
