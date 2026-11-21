"""Freezes the specification of a real-photograph stage before it is run, as the study did: writes results/<stage>_spec.sha256.
python tools/freeze.py m3 | m3c | m4   (freeze in that order; each stage's extraction script refuses to run if any spec file changes afterwards)"""
import sys, os, datetime, importlib.util
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
stage = sys.argv[1]; path = os.path.join(HERE, "tests", f"{stage}_extract.py")
spec = importlib.util.spec_from_file_location(f"{stage}_extract", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
h = m.spec_hash(); now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
open(os.path.join(HERE, "results", f"{stage}_spec.sha256"), "w").write(f"{h} frozen {now}\n"); print("frozen", stage, h, now, len(m.spec_files()), "files")
