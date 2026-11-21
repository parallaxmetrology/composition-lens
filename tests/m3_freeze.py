import sys, os, datetime
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(HERE, "tests")); sys.path.insert(0, HERE)
from m3_extract import spec_hash, spec_files
h = spec_hash(); now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
open(os.path.join(HERE, "results", "m3_spec.sha256"), "w").write(f"{h} frozen {now}\n"); open(os.path.join(HERE, "results", "m3_spec_manifest.txt"), "w").write("\n".join(spec_files()) + "\n")
print("frozen", h, now, len(spec_files()), "files")
