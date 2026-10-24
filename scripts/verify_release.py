"""Verify the published source/configuration bytes without GPU dependencies."""
import json
from common import ROOT, sha


def main():
    manifest = json.loads((ROOT / 'provenance/release_code_sha256.json').read_text())
    failures = []
    for relative, expected in manifest['files'].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            failures.append(relative)
    if failures:
        raise SystemExit('Release file mismatch: ' + ', '.join(failures))
    print(f"RELEASE_VERIFIED: {len(manifest['files'])} source/configuration files")


if __name__ == '__main__':
    main()
