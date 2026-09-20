"""Check tracked release files, recursively including gzip, tar and zip contents."""

import gzip
import io
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path

from dotenv import dotenv_values

# Match private administrative material, not public benchmark subjects or model traces.
MARKERS = re.compile(
    r"https?://[^\s\"<>]*(?:linear\.app|notion\.so|slack\.com)"
    r"|\bSLA-\d+\b|/(?:Users|home)/[^\s\"]+"
    r"|assistant\s+misinterpreted|user\s+accepted|used\s+Terra\s+in\s+error"
    r"|account\s+had\s+no\s+credits|credit_balance_(?:exhausted)|residential\s+Wi-Fi",
    re.I,
)


def inspect(name, data, secrets=(), depth=0):
    if depth > 6:
        raise ValueError("Archive nesting exceeds inspection limit")
    findings = []
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    if len(data) > 262 and data[257:262] == b"ustar":
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive:
                if member.uname or member.gname:
                    findings.append(name + "!" + member.name + ": archive owner metadata")
                if member.isfile():
                    findings.extend(
                        inspect(
                            name + "!" + member.name,
                            archive.extractfile(member).read(),
                            secrets,
                            depth + 1,
                        )
                    )
        return findings
    if data[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.namelist():
                findings.extend(
                    inspect(name + "!" + member, archive.read(member), secrets, depth + 1)
                )
        return findings
    if MARKERS.search(data.decode("utf-8", errors="replace")):
        findings.append(name + ": private path, link or administrative prose")
    if any(secret in data for secret in secrets):
        findings.append(name + ": credential value (not displayed)")
    return findings


def main():
    secrets = tuple(
        value.encode()
        for key, value in dotenv_values(".env").items()
        if value and len(value) > 8 and any(word in key for word in ("KEY", "TOKEN", "SECRET"))
    )
    names = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    findings = [
        finding
        for name in names
        if name
        for finding in inspect(name, Path(name).read_bytes(), secrets)
    ]
    if findings:
        raise SystemExit("\n".join(findings))
    print(
        "PASS: tracked files and nested archives; no configured privacy markers or credential values. Git history requires a separate review."
    )


if __name__ == "__main__":
    main()
