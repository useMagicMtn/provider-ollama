#!/bin/sh
set -eu
cd "$(dirname "$0")"
cargo build --release --locked
python3 - <<'PACKAGE'
import hashlib, io, json, platform, tarfile, tomllib
from pathlib import Path
crate=tomllib.loads(Path('Cargo.toml').read_text())['package']
name=crate['name'].removeprefix('magicmtn-provider-')
exe=Path('target/release')/crate['name']
os={'Darwin':'macos','Linux':'linux','Windows':'windows'}[platform.system()]
arch={'arm64':'aarch64','aarch64':'aarch64','x86_64':'x86_64','AMD64':'x86_64'}[platform.machine()]
manifest={'name':name,'version':crate['version'],'protocol_version':1,'os':os,'arch':arch,'sha256':hashlib.sha256(exe.read_bytes()).hexdigest()}
Path('dist').mkdir(exist_ok=True)
output=Path('dist')/f"{name}-{crate['version']}-{os}-{arch}.tar.gz"
with tarfile.open(output,'w:gz',format=tarfile.USTAR_FORMAT) as archive:
    data=json.dumps(manifest).encode(); info=tarfile.TarInfo('package.json');info.size=len(data)
    archive.addfile(info,io.BytesIO(data))
    archive.add(exe,arcname='provider');archive.add('LICENSE',arcname='LICENSE')
print(output)
PACKAGE
