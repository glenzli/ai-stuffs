// Bundle creation owns output files only; accepting a bundle is a separate operation.
import { readFile, writeFile, readdir, mkdir, mkdtemp, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { spawn, execFile } from "node:child_process";
import { promisify } from "node:util";
import { tmpdir } from "node:os";
import path from "node:path";

export const hash = bytes => createHash("sha256").update(bytes).digest("hex");
export const readJSON = async file => JSON.parse(await readFile(file, "utf8"));
export const describe = async file => { const bytes = await readFile(file); return { path: file, bytes: bytes.length, sha256: hash(bytes) }; };
export function relativePath(file) {
  if (typeof file !== "string" || path.isAbsolute(file) || file.split(/[\\/]/).some(p => !p || p === "." || p === "..")) throw new Error(`Unsafe relative path: ${file}`);
  return file;
}
export async function filesIn(directory) {
  const files = [];
  for (const item of await readdir(directory, { withFileTypes: true })) {
    const file = path.join(directory, item.name);
    if (item.isDirectory()) files.push(...await filesIn(file));
    else if (item.isFile()) files.push(file);
    else throw new Error(`Symlinks and special files are not evidence: ${file}`);
  }
  return files.sort();
}
export async function writeChecksums(directory, output) {
  const rows = [];
  for (const file of await filesIn(directory)) rows.push(`${hash(await readFile(file))}  ${path.relative(directory, file)}`);
  await writeFile(output, rows.join("\n") + "\n", { flag: "wx" });
}
export async function checkAsset(file, root = process.cwd()) {
  relativePath(file?.path);
  const bytes = await readFile(path.join(root, file.path));
  if (bytes.length !== file.bytes || hash(bytes) !== file.sha256) throw new Error(`Evidence identity mismatch: ${file.path}`);
  return bytes;
}
export async function checkChecksumFile(file, directory) {
  const lines = (await readFile(file, "utf8")).trim().split("\n");
  if (!lines.length || !lines[0]) throw new Error("Empty evidence checksum list");
  const seen = new Set();
  for (const line of lines) {
    const match = /^([a-f0-9]{64})  (.+)$/.exec(line);
    if (!match) throw new Error("Invalid evidence checksum line");
    const [, sha, name] = match; relativePath(name);
    if (seen.has(name)) throw new Error(`Repeated checksum path: ${name}`); seen.add(name);
    if (hash(await readFile(path.join(directory, name))) !== sha) throw new Error(`Raw evidence changed: ${name}`);
  }
}
export async function archiveBundle(root, id) {
  await writeChecksums(`${root}/${id}`, `${root}/${id}/SHA256SUMS`);
  const rows = [];
  for (const file of await filesIn(`${root}/source`)) rows.push(`${hash(await readFile(file))}  ${path.relative(root, file)}`);
  await writeFile(`${root}/SOURCE-SHA256SUMS`, rows.join("\n") + "\n", { flag: "wx" });
  await mkdir(`${root}/downloads`);
  const archive = `${root}/downloads/${id}-sources.tar.gz`;
  await new Promise((resolve, reject) => {
    const child = spawn("tar", ["-czf", path.resolve(archive), "-C", path.resolve(root), id, "source", "SOURCE-SHA256SUMS"], { stdio: "inherit" });
    child.once("error", reject); child.once("exit", code => code === 0 ? resolve() : reject(new Error(`Archive failed: ${id}`)));
  });
  return { report: await describe(`${root}/${id}/report.json`), archive: await describe(archive),
    checksums: await describe(`${root}/${id}/SHA256SUMS`), source: await describe(`${root}/SOURCE-SHA256SUMS`) };
}
export async function checkArchive(entry, root = process.cwd()) {
  const run = promisify(execFile), archive = path.join(root, entry.archive.path);
  const listing = await run('tar', ['-tzf', archive]);
  const seen = new Set();
  for (const line of listing.stdout.trim().split('\n')) {
    const file = line.replace(/\/+$/, ''); relativePath(file);
    if (seen.has(file) || !(file === entry.id || file.startsWith(entry.id + '/') || file === 'source' || file.startsWith('source/') || file === 'SOURCE-SHA256SUMS')) throw new Error('Unexpected or repeated archive entry');
    seen.add(file);
  }
  const verbose = await run('tar', ['-tvzf', archive]);
  if (verbose.stdout.trim().split('\n').some(line => !['d', '-'].includes(line[0]))) throw new Error('Links and special entries are not accepted in evidence archives');
  const directory = await mkdtemp(path.join(tmpdir(), 'ai-stuffs-accept-'));
  try {
    await run('tar', ['-xzf', archive, '-C', directory]);
    const saved = path.join(directory, entry.id);
    if (hash(await readFile(path.join(saved, 'report.json'))) !== entry.report.sha256 || hash(await readFile(path.join(saved, 'SHA256SUMS'))) !== entry.checksums.sha256 || hash(await readFile(path.join(directory, 'SOURCE-SHA256SUMS'))) !== entry.source.sha256) throw new Error('Archive envelope does not match the candidate');
    await checkChecksumFile(path.join(saved, 'SHA256SUMS'), saved);
    await checkChecksumFile(path.join(directory, 'SOURCE-SHA256SUMS'), directory);
  } finally { await rm(directory, { recursive: true, force: true }); }
}
export async function writeCandidate({ root, entry, projection, baseManifestSha256 }) {
  relativePath(root);
  if (!root.startsWith("artifacts/")) throw new Error("Experiment output must be under artifacts/");
  const candidate = { schema: "experiment-candidate-v1", base_manifest_sha256: baseManifestSha256,
    bundle_root: root, entry: { ...entry, bundle_root: root }, projection };
  await writeFile(`${root}/candidate.json`, JSON.stringify(candidate, null, 2) + "\n", { flag: "wx" });
  return candidate;
}
