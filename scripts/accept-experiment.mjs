// One mutation owner for the accepted report, display projection and Git inclusion rules.
import { readFile, writeFile, mkdir, rename, rm, access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { hash, readJSON, checkAsset, checkChecksumFile, checkArchive, relativePath } from "./experiment-bundles.mjs";
import { experimentOwners } from "./experiment-sources.mjs";

const lockName = ".cache/experiment-adoption.lock";
const ignoreMarker = "# Accepted experiment bundles (managed by accept-experiment.mjs)";
const ignoreEnd = "# End accepted experiment bundles";
const absent = async file => { try { await access(file); return false; } catch (e) { if (e.code === "ENOENT") return true; throw e; } };
async function oldBytes(file) { try { return (await readFile(file)).toString("base64"); } catch (e) { if (e.code === "ENOENT") return null; throw e; } }
async function atomicWrite(file, bytes) {
  await mkdir(path.dirname(file), { recursive: true });
  const temporary = `${file}.adopting-${process.pid}`;
  await writeFile(temporary, bytes, { flag: "wx" });
  try { await rename(temporary, file); } finally { await rm(temporary, { force: true }); }
}
function inclusionRules(original, manifest) {
  const roots = [...new Set(Object.values(manifest.experiments).map(e => e.bundle_root).filter(Boolean))].sort();
  const allowed = new Set(roots), parents = new Set();
  for (const root of roots) {
    relativePath(root); if (!root.startsWith("artifacts/")) throw new Error("Evidence is outside artifacts/");
    const parts = root.split("/"); for (let i = 2; i <= parts.length; i++) parents.add(parts.slice(0, i).join("/"));
  }
  const lines = [];
  for (const directory of [...parents].sort()) {
    if (roots.some(r => r !== directory && directory.startsWith(r + "/"))) continue;
    lines.push(`!/${directory}/`);
    if (!allowed.has(directory)) lines.push(`/${directory}/*`);
    else lines.push(`!/${directory}/**`);
  }
  const suffix = original.includes(ignoreEnd) ? original.slice(original.indexOf(ignoreEnd) + ignoreEnd.length).trim() : "";
  return original.split(ignoreMarker)[0].trimEnd() + `\n\n${ignoreMarker}\n` + lines.join("\n") + `\n${ignoreEnd}\n` + (suffix ? `\n${suffix}\n` : "");
}
export async function recoverAdoption(root = process.cwd()) {
  const lock = path.join(root, lockName); if (await absent(lock)) return { recovered: false };
  const journalFile = path.join(lock, "journal.json");
  if (await absent(journalFile)) throw new Error("Adoption lock has no complete journal; inspect the interrupted writer before removing the lock.");
  const journal = await readJSON(journalFile);
  try { process.kill(journal.pid, 0); throw new Error("An adoption process is still active."); } catch (e) { if (e.code !== "ESRCH") throw e; }
  if (!journal.committed) for (const item of [...journal.files].reverse()) {
    if (item.previous === null) await rm(path.join(root, item.file), { force: true });
    else await atomicWrite(path.join(root, item.file), Buffer.from(item.previous, "base64"));
  }
  await rm(lock, { recursive: true }); return { recovered: true, kept: journal.committed ? "accepted" : "previous" };
}
export async function acceptExperiments(candidates, { root = process.cwd(), afterWrite } = {}) {
  const manifestPath = path.join(root, "src/experiment-sources.json"), previousBytes = await readFile(manifestPath);
  const previous = JSON.parse(previousBytes), next = structuredClone(previous), updates = new Map();
  const receipts = []; let alreadyAccepted = 0;
  for (const file of candidates) {
    const candidate = await readJSON(path.resolve(root, file));
    const acceptedValue = { ...candidate.entry, ...(candidate.projection ? { projection: candidate.projection } : {}) };
    const already = candidate.entry?.id && JSON.stringify(previous.experiments[candidate.entry.id]) === JSON.stringify(acceptedValue);
    if (candidate.schema !== "experiment-candidate-v1" || (!already && candidate.base_manifest_sha256 !== hash(previousBytes))) throw new Error("Stale candidate: the accepted manifest changed; regenerate the candidate against the current base.");
    if (already) alreadyAccepted++;
    const { entry, projection } = candidate;
    if (!entry?.id || entry.status !== "passed" || !Array.isArray(entry.pages) || !entry.pages.length || entry.bundle_root !== candidate.bundle_root) throw new Error("Incomplete experiment candidate");
    if (JSON.stringify(entry.pages) !== JSON.stringify(experimentOwners[entry.id])) throw new Error("Candidate does not own these pages");
    relativePath(candidate.bundle_root);
    if (!/^artifacts\/[\w-]+(?:\/[\w-]+)*$/.test(candidate.bundle_root)) throw new Error("Evidence must remain in a named artifacts/ folder");
    for (const kind of ["report", "archive", "checksums", "source"]) {
      if (!entry[kind]?.path?.startsWith(candidate.bundle_root + "/")) throw new Error("Candidate references a different bundle");
      await checkAsset(entry[kind], root);
    }
    const report = JSON.parse(await checkAsset(entry.report, root));
    if (report.schema !== entry.schema || report.status !== "passed") throw new Error("Report schema or status differs from candidate");
    await checkChecksumFile(path.join(root, entry.checksums.path), path.dirname(path.join(root, entry.report.path)));
    await checkChecksumFile(path.join(root, entry.source.path), path.join(root, candidate.bundle_root));
    await checkArchive(entry, root);
    if (projection) {
      if (!/^src\/scripts\/features\/[a-z0-9-]+-data\.js$/.test(projection.target) || !projection.path.startsWith(candidate.bundle_root + "/")) throw new Error("Invalid display projection target");
      const bytes = await checkAsset(projection, root);
      if (updates.has(projection.target)) throw new Error("Two candidates own the same display projection");
      updates.set(projection.target, bytes);
    }
    if (receipts.some(r => r.id === entry.id)) throw new Error("Duplicate experiment ID in adoption");
    next.experiments[entry.id] = { ...entry, ...(projection ? { projection } : {}) }; receipts.push({ id: entry.id, report_sha256: entry.report.sha256 });
  }
  if (alreadyAccepted === candidates.length) {
    for (const [target, bytes] of updates) if (hash(await readFile(path.join(root, target))) !== hash(bytes)) throw new Error('Accepted display projection changed');
    return { accepted: receipts, already_accepted: true, manifest_sha256: hash(previousBytes) };
  }
  updates.set("src/experiment-sources.json", Buffer.from(JSON.stringify(next, null, 2) + "\n"));
  updates.set(".gitignore", Buffer.from(inclusionRules(await readFile(path.join(root, ".gitignore"), "utf8"), next)));
  const lock = path.join(root, lockName); await mkdir(path.dirname(lock), { recursive: true });
  await mkdir(lock); // Exclusive writer; readers reject this directory before replacing dist.
  const journal = { pid: process.pid, committed: false, files: [] };
  try {
    if (hash(await readFile(manifestPath)) !== hash(previousBytes)) throw new Error("Accepted manifest changed during preparation");
    // Recheck evidence at the write boundary, after acquiring the writer lock.
    for (const receipt of receipts) {
      const entry = next.experiments[receipt.id];
      for (const key of ["report", "archive", "checksums", "source"]) await checkAsset(entry[key], root);
      await checkChecksumFile(path.join(root, entry.checksums.path), path.dirname(path.join(root, entry.report.path)));
      await checkChecksumFile(path.join(root, entry.source.path), path.join(root, entry.bundle_root));
      if (entry.projection) await checkAsset(entry.projection, root);
    }
    for (const [file, bytes] of updates) journal.files.push({ file, previous: await oldBytes(path.join(root, file)), next_sha256: hash(bytes) });
    await atomicWrite(path.join(lock, "journal.json"), Buffer.from(JSON.stringify(journal)));
    for (const [file, bytes] of updates) { await atomicWrite(path.join(root, file), bytes); await afterWrite?.(file); }
    journal.committed = true; await atomicWrite(path.join(lock, "journal.json"), Buffer.from(JSON.stringify(journal)));
    await rm(lock, { recursive: true }); return { accepted: receipts, manifest_sha256: hash(updates.get("src/experiment-sources.json")) };
  } catch (error) {
    for (const item of [...journal.files].reverse()) {
      if (item.previous === null) await rm(path.join(root, item.file), { force: true });
      else await atomicWrite(path.join(root, item.file), Buffer.from(item.previous, "base64"));
    }
    await rm(lock, { recursive: true }); throw error;
  }
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  if (args.length === 1 && args[0] === "--recover") console.log(JSON.stringify(await recoverAdoption()));
  else if (args[0] === "--candidate" && args.length >= 2) console.log(JSON.stringify(await acceptExperiments(args.slice(1))));
  else throw new Error("Usage: node scripts/accept-experiment.mjs --candidate <candidate.json> [other-candidate.json ...] | --recover");
}
