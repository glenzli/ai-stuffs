import { readFile, readdir } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { pages, site } from "../src/pages.mjs";
import { searchMetadata } from "../src/search.mjs";
import { pageEvidence, sources } from "../src/references.mjs";
import { wikiGroups } from "../src/wiki.mjs";
import { parseFragment } from "parse5";
import { learningLinks, renderLearningPath } from "./render-learning.mjs";

const root = process.cwd();
const issues = [];

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await walk(target));
    else files.push(target);
  }
  return files;
}

const jsFiles = await walk(path.join(root, "src/scripts"));
for (const file of jsFiles.filter((item) => item.endsWith(".js"))) {
  const check = spawnSync(process.execPath, ["--check", file], { encoding: "utf8" });
  if (check.status !== 0) issues.push(`${path.relative(root, file)}: ${check.stderr.trim()}`);
}

const cssFiles = await walk(path.join(root, "src/styles"));
for (const file of cssFiles.filter((item) => item.endsWith(".css"))) {
  const css = await readFile(file, "utf8");
  let depth = 0;
  for (const char of css) {
    if (char === "{") depth += 1;
    if (char === "}") depth -= 1;
    if (depth < 0) break;
  }
  if (depth !== 0) issues.push(`${path.relative(root, file)}: unbalanced braces (${depth})`);
}

const slugs = new Set();
const files = new Set();
const expectedContent = new Set();
const knownStyles = new Set(cssFiles);
const knownFeatures = new Set(jsFiles);
const authoredPages = new Map();
if (!pages.length) issues.push("no pages found in src/pages.mjs");
for (const page of pages) {
  if (slugs.has(page.slug)) issues.push(`duplicate page slug: ${page.slug}`);
  slugs.add(page.slug);
  if (files.has(page.file)) issues.push(`duplicate page file: ${page.file}`);
  files.add(page.file);
  const contentFile = path.join(root, "src/content", `${page.slug}.html`);
  expectedContent.add(contentFile);
  try {
    authoredPages.set(page.file, await readFile(contentFile, "utf8"));
  } catch {
    issues.push(`missing content for page: ${page.slug}`);
  }
  if (!site.navigation.some((item) => item.href === page.nav)) {
    issues.push(`unknown navigation target for page: ${page.slug}`);
  }
  if (!Object.hasOwn(searchMetadata, page.file) || !searchMetadata[page.file]) {
    issues.push(`missing search metadata for page: ${page.file}`);
  }
  if (!Object.hasOwn(pageEvidence, page.slug) || !pageEvidence[page.slug]) {
    issues.push(`missing evidence metadata for page: ${page.slug}`);
  }
  for (const style of page.styles) {
    const file = path.join(root, "src/styles", `${style}.css`);
    if (!knownStyles.has(file)) {
      issues.push(`missing stylesheet for page ${page.slug}: ${path.relative(root, file)}`);
    }
  }
  for (const feature of page.features) {
    const file = path.join(root, "src/scripts/features", `${feature}.js`);
    if (!knownFeatures.has(file)) {
      issues.push(`missing feature module for page ${page.slug}: ${path.relative(root, file)}`);
    }
  }
}

for (const item of site.navigation) {
  if (!files.has(item.href)) issues.push(`orphan navigation page file: ${item.href}`);
}
for (const file of Object.keys(searchMetadata)) {
  if (!files.has(file)) issues.push(`orphan search page file: ${file}`);
}
for (const [slug, evidence] of Object.entries(pageEvidence)) {
  if (!slugs.has(slug)) issues.push(`orphan evidence page slug: ${slug}`);
  if (!Array.isArray(evidence?.refs) || !evidence.refs.length) {
    issues.push(`missing reference keys for page: ${slug}`);
    continue;
  }
  for (const key of evidence.refs) {
    if (typeof key !== "string" || !key.trim() || !Object.hasOwn(sources, key)) {
      issues.push(`invalid reference key for page ${slug}: ${String(key)}`);
    }
  }
}
for (const [key, source] of Object.entries(sources)) {
  if (!key.trim()) issues.push("invalid reference key: empty key");
  try {
    if (typeof source?.url !== "string" || new URL(source.url).protocol !== "https:") {
      throw new Error("expected an absolute HTTPS URL");
    }
  } catch {
    issues.push(`invalid HTTPS URL for reference ${key}: ${source?.url}`);
  }
}

const wikiCounts = new Map();
for (const group of wikiGroups) {
  for (const [slug] of group.pages) {
    if (!slugs.has(slug)) issues.push(`orphan wiki page slug in ${group.title}: ${slug}`);
    wikiCounts.set(slug, (wikiCounts.get(slug) || 0) + 1);
  }
}
for (const slug of slugs) {
  const count = wikiCounts.get(slug) || 0;
  if (count !== 1) issues.push(`wiki entry count for page ${slug}: expected 1, found ${count}`);
}

const authoredIds = new Map();
for (const [file, source] of authoredPages) {
  const root = parseFragment(source.replaceAll("{{learningPathDetailed}}", renderLearningPath(true)));
  const ids = new Set();
  const visit = node => {
    const id = node.attrs?.find(item => item.name === "id")?.value;
    if (id) ids.add(id);
    for (const child of node.childNodes || []) visit(child);
  };
  visit(root); authoredIds.set(file, ids);
}
for (const [href] of learningLinks()) {
  const [file, id] = href.split("#");
  if (!files.has(file)) issues.push(`learning route links to missing page: ${href}`);
  else if (id && !authoredIds.get(file)?.has(id)) issues.push(`learning route links to missing anchor: ${href}`);
}

const contentFiles = await walk(path.join(root, "src/content"));
for (const file of contentFiles.filter((item) => item.endsWith(".html"))) {
  if (!expectedContent.has(file)) issues.push(`orphan content HTML: ${path.relative(root, file)}`);
}

if (issues.length) {
  console.error(issues.join("\n"));
  process.exitCode = 1;
} else {
  console.log(`Source check passed: ${pages.length} pages, ${jsFiles.length} scripts, ${cssFiles.length} styles.`);
}
