#!/usr/bin/env node
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const routeDir = dirname(fileURLToPath(import.meta.url));
const pluginRoot = resolve(routeDir, "../../../../..");
const registry = JSON.parse(readFileSync(join(routeDir, "route-registry.json"), "utf8"));
const requiredDomains = new Set(["branding-design", "content-copywriting", "e-commerce-products", "email-marketing-automation", "sales-funnels", "seo-search", "social-media"]);
const ids = registry.routes.map(route => route.id);
const errors = [];
const stages = new Set(["discover", "design", "operate", "measure"]);
if (registry.routes.length !== 234) errors.push(`expected 234 routes; observed ${registry.routes.length}`);
if (new Set(ids).size !== ids.length) errors.push("route IDs are not unique");
for (const route of registry.routes) {
  for (const field of ["id", "triggers", "owning_roles", "brain_packs", "lifecycle_stage", "commerce_surface", "source_auth", "prohibited_uses", "lazy_loading"]) if (route[field] == null) errors.push(`${route.id}: missing ${field}`);
  if (!requiredDomains.has(route.source?.domain)) errors.push(`${route.id}: unknown domain`);
  if (!stages.has(route.lifecycle_stage)) errors.push(`${route.id}: unknown lifecycle stage`);
  if (!Array.isArray(route.triggers?.phrases) || route.triggers.phrases.length === 0) errors.push(`${route.id}: triggers missing`);
  if (!Array.isArray(route.owning_roles) || route.owning_roles.length === 0) errors.push(`${route.id}: owner missing`);
  if (!Array.isArray(route.brain_packs) || route.brain_packs.length === 0) errors.push(`${route.id}: Brain packs missing`);
  if (route.source_auth?.missing_evidence_disposition !== "UNVERIFIED") errors.push(`${route.id}: evidence does not fail closed`);
  if (!route.prohibited_uses?.includes("trust_filename_as_identity")) errors.push(`${route.id}: filename guard missing`);
  const sourceFile = join(pluginRoot, registry.generated_from.root, route.source.relative_path);
  if (!existsSync(sourceFile)) errors.push(`${route.id}: source file missing`);
  else if (createHash("sha256").update(readFileSync(sourceFile)).digest("hex") !== route.source.sha256) errors.push(`${route.id}: source hash stale`);
}
for (const domain of requiredDomains) if (!registry.routes.some(route => route.source.domain === domain)) errors.push(`domain missing: ${domain}`);
if (errors.length) { console.error(errors.join("\n")); process.exit(1); }
console.log(JSON.stringify({ verdict: "PASS_PLUGIN_ROUTE_INTEGRITY_ONLY", route_count: registry.routes.length, domain_count: requiredDomains.size, unique_ids: new Set(ids).size, aggregate_sha256: registry.generated_from.aggregate_sha256 }));
