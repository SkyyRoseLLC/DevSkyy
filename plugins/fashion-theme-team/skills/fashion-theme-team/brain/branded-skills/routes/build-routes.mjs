#!/usr/bin/env node
import { createHash } from "node:crypto";
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const outputDir = dirname(fileURLToPath(import.meta.url));
const pluginRoot = resolve(outputDir, "../../../../..");
const sourceRoot = join(pluginRoot, "vendor/branded-skills");
const domains = {
  "branding-design": { stage: "discover", surface: "brand-system", owners: ["fashion-brand-systems-researcher", "brand-experience-architect"], packs: ["brand/skyyrose-artifact-system.json", "knowledge/do-dont.md"], prohibited: ["invent_brand_fact", "alter_logo_without_founder_approval", "treat_competitor_reference_as_canon"] },
  "content-copywriting": { stage: "design", surface: "content", owners: ["fashion-accessibility-content-engineer", "brand-experience-architect"], packs: ["knowledge/do-dont.md", "pages/page-blueprints.md"], prohibited: ["invent_product_fact", "invent_claim_or_testimonial", "publish_without_authority"] },
  "e-commerce-products": { stage: "design", surface: "commerce-service", owners: ["fashion-commerce-strategist", "fashion-product-fit-returns-specialist"], packs: ["knowledge/fashion-commerce-fundamentals.md", "knowledge/fit-imagery-and-returns.md"], prohibited: ["invent_catalog_or_policy", "mutate_live_commerce", "claim_unmeasured_uplift"] },
  "email-marketing-automation": { stage: "operate", surface: "lifecycle-messaging", owners: ["fashion-merchandising-conversion-architect", "ecommerce-growth-analytics-engineer"], packs: ["knowledge/merchandising-and-conversion.md", "prompts/prompt-orchestration.md"], prohibited: ["send_without_authority", "invent_urgency_or_consent", "claim_unmeasured_uplift"] },
  "sales-funnels": { stage: "design", surface: "funnel", owners: ["fashion-commerce-strategist", "fashion-merchandising-conversion-architect"], packs: ["knowledge/fashion-commerce-fundamentals.md", "knowledge/merchandising-and-conversion.md", "pages/page-blueprints.md"], prohibited: ["invent_offer_or_proof", "use_dark_pattern", "claim_unmeasured_uplift"] },
  "seo-search": { stage: "operate", surface: "discovery", owners: ["fashion-knowledge-curator", "ecommerce-growth-analytics-engineer"], packs: ["pages/page-blueprints.md", "prompts/prompt-orchestration.md"], prohibited: ["invent_search_evidence", "keyword_stuffing", "publish_or_mutate_without_authority"] },
  "social-media": { stage: "operate", surface: "social-editorial", owners: ["brand-experience-architect", "fashion-brand-systems-researcher"], packs: ["brand/skyyrose-artifact-system.json", "knowledge/do-dont.md"], prohibited: ["post_without_authority", "use_unverified_media_or_rights", "invent_social_proof"] }
};

const skillFiles = Object.keys(domains).flatMap(domain =>
  readdirSync(join(sourceRoot, domain), { withFileTypes: true })
    .filter(entry => entry.isDirectory())
    .map(entry => ({ domain, file: join(sourceRoot, domain, entry.name, "SKILL.md") }))
).sort((a, b) => a.file.localeCompare(b.file));

function frontmatter(text, key) {
  const match = text.match(new RegExp(`^${key}:\\s*(.+)$`, "m"));
  if (!match) return null;
  return match[1].trim().replace(/^['"]|['"]$/g, "");
}

function classify(name, description, base) {
  const haystack = `${name} ${description}`.toLowerCase();
  const result = structuredClone(base);
  if (/product|cart|checkout|order|return|exchange|pricing|marketplace|gift/.test(haystack)) result.surface = "commerce-journey";
  if (/email|newsletter|sms|drip|deliverability|subject/.test(haystack)) result.surface = "lifecycle-messaging";
  if (/seo|search|keyword|schema|meta|snippet|link/.test(haystack)) result.surface = "search-discovery";
  if (/social|instagram|facebook|linkedin|pinterest|reddit|influencer|community|hashtag/.test(haystack)) result.surface = "social-editorial";
  if (/brand|logo|palette|typography|packaging|icon|identity/.test(haystack)) result.surface = "brand-system";
  if (/landing|page|funnel|checkout/.test(haystack)) result.stage = "design";
  if (/audit|research|analysis|persona|segmentation|journey/.test(haystack)) result.stage = "discover";
  if (/launch|campaign|calendar|automation|outreach|moderation/.test(haystack)) result.stage = "operate";
  if (/report|review|test|optimizer|optimization/.test(haystack)) result.stage = "measure";
  if (/accessib|wcag/.test(haystack)) result.owners = ["fashion-accessibility-content-engineer", ...result.owners];
  if (/product|fit|size|return|exchange/.test(haystack)) result.owners = ["fashion-product-fit-returns-specialist", "catalog-sot-integrator", ...result.owners];
  return result;
}

const routes = skillFiles.map(({ domain, file }) => {
  const text = readFileSync(file, "utf8");
  const name = frontmatter(text, "name") ?? basename(dirname(file));
  const description = frontmatter(text, "description") ?? "UNKNOWN";
  const c = classify(name, description, domains[domain]);
  return {
    id: `bsk.${domain}.${name}`,
    source: { domain, skill: name, relative_path: `${domain}/${basename(dirname(file))}/SKILL.md`, sha256: createHash("sha256").update(text).digest("hex") },
    triggers: { skill_name: name, phrases: [...new Set(name.split("-").filter(x => x.length > 2).concat(description.toLowerCase().match(/[a-z][a-z-]{3,}/g) ?? []))].slice(0, 12), description },
    owning_roles: [...new Set(c.owners)],
    brain_packs: [...new Set(["README.md", ...c.packs])],
    lifecycle_stage: c.stage,
    commerce_surface: c.surface,
    source_auth: {
      repository_canon_required: true,
      authenticated_primary_required_for_external_claims: true,
      catalog_sot_required_for_product_claims: /product|catalog|inventory|price|fit|size|return|exchange/.test(`${name} ${description}`.toLowerCase()),
      media_registry_and_eyes_on_required_for_imagery: /image|photo|visual|social|content|brand|design/.test(`${name} ${description}`.toLowerCase()),
      missing_evidence_disposition: "UNVERIFIED"
    },
    prohibited_uses: [...new Set([...c.prohibited, "fabricate_skyyrose_claim", "self_certify_visual_or_release", "trust_filename_as_identity"])],
    lazy_loading: {
      mode: "on_trigger_only",
      load: ["source_skill", "route_brain_packs", "target_repository_canon", "applicable_sot"],
      never_preload: ["all_234_source_skills", "unrelated_domain_packs", "optional_provider_runtime"],
      unload_after_handoff: true
    },
    status: "ROUTABLE_WITH_CANON_GATES",
    unknowns: ["target_audience_segment_until_supplied", "target_surface_state_until_supplied", "rights_and_approval_state_until_verified"]
  };
});

const aggregate = createHash("sha256").update(routes.map(r => `${r.id}:${r.source.sha256}`).join("\n")).digest("hex");
const registry = {
  schema_version: "1.0.0",
  id_policy: "bsk.<source-domain>.<source-frontmatter-name>",
  brand: { id: "skyyrose", canon: "references/skyyrose-design-canon.md", claim_status: "BRAND_SPECIFIC" },
  generated_from: { root: "vendor/branded-skills", expected_domains: Object.keys(domains), expected_skill_count: 234, observed_skill_count: routes.length, aggregate_sha256: aggregate },
  routing_policy: {
    authority_order: ["repository_and_founder_canon", "catalog_and_media_SOT", "authenticated_current_primary_sources", "dated_research", "Brain_guidance", "inference_or_experiment"],
    tie_break: ["exact_skill_name", "exact_domain_and_trigger", "commerce_surface", "lifecycle_stage"],
    ambiguity: "Return ranked route IDs and mark selection UNKNOWN; do not load multiple source skills speculatively.",
    visual_boundary: "Routes may brief or constrain work but do not implement or approve visuals."
  },
  routes
};
writeFileSync(join(outputDir, "route-registry.json"), `${JSON.stringify(registry, null, 2)}\n`);
console.log(JSON.stringify({ routes: routes.length, domains: Object.fromEntries(Object.keys(domains).map(d => [d, routes.filter(r => r.source.domain === d).length])), aggregate_sha256: aggregate }));
