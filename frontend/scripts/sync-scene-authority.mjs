#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const frontendRoot = resolve(scriptDirectory, '..');
const repositoryRoot = resolve(frontendRoot, '..');
const ledgerPath = join(repositoryRoot, 'Comfy', 'scene-contracts', 'scene-ooda-ledger.json');
const contractsRoot = join(repositoryRoot, 'Comfy', 'scene-contracts');
const outputPath = join(frontendRoot, 'data', 'scene-authority.json');
const checkOnly = process.argv.includes('--check');

function sha256(content) {
  return createHash('sha256').update(content).digest('hex');
}

function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

function collectionFromSceneId(sceneId) {
  if (sceneId.startsWith('LH-')) return 'love-hurts';
  if (sceneId.startsWith('BR-')) return 'black-rose';
  if (sceneId.startsWith('SIG-')) return 'signature';
  return 'unknown';
}

function assertProjection(projection) {
  if (projection?.schema !== 'skyyrose.scene-authority-projection/1') {
    throw new Error('Scene-authority projection has an unsupported schema.');
  }
  if (!Array.isArray(projection.scenes) || projection.scenes.length === 0) {
    throw new Error('Scene-authority projection contains no scenes.');
  }

  const serialized = JSON.stringify(projection);
  const forbiddenFragments = ['/Users/', '/private/', '/var/', '/tmp/', 'file://', 'Comfy/', 'tasks/', '.env'];
  const leaked = forbiddenFragments.find(fragment => serialized.includes(fragment));
  if (leaked) {
    throw new Error(`Scene-authority projection leaks an internal path or secret marker: ${leaked}`);
  }

  const sceneIds = projection.scenes.map(scene => scene.sceneId);
  if (new Set(sceneIds).size !== sceneIds.length) {
    throw new Error('Scene-authority projection contains duplicate scene IDs.');
  }
}

function buildProjection() {
  const ledgerSource = readFileSync(ledgerPath, 'utf8');
  const ledger = JSON.parse(ledgerSource);

  if (ledger.schema !== 'skyyrose.scene-ooda-ledger/1') {
    throw new Error('Scene OODA ledger has an unsupported schema.');
  }
  if (!Array.isArray(ledger.phases) || ledger.phases.length === 0) {
    throw new Error('Scene OODA ledger contains no phases.');
  }

  const scenes = ledger.phases.map(phase => {
    const contractPath = resolve(repositoryRoot, phase.contract);
    if (!contractPath.startsWith(`${contractsRoot}${sep}`)) {
      throw new Error(`${phase.scene_id} contract escapes the governed scene-contract directory.`);
    }
    const contractSource = readFileSync(contractPath, 'utf8');
    const actualContractHash = sha256(contractSource);

    if (actualContractHash !== phase.contract_sha256) {
      throw new Error(
        `${phase.scene_id} contract hash drift: ledger=${phase.contract_sha256} actual=${actualContractHash}`
      );
    }

    const contract = JSON.parse(contractSource);
    if (contract.scene_id !== phase.scene_id) {
      throw new Error(
        `${phase.scene_id} contract identity mismatch: contract declares ${contract.scene_id ?? 'no scene ID'}`
      );
    }
    const casting = contract.upstream_model_casting ?? {};
    const creditControl = contract.credit_control ?? {};
    const approvalReceiptRecorded = Boolean(creditControl.approval_receipt);
    const paidAttemptsRecorded = Number(creditControl.paid_generations_recorded ?? 0);
    const maximumPaidAttempts = Number(creditControl.max_paid_generations ?? 0);
    const attemptAvailable = maximumPaidAttempts > paidAttemptsRecorded;
    const blockers = Array.isArray(phase.blockers) ? phase.blockers : [];

    return {
      sceneId: phase.scene_id,
      collection: contract.collection ?? collectionFromSceneId(phase.scene_id),
      state: phase.state,
      purpose: contract.purpose ?? null,
      owner: phase.owner,
      accountable: phase.accountable,
      attempt: phase.attempt,
      gates: {
        identity: casting.identity_and_anatomy ?? 'NOT_RECORDED',
        productFidelity: casting.product_directionality ?? 'NOT_RECORDED',
        sceneInputEligible: casting.eligible_as_scene_input === true,
        paidAuthorization: approvalReceiptRecorded ? 'RECORDED' : 'NOT_AUTHORIZED',
        paidExecutionReady: approvalReceiptRecorded && attemptAvailable && blockers.length === 0,
        runtimeWiring: ledger.approval_boundaries.runtime_wiring,
        deployment: ledger.approval_boundaries.deployment,
        promotionRequirement: contract.promotion_gate ?? ledger.approval_boundaries.promotion,
      },
      identity: casting.approved_identity_id
        ? {
            id: casting.approved_identity_id,
            referenceAssetCount: Array.isArray(casting.approved_identity_assets)
              ? casting.approved_identity_assets.length
              : 0,
            fullCandidateVerdict: casting.verdict ?? 'NOT_RECORDED',
          }
        : null,
      evidenceHashes: {
        contract: phase.contract_sha256,
        identityApproval: casting.founder_identity_approval_sha256 ?? null,
        identityManifest: casting.approved_identity_manifest_sha256 ?? null,
      },
      blockers,
      nextAuthority: phase.next_authority,
    };
  });

  const projection = {
    schema: 'skyyrose.scene-authority-projection/1',
    recordedAt: ledger.recorded_at,
    source: {
      ledgerSha256: sha256(ledgerSource),
    },
    approvalBoundaries: {
      paidGeneration: ledger.approval_boundaries.paid_generation,
      runtimeWiring: ledger.approval_boundaries.runtime_wiring,
      deployment: ledger.approval_boundaries.deployment,
      promotion: ledger.approval_boundaries.promotion,
    },
    scenes,
  };

  assertProjection(projection);
  return projection;
}

function serializedProjection(projection) {
  return `${JSON.stringify(projection, null, 2)}\n`;
}

if (!existsSync(ledgerPath)) {
  if (!existsSync(outputPath)) {
    throw new Error('Neither repository scene authority nor a committed projection is available.');
  }
  assertProjection(readJson(outputPath));
  process.stdout.write('scene-authority: committed projection valid (repository source unavailable)\n');
  process.exit(0);
}

const expected = serializedProjection(buildProjection());
if (checkOnly) {
  if (!existsSync(outputPath) || readFileSync(outputPath, 'utf8') !== expected) {
    throw new Error('Scene-authority projection is stale. Run npm run sync:scene-authority.');
  }
  process.stdout.write('scene-authority: projection fresh\n');
  process.exit(0);
}

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, expected);
process.stdout.write('scene-authority: projection synchronized\n');
