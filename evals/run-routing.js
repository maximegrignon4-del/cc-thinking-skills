#!/usr/bin/env node
'use strict';

/**
 * Tier 2 — Invocation / routing accuracy.
 *
 * Given a realistic user prompt plus the catalog of every skill's `description`,
 * does a Claude-class model pick the correct skill — and correctly pick NONE for
 * irrelevant prompts? This tests the description fields that drive Claude Code's
 * auto-invocation, and the router's discriminability. Negatives guard against
 * "fires everywhere" noise.
 *
 * Usage: EVAL_RUN=run1 ROUTER_MODEL=claude-sonnet-4-6 node evals/run-routing.js
 */

const fs = require('fs');
const path = require('path');
const { loadAllSkills } = require('./lib/skills');
const { droidJsonAsync, maxEffortFor } = require('./lib/droid');
const { runDir, writeJson, mapPool } = require('./lib/io');

const ROUTER = process.env.ROUTER_MODEL || 'claude-sonnet-5-5';
const CONC = parseInt(process.env.CONC || '4', 10);
const DATASET = process.env.ROUTING_DATASET || path.join(__dirname, 'datasets', 'routing-cases.jsonl');

function loadCases() {
  return fs.readFileSync(DATASET, 'utf8').split('\n').map(l => l.trim()).filter(Boolean).map(l => JSON.parse(l));
}

function buildCatalog() {
  return loadAllSkills().map(s => `- ${s.name}: ${s.description}`).join('\n');
}

function buildPrompt(catalog, userPrompt) {
  return `You are the skill-router inside an AI coding agent. Below is a catalog of optional "thinking skill" guides, each with a one-line description. A user has sent a message. Decide which SINGLE skill (if any) should be auto-loaded to help answer it best.\n\n` +
    `Rules:\n- Pick the one skill whose description best matches the user's actual need.\n- If NO skill is a clear, strong fit (the request is routine and a mental-model guide would just be noise), answer "none".\n- Do not force a match. "none" is often correct.\n\n` +
    `=== SKILL CATALOG ===\n${catalog}\n\n` +
    `=== USER MESSAGE ===\n${userPrompt}\n\n` +
    `Return ONLY JSON: {"skill": "thinking-... or none", "confidence": 0.0-1.0, "rationale": "<one sentence>"}`;
}

function judge(chosen, c) {
  const exp = c.expected; // string or null
  if (exp === null || exp === 'none') return chosen === 'none';
  const acceptable = new Set([exp, ...(c.acceptable || [])]);
  return acceptable.has(chosen);
}

// Strict: a positive/ambiguous must match the SINGLE expected skill, ignoring the
// `acceptable` set. Surfaces the discrimination problem that lenient scoring hides.
function judgeStrict(chosen, c) {
  if (c.expected === null || c.expected === 'none') return chosen === 'none';
  return chosen === c.expected;
}

async function main() {
  let cases = loadCases();
  if (process.env.LIMIT) cases = cases.slice(0, parseInt(process.env.LIMIT, 10));
  const catalog = buildCatalog();
  console.log(`Tier 2 — routing: ${cases.length} cases, router=${ROUTER} (${maxEffortFor(ROUTER)}), conc=${CONC}`);

  let n = 0;
  const results = await mapPool(cases, CONC, async (c) => {
    const r = await droidJsonAsync({ model: ROUTER, prompt: buildPrompt(catalog, c.prompt) });
    n++;
    if (!r.ok) { console.log(`  [${n}/${cases.length}] ${c.id}: FAILED (${r.error})`); return { id: c.id, ok: false, error: r.error }; }
    const chosen = (r.json.skill || 'none').trim();
    const correct = judge(chosen, c);
    const correctStrict = judgeStrict(chosen, c);
    if (!correct) console.log(`  [${n}/${cases.length}] ${c.id} ✗ chose ${chosen} (expected ${c.expected || 'none'}) [${c.type}]`);
    return { id: c.id, ok: true, type: c.type, expected: c.expected, chosen, correct, correctStrict, confidence: r.json.confidence, rationale: r.json.rationale };
  });

  const ok = results.filter(r => r.ok);
  const byType = {};
  for (const r of ok) {
    byType[r.type] = byType[r.type] || { total: 0, correct: 0 };
    byType[r.type].total++; if (r.correct) byType[r.type].correct++;
  }
  const acc = ok.length ? ok.filter(r => r.correct).length / ok.length : 0;
  const accStrict = ok.length ? ok.filter(r => r.correctStrict).length / ok.length : 0;
  const out = {
    tier: 2, router: ROUTER, effort: maxEffortFor(ROUTER),
    n: results.length, n_ok: ok.length,
    accuracy_lenient: acc, accuracy_strict: accStrict,
    by_type: Object.fromEntries(Object.entries(byType).map(([k, v]) => [k, { ...v, acc: +(v.correct / v.total).toFixed(2) }])),
    misroutes: ok.filter(r => !r.correct),
    discrimination_only_errors: ok.filter(r => r.correct && !r.correctStrict).map(r => ({ id: r.id, expected: r.expected, chose: r.chosen })),
    results,
  };
  const file = path.join(runDir(), 'tier2-routing.json');
  writeJson(file, out);
  console.log(`\n  accuracy: lenient ${(acc * 100).toFixed(0)}%  |  strict/unique ${(accStrict * 100).toFixed(0)}%`);
  for (const [t, v] of Object.entries(out.by_type)) console.log(`    ${t}: ${v.correct}/${v.total} (${(v.acc * 100).toFixed(0)}%)`);
  console.log(`  -> ${file}`);
}

main();
