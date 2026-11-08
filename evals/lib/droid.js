'use strict';

/**
 * Thin wrapper around the `droid exec` CLI (Factory.ai Droid, v0.137.x).
 *
 * Why droid and not `claude -p`: the headless `claude -p` subprocess does not
 * inherit this machine's interactive OAuth and fails with a 401. `droid` has
 * provider API keys configured in ~/.factory/config.json, so it can drive
 * Claude, GPT, Gemini and DeepSeek models uniformly. We therefore route ALL
 * model calls (solvers, judges, adversarial reviewers) through this one
 * authenticated CLI.
 *
 * Canonical entry point is `executeDroid(opts)`. Legacy wrappers preserve their
 * historical shapes and call into the same execution/classification path.
 *
 * `droid exec ... --output-format json` prints a single JSON object whose
 * `.result` field holds the model's textual answer plus a `.usage` block.
 */

const { spawnSync, spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { failureRecord } = require('./result');

// Highest reasoning effort actually supported per model (literal "max" is not
// valid for GPT/Gemini). See plan / `droid` model listing.
const MAX_EFFORT = {
  'gpt-5.5-pro': 'xhigh',
  'gemini-3.1-pro-preview': 'high',
  'deepseek-v4-pro': 'max',
  'claude-opus-5-5': 'max',
  'claude-sonnet-5-5': 'high',
  'claude-haiku-4-5-20251001': 'high',
};

const COST_MODEL_VERSION = '2025-07-01-estimates';
const DEFAULT_TIMEOUT_MS = 600000;
const DEFAULT_EXECUTABLE = 'droid';

const FAILURE_TYPES = Object.freeze(['transport', 'timeout', 'tool_leakage', 'parse']);

function maxEffortFor(model) {
  return MAX_EFFORT[model] || 'high';
}

// --- Harness isolation -------------------------------------------------------
// The Factory session injects ~/.factory/AGENTS.md and a ~150-skill catalog (incl.
// the thinking skills) into every solver. We CANNOT remove that passive context
// without a Factory API key (the WorkOS session is keyring-bound and home-override
// breaks auth). What we CAN and MUST do: disable every tool so the solver cannot
// INVOKE another skill or READ the repo (the active confound), run from a neutral
// cwd (no project AGENTS.md/files), and instruct it to ignore the passive context.
// The residual catalog is identical across all conditions, so it cannot bias a
// skill-vs-placebo delta or a capability slope. Set EVAL_NO_ISOLATE=1 to opt out.
const ISOLATION_DISABLED_TOOLS = [
  'Read', 'LS', 'Execute', 'Grep', 'Glob', 'WebSearch', 'TodoWrite', 'FetchUrl',
  'Skill', 'Edit', 'Create', 'Task', 'ToolSearch', 'GenerateDroid', 'ProposeMission',
  'StartMissionRun', 'DismissHandoffItems', 'EndFeatureRun', 'ExitSpecMode',
];
const ISOLATION_PROMPT = 'You are answering a SELF-CONTAINED task. Ignore any "Available skills" list, AGENTS.md, "Agent Guidelines", personal/calendar/email instructions, plugins, custom droids, or external tools in your context — they are irrelevant here. Do not invoke skills or read files. Reason using ONLY the content of this message.';

// Env keys allowed into the isolated child. Provider auth prefixes stay so the
// real CLI can still authenticate; arbitrary project secrets are not forwarded.
const ISOLATION_ENV_ALLOWLIST = Object.freeze([
  'PATH', 'HOME', 'USER', 'LOGNAME', 'TMPDIR', 'TEMP', 'TMP',
  'LANG', 'LC_ALL', 'LC_CTYPE', 'TERM', 'SHELL', 'COLORTERM',
  'XDG_CONFIG_HOME', 'XDG_CACHE_HOME', 'XDG_DATA_HOME', 'XDG_RUNTIME_DIR',
  'SSH_AUTH_SOCK', 'DISPLAY',
  'FACTORY_API_KEY', 'FACTORY_CONFIG', 'FACTORY_HOME',
  'ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL',
  'OPENAI_API_KEY', 'OPENAI_BASE_URL',
  'GOOGLE_API_KEY', 'GEMINI_API_KEY',
  'DEEPSEEK_API_KEY', 'XAI_API_KEY',
  'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_REGION', 'AWS_DEFAULT_REGION',
  'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'no_proxy',
  'NODE_OPTIONS', 'NODE_PATH', 'NODE_ENV',
  'EVAL_NO_ISOLATE', 'MAX_DROID_INFLIGHT',
]);

function isolationEnabled(opts = {}) {
  if (typeof opts.isolate === 'boolean') return opts.isolate;
  return process.env.EVAL_NO_ISOLATE !== '1';
}

function effectiveCwd(opts = {}) {
  return opts.cwd || os.tmpdir();
}

function isolationMetadata(opts = {}) {
  const enabled = isolationEnabled(opts);
  return {
    enabled,
    effective_cwd: effectiveCwd(opts),
    disabled_tools: enabled
      ? (opts.restrictTools ? [] : (opts.disabledTools || ISOLATION_DISABLED_TOOLS).slice())
      : [],
    restricted_tools: enabled && opts.restrictTools ? opts.restrictTools.slice() : [],
    env_allowlist: enabled ? ISOLATION_ENV_ALLOWLIST.slice() : null,
    system_prompt_appended: enabled,
    allow_tool_use: enabled && opts.allowToolUse === true,
  };
}

function buildArgs(model, effort, promptFile, opts = {}) {
  const args = ['exec', '-m', model, '-r', effort, '-f', promptFile, '--output-format', 'json'];
  if (isolationEnabled(opts)) {
    const isolation = isolationMetadata(opts);
    if (isolation.restricted_tools.length) {
      args.push('--restrict-tools', isolation.restricted_tools.join(' '));
    } else {
      args.push('--disabled-tools', isolation.disabled_tools.join(' '));
    }
    args.push('--append-system-prompt', opts.appendSystemPrompt || ISOLATION_PROMPT);
  }
  return args;
}

function filterEnv(baseEnv, opts = {}) {
  if (!isolationEnabled(opts)) return { ...baseEnv, ...(opts.env || {}) };
  const out = {};
  for (const key of ISOLATION_ENV_ALLOWLIST) {
    if (baseEnv[key] !== undefined) out[key] = baseEnv[key];
  }
  // Keep provider-prefixed keys that may not be enumerated above.
  for (const key of Object.keys(baseEnv)) {
    if (/^(FACTORY|ANTHROPIC|OPENAI|GOOGLE|GEMINI|DEEPSEEK|XAI|AWS)_/i.test(key)) {
      out[key] = baseEnv[key];
    }
  }
  return { ...out, ...(opts.env || {}) };
}

function spawnOpts(opts = {}, extra = {}) {
  return {
    encoding: 'utf8',
    cwd: effectiveCwd(opts),
    env: filterEnv(process.env, opts),
    maxBuffer: 64 * 1024 * 1024,
    ...extra,
  };
}

let tmpCounter = 0;
function writeTempPrompt(prompt, opts = {}) {
  // Avoid Date.now()/Math.random() (unavailable in some sandboxes elsewhere;
  // harmless here but keep deterministic): use pid + counter.
  const dir = opts.tempDir || os.tmpdir();
  const name = `droid-prompt-${process.pid}-${tmpCounter++}.txt`;
  const p = path.join(dir, name);
  fs.writeFileSync(p, prompt, 'utf8');
  return p;
}

function cleanupTempPrompt(promptFile) {
  if (!promptFile) return;
  try { fs.unlinkSync(promptFile); } catch (_) { /* ignore */ }
}

/**
 * Structured execution failure. Not thrown for per-item solver failures —
 * returned on the result object so denominators stay explicit.
 */
class DroidFailure {
  constructor({ type, message, details = null, attempt = null }) {
    if (!FAILURE_TYPES.includes(type)) {
      throw new TypeError(`unsupported droid failure type: ${type}`);
    }
    this.type = type;
    this.message = String(message || type);
    this.details = details;
    this.attempt = attempt;
    this.name = 'DroidFailure';
  }

  toRecord(extra = {}) {
    return failureRecord({
      type: this.type,
      message: this.message,
      attempt: this.attempt,
      details: this.details,
      ...extra,
    });
  }

  toJSON() {
    return {
      type: this.type,
      message: this.message,
      details: this.details,
      attempt: this.attempt,
    };
  }
}

function makeFailure(type, message, details, attempt) {
  return new DroidFailure({ type, message, details, attempt });
}

function parseDroidStdout(stdout) {
  const lines = String(stdout || '').split('\n').map(l => l.trim()).filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i--) {
    try {
      const obj = JSON.parse(lines[i]);
      if (obj && typeof obj === 'object' && ('result' in obj || 'type' in obj)) return obj;
    } catch (_) { /* keep scanning */ }
  }
  try { return JSON.parse(stdout); } catch (_) { return null; }
}

/**
 * Extract a JSON value from arbitrary model text (handles ```json fences and
 * leading/trailing prose). Returns null if nothing parses.
 */
function extractJson(text) {
  if (!text) return null;
  const fence = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidates = [];
  if (fence) candidates.push(fence[1]);
  candidates.push(text);
  for (const c of candidates) {
    const trimmed = c.trim();
    try { return JSON.parse(trimmed); } catch (_) { /* try substring */ }
    const start = trimmed.search(/[{[]/);
    if (start === -1) continue;
    const open = trimmed[start];
    const close = open === '{' ? '}' : ']';
    let depth = 0, end = -1, inStr = false, esc = false;
    for (let i = start; i < trimmed.length; i++) {
      const ch = trimmed[i];
      if (inStr) {
        if (esc) esc = false;
        else if (ch === '\\') esc = true;
        else if (ch === '"') inStr = false;
      } else {
        if (ch === '"') inStr = true;
        else if (ch === open) depth++;
        else if (ch === close) { depth--; if (depth === 0) { end = i; break; } }
      }
    }
    if (end !== -1) {
      try { return JSON.parse(trimmed.slice(start, end + 1)); } catch (_) { /* fallthrough */ }
    }
  }
  return null;
}

function detectToolLeakage(_text, parsed, isolation) {
  if (!isolation || !isolation.enabled || !parsed || typeof parsed !== 'object') return false;
  if (isolation.allow_tool_use) {
    if (!isolation.restricted_tools || !isolation.restricted_tools.length) return false;
    const allowed = new Set(isolation.restricted_tools);
    const names = [];
    const collect = (event) => {
      if (!event || typeof event !== 'object') return;
      const name = event.name || event.tool_name || (event.tool && event.tool.name);
      if (name) names.push(String(name));
    };
    collect(parsed.tool_use);
    collect(parsed.function_call);
    if (Array.isArray(parsed.tool_calls)) parsed.tool_calls.forEach(collect);
    if (Array.isArray(parsed.tool_events)) parsed.tool_events.forEach(collect);
    if (Array.isArray(parsed.content)) parsed.content.forEach(collect);
    if (Array.isArray(parsed.events)) parsed.events.forEach((event) => {
      collect(event);
      collect(event && event.tool_use);
      collect(event && event.tool_call);
    });
    return names.some((name) => !allowed.has(name));
  }
  // Structured CLI/tool event fields only. Do NOT regex the model answer text —
  // ordinary answers may legitimately mention tools/functions without invoking them.
  if (parsed.tool_use || parsed.tool_calls || parsed.function_call || parsed.tool_events) return true;
  if (Array.isArray(parsed.content) && parsed.content.some(part => (
    part && (part.type === 'tool_use' || part.type === 'tool_call' || part.type === 'function_call')
  ))) return true;
  if (Array.isArray(parsed.events) && parsed.events.some(ev => (
    ev && (ev.type === 'tool_use' || ev.type === 'tool_call' || ev.tool_use || ev.tool_call)
  ))) return true;
  return false;
}

function emptyUsage() {
  return null;
}

function classifyRunnerResult(rawResult, opts = {}, attempt = 1) {
  const isolation = isolationMetadata(opts);
  const stdout = rawResult && rawResult.stdout != null ? String(rawResult.stdout) : '';
  const stderr = rawResult && rawResult.stderr != null ? String(rawResult.stderr) : '';
  const raw = stdout + (stderr ? `\n[stderr] ${stderr}` : '');
  const durationMs = rawResult && rawResult.durationMs != null
    ? rawResult.durationMs
    : (rawResult && rawResult.duration_ms != null ? rawResult.duration_ms : null);
  const partialParsed = parseDroidStdout(stdout);
  const partialUsage = partialParsed && partialParsed.usage ? partialParsed.usage : emptyUsage();
  const partialDuration = durationMs != null
    ? durationMs
    : (partialParsed && partialParsed.duration_ms != null ? partialParsed.duration_ms : null);

  if (rawResult && rawResult.timedOut) {
    return {
      ok: false,
      text: '',
      json: null,
      usage: partialUsage,
      durationMs: partialDuration,
      failure: makeFailure('timeout', rawResult.error || 'timeout', { signal: rawResult.signal || 'SIGKILL' }, attempt),
      error: 'timeout',
      raw,
      isolation,
    };
  }

  if (rawResult && rawResult.error && !rawResult.status && rawResult.status !== 0) {
    const msg = String(rawResult.error.message || rawResult.error);
    const isTimeout = /ETIMEDOUT|timeout/i.test(msg);
    return {
      ok: false,
      text: '',
      json: null,
      usage: partialUsage,
      durationMs: partialDuration,
      failure: makeFailure(isTimeout ? 'timeout' : 'transport', msg, { code: rawResult.error.code || null }, attempt),
      error: msg,
      raw,
      isolation,
    };
  }

  if (rawResult && rawResult.status != null && rawResult.status !== 0) {
    return {
      ok: false,
      text: partialParsed && typeof partialParsed.result === 'string' ? partialParsed.result : '',
      json: null,
      usage: partialUsage,
      durationMs: partialDuration,
      failure: makeFailure('transport', `exit ${rawResult.status}`, { status: rawResult.status, signal: rawResult.signal || null }, attempt),
      error: `exit ${rawResult.status}`,
      raw,
      isolation,
    };
  }

  const parsed = parseDroidStdout(stdout);
  if (!parsed) {
    return {
      ok: false,
      text: '',
      json: null,
      usage: emptyUsage(),
      durationMs,
      failure: makeFailure('parse', 'could not parse droid json', { stdout_bytes: Buffer.byteLength(stdout) }, attempt),
      error: 'could not parse droid json',
      raw,
      isolation,
    };
  }

  const text = typeof parsed.result === 'string' ? parsed.result : JSON.stringify(parsed.result);
  const json = extractJson(text);
  if (detectToolLeakage(text, parsed, isolation)) {
    return {
      ok: false,
      text,
      json,
      usage: parsed.usage || emptyUsage(),
      durationMs: durationMs != null ? durationMs : (parsed.duration_ms || null),
      failure: makeFailure('tool_leakage', 'tool invocation observed under isolation', { disabled_tools: isolation.disabled_tools }, attempt),
      error: 'tool_leakage',
      raw,
      isolation,
    };
  }

  if (parsed.is_error) {
    return {
      ok: false,
      text,
      json,
      usage: parsed.usage || emptyUsage(),
      durationMs: durationMs != null ? durationMs : (parsed.duration_ms || null),
      failure: makeFailure('transport', parsed.result || 'is_error', { is_error: true }, attempt),
      error: parsed.result || 'is_error',
      raw,
      isolation,
    };
  }

  return {
    ok: true,
    text,
    json,
    usage: parsed.usage || emptyUsage(),
    durationMs: durationMs != null ? durationMs : (parsed.duration_ms || null),
    failure: null,
    error: null,
    raw,
    isolation,
  };
}

function defaultSyncRunner({ command, args, cwd, env, timeoutMs, maxBuffer }) {
  const started = Date.now();
  const res = spawnSync(command, args, {
    encoding: 'utf8',
    cwd,
    env,
    timeout: timeoutMs,
    maxBuffer: maxBuffer || 64 * 1024 * 1024,
  });
  const durationMs = Date.now() - started;
  if (res.error && res.error.code === 'ETIMEDOUT') {
    return {
      status: res.status,
      stdout: res.stdout || '',
      stderr: res.stderr || '',
      error: res.error,
      timedOut: true,
      signal: res.signal,
      durationMs,
    };
  }
  return {
    status: res.status,
    stdout: res.stdout || '',
    stderr: res.stderr || '',
    error: res.error || null,
    timedOut: false,
    signal: res.signal,
    durationMs,
  };
}

function defaultAsyncRunner({ command, args, cwd, env, timeoutMs, maxBuffer }) {
  return new Promise((resolve) => {
    const started = Date.now();
    const child = spawn(command, args, {
      cwd,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    let settled = false;
    let timedOut = false;
    const max = maxBuffer || 64 * 1024 * 1024;

    const finish = (extra = {}) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({
        status: extra.status != null ? extra.status : (child.exitCode != null ? child.exitCode : null),
        stdout,
        stderr,
        error: extra.error || null,
        timedOut: Boolean(extra.timedOut || timedOut),
        signal: extra.signal != null ? extra.signal : child.signalCode,
        durationMs: Date.now() - started,
      });
    };

    const timer = setTimeout(() => {
      timedOut = true;
      try { child.kill('SIGKILL'); } catch (_) { /* ignore */ }
      finish({ timedOut: true, status: null, signal: 'SIGKILL', error: new Error('timeout') });
    }, timeoutMs);

    child.stdout.on('data', (chunk) => {
      stdout += chunk;
      if (stdout.length > max) {
        try { child.kill('SIGKILL'); } catch (_) { /* ignore */ }
        finish({ status: null, error: new Error('maxBuffer exceeded'), signal: 'SIGKILL' });
      }
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk;
    });
    child.on('error', (err) => finish({ error: err, status: null }));
    child.on('close', (code, signal) => {
      if (timedOut) return finish({ timedOut: true, status: code, signal: signal || 'SIGKILL', error: new Error('timeout') });
      finish({ status: code, signal });
    });
  });
}

function missingModelPromptResult(opts, attempt) {
  const isolation = isolationMetadata(opts);
  return {
    ok: false,
    text: '',
    json: null,
    usage: emptyUsage(),
    durationMs: null,
    failure: makeFailure('transport', 'model and prompt required', null, attempt),
    error: 'model and prompt required',
    raw: '',
    isolation,
  };
}

function thrownRunnerResult(err, opts, attempt) {
  const isolation = isolationMetadata(opts);
  return {
    ok: false,
    text: '',
    json: null,
    usage: emptyUsage(),
    durationMs: null,
    failure: makeFailure('transport', String(err && err.message || err), { thrown: true }, attempt),
    error: String(err && err.message || err),
    raw: '',
    isolation,
  };
}

/**
 * Single execution core for both sync and async wrappers.
 * Always writes/cleans the temp prompt and classifies failures without throwing.
 * `invokeRunner(ctx)` may return a value or a Promise.
 */
function runOnceCore(opts, attempt, invokeRunner) {
  const model = opts.model;
  const prompt = opts.prompt;
  if (!model || !prompt) return missingModelPromptResult(opts, attempt);

  const effort = opts.effort || maxEffortFor(model);
  const timeoutMs = opts.timeoutMs || DEFAULT_TIMEOUT_MS;
  const command = opts.executable || opts.command || DEFAULT_EXECUTABLE;
  const promptFile = writeTempPrompt(prompt, opts);
  const args = buildArgs(model, effort, promptFile, opts);
  const isolation = isolationMetadata(opts);
  const ctx = {
    command,
    args,
    cwd: isolation.effective_cwd,
    env: filterEnv(process.env, opts),
    timeoutMs,
    maxBuffer: opts.maxBuffer || 64 * 1024 * 1024,
    promptFile,
    model,
    effort,
    attempt,
    isolation,
  };

  const finish = (rawResult) => classifyRunnerResult(rawResult, opts, attempt);
  const fail = (err) => thrownRunnerResult(err, opts, attempt);

  try {
    const maybe = invokeRunner(ctx);
    if (maybe && typeof maybe.then === 'function') {
      return maybe.then(finish, fail).finally(() => cleanupTempPrompt(promptFile));
    }
    try {
      return finish(maybe);
    } catch (err) {
      return fail(err);
    } finally {
      cleanupTempPrompt(promptFile);
    }
  } catch (err) {
    cleanupTempPrompt(promptFile);
    return fail(err);
  }
}

async function runOnce(opts, attempt) {
  const runner = opts.runner || defaultAsyncRunner;
  return runOnceCore(opts, attempt, (ctx) => runner(ctx));
}

function runOnceSync(opts, attempt) {
  const runner = opts.syncRunner || opts.runner || defaultSyncRunner;
  return runOnceCore(opts, attempt, (ctx) => runner(ctx));
}

function mergeUsage(a, b) {
  if (!a && !b) return null;
  const left = a || {};
  const right = b || {};
  const keys = new Set([...Object.keys(left), ...Object.keys(right)]);
  const out = {};
  for (const key of keys) {
    const lv = left[key];
    const rv = right[key];
    if (typeof lv === 'number' || typeof rv === 'number') {
      out[key] = Number(lv || 0) + Number(rv || 0);
    } else if (rv !== undefined) {
      out[key] = rv;
    } else {
      out[key] = lv;
    }
  }
  return out;
}

function finalizeResult(result, attempts, totals = {}) {
  const durationMs = totals.durationMs != null
    ? totals.durationMs
    : (result.durationMs != null ? result.durationMs : null);
  const usage = totals.usage !== undefined
    ? totals.usage
    : (result.usage || null);
  return {
    ok: Boolean(result.ok),
    text: result.text || '',
    json: result.json != null ? result.json : null,
    usage,
    durationMs,
    attempts,
    attempt_records: totals.attemptRecords || undefined,
    failure: result.failure || null,
    isolation: result.isolation || isolationMetadata(),
    error: result.error || (result.failure ? result.failure.message : null),
    raw: result.raw || '',
    cost_model_version: COST_MODEL_VERSION,
  };
}

function accumulateAttempts(runAttempt, maxAttempts) {
  let last = null;
  let totalUsage = null;
  let totalDurationMs = 0;
  const attemptRecords = [];

  const record = (attempt, result) => {
    last = result;
    totalUsage = mergeUsage(totalUsage, result.usage);
    if (typeof result.durationMs === 'number') totalDurationMs += result.durationMs;
    attemptRecords.push({
      attempt,
      ok: Boolean(result.ok),
      usage: result.usage || null,
      durationMs: result.durationMs != null ? result.durationMs : null,
      failure: result.failure ? result.failure.toJSON() : null,
    });
    return {
      usage: totalUsage,
      durationMs: totalDurationMs,
      attemptRecords,
    };
  };

  const shouldStop = (result) => {
    if (result.ok) return true;
    if (result.failure && result.failure.type !== 'transport' && result.failure.type !== 'timeout') return true;
    return false;
  };

  return { record, shouldStop, getLast: () => last };
}

/**
 * Canonical async droid execution path.
 * Returns structured transport/timeout/tool_leakage/parse failures rather than
 * throwing for per-item solver failures. Temp prompt files are always cleaned up.
 * Usage and wall duration accumulate across retries so denominators stay honest.
 *
 * @param {object} opts
 * @param {string} opts.model
 * @param {string} opts.prompt
 * @param {string} [opts.effort]
 * @param {number} [opts.timeoutMs]
 * @param {number} [opts.attempts=1] total attempts including the first
 * @param {Function} [opts.runner] dependency-injected async runner for tests
 * @param {string} [opts.executable] command name/path (default `droid`)
 * @param {boolean} [opts.isolate] override EVAL_NO_ISOLATE
 * @param {string} [opts.cwd] override neutral cwd
 * @returns {Promise<{ok:boolean,text:string,json:*,usage:object|null,durationMs:number|null,attempts:number,failure:DroidFailure|null,isolation:object,error:string|null,raw:string,cost_model_version:string}>}
 */
async function executeDroid(opts = {}) {
  const maxAttempts = Math.max(1, Number(opts.attempts || opts.maxAttempts || 1));
  const acc = accumulateAttempts();
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const result = await runOnce(opts, attempt);
    const totals = acc.record(attempt, result);
    if (acc.shouldStop(result) || attempt === maxAttempts) {
      return finalizeResult(result, attempt, totals);
    }
  }
  // Unreachable, but keep a defensive return.
  return finalizeResult(acc.getLast(), maxAttempts, {
    usage: null,
    durationMs: 0,
    attemptRecords: [],
  });
}

/**
 * Sync variant of the same core used by legacy droidExec.
 * Throws only for programmer-error missing model/prompt (legacy contract).
 * Runner throws become transport failures.
 */
function executeDroidSync(opts = {}) {
  const maxAttempts = Math.max(1, Number(opts.attempts || opts.maxAttempts || 1));
  const acc = accumulateAttempts();
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const result = runOnceSync(opts, attempt);
    if (result && typeof result.then === 'function') {
      throw new TypeError('executeDroidSync received an async runner; use executeDroid');
    }
    const totals = acc.record(attempt, result);
    if (acc.shouldStop(result) || attempt === maxAttempts) {
      return finalizeResult(result, attempt, totals);
    }
  }
  return finalizeResult(acc.getLast(), maxAttempts, {
    usage: null,
    durationMs: 0,
    attemptRecords: [],
  });
}

/**
 * Run a single droid exec call (sync legacy path).
 * Routes through the same core as executeDroid.
 */
function droidExec(opts = {}) {
  if (!opts.model) throw new Error('droidExec: model required');
  if (!opts.prompt) throw new Error('droidExec: prompt required');
  const final = executeDroidSync({ ...opts, attempts: 1 });
  return {
    ok: final.ok,
    text: final.text,
    usage: final.usage,
    durationMs: final.durationMs,
    error: final.error,
    raw: final.raw,
    failure: final.failure,
    isolation: final.isolation,
    attempts: final.attempts,
    json: final.json,
    cost_model_version: final.cost_model_version,
  };
}

// --- In-flight call budget (prevents rate-limit/queue explosion from nested concurrency) ---
const MAX_DROID_INFLIGHT = parseInt(process.env.MAX_DROID_INFLIGHT || '32', 10);
let _inflight = 0;
const _waitQueue = [];
function _acquireSlot() {
  if (_inflight < MAX_DROID_INFLIGHT) { _inflight++; return Promise.resolve(); }
  return new Promise(resolve => _waitQueue.push(resolve));
}
function _releaseSlot() {
  _inflight--;
  if (_waitQueue.length > 0 && _inflight < MAX_DROID_INFLIGHT) {
    _inflight++;
    _waitQueue.shift()();
  }
}

/** Async variant using spawn (non-blocking) for real concurrency. */
async function droidExecAsync(opts = {}) {
  await _acquireSlot();
  try {
    const r = await executeDroid({ ...opts, attempts: 1 });
    return {
      ok: r.ok,
      text: r.text,
      usage: r.usage,
      durationMs: r.durationMs,
      error: r.error,
      raw: r.raw,
      failure: r.failure,
      isolation: r.isolation,
      attempts: r.attempts,
      json: r.json,
      cost_model_version: r.cost_model_version,
    };
  } finally {
    _releaseSlot();
  }
}

async function droidJsonAsync(opts = {}) {
  // Two attempts: retry once when JSON cannot be extracted (legacy behavior).
  let last = null;
  for (let attempt = 0; attempt < 2; attempt++) {
    const r = await droidExecAsync(opts);
    last = r;
    if (!r.ok) continue;
    const json = r.json != null ? r.json : extractJson(r.text);
    if (json !== null) {
      return {
        ok: true,
        json,
        usage: r.usage,
        durationMs: r.durationMs,
        raw: r.text,
        error: null,
        failure: null,
        isolation: r.isolation,
        attempts: attempt + 1,
        cost_model_version: r.cost_model_version,
      };
    }
  }
  const parseFailure = makeFailure('parse', (last && last.error) || 'no json', null, 2);
  return {
    ok: false,
    json: null,
    usage: last && last.usage,
    durationMs: last && last.durationMs,
    raw: last && (last.text || last.raw),
    error: (last && last.error) || 'no json',
    failure: last && last.failure ? last.failure : parseFailure,
    isolation: last && last.isolation,
    attempts: 2,
    cost_model_version: last && last.cost_model_version || COST_MODEL_VERSION,
  };
}

/** Run a droid call and parse its result as JSON, with one retry. */
function droidJson(opts = {}) {
  let last = null;
  for (let attempt = 0; attempt < 2; attempt++) {
    const r = droidExec(opts);
    last = r;
    if (!r.ok) continue;
    const json = r.json != null ? r.json : extractJson(r.text);
    if (json !== null) {
      return {
        ok: true,
        json,
        usage: r.usage,
        durationMs: r.durationMs,
        raw: r.text,
        error: null,
        failure: null,
        isolation: r.isolation,
        attempts: attempt + 1,
        cost_model_version: r.cost_model_version,
      };
    }
  }
  const parseFailure = makeFailure('parse', (last && last.error) || 'no json', null, 2);
  return {
    ok: false,
    json: null,
    usage: last && last.usage,
    durationMs: last && last.durationMs,
    raw: last && (last.text || last.raw),
    error: (last && last.error) || 'no json',
    failure: last && last.failure ? last.failure : parseFailure,
    isolation: last && last.isolation,
    attempts: 2,
    cost_model_version: last && last.cost_model_version || COST_MODEL_VERSION,
  };
}

/** Run droidExecAsync with bounded retry on transient solver/transport failures. */
async function droidExecWithRetry(opts = {}, attempts = 3) {
  const r = await executeDroid({ ...opts, attempts });
  // Acquire global slot once around the whole retry budget so retries still
  // count as one logical in-flight call for rate limiting.
  return {
    ok: r.ok,
    text: r.text,
    usage: r.usage,
    durationMs: r.durationMs,
    error: r.error,
    raw: r.raw,
    failure: r.failure,
    isolation: r.isolation,
    attempts: r.attempts,
    json: r.json,
    cost_model_version: r.cost_model_version,
  };
}

// Wrap withRetry in the concurrency gate for live callers.
const _droidExecWithRetryInner = droidExecWithRetry;
async function droidExecWithRetryGated(opts = {}, attempts = 3) {
  await _acquireSlot();
  try {
    return await _droidExecWithRetryInner(opts, attempts);
  } finally {
    _releaseSlot();
  }
}

const COST_PER_MTOKEN = {
  claude: { input: 3.0, output: 15.0, cache_read: 0.3, cache_creation: 3.75 },
  gpt: { input: 2.5, output: 10.0, cache_read: 0.5, cache_creation: 2.5 },
  gemini: { input: 1.25, output: 5.0, cache_read: 0.3, cache_creation: 1.25 },
  deepseek: { input: 0.27, output: 1.1, cache_read: 0.07, cache_creation: 0.27 },
  grok: { input: 5.0, output: 15.0, cache_read: 0.5, cache_creation: 5.0 },
  glm: { input: 0.5, output: 2.0, cache_read: 0.1, cache_creation: 0.5 },
  unknown: { input: 1.0, output: 5.0, cache_read: 0.2, cache_creation: 1.0 },
};

function modelFamilyForCost(model) {
  const m = String(model || '').toLowerCase();
  if (m.includes('claude') || m.includes('sonnet') || m.includes('opus') || m.includes('haiku')) return 'claude';
  if (m.includes('gpt') || m.includes('o1') || m.includes('o3') || m.includes('o4')) return 'gpt';
  if (m.includes('gemini') || m.includes('google')) return 'gemini';
  if (m.includes('deepseek')) return 'deepseek';
  if (m.includes('grok')) return 'grok';
  if (m.includes('glm')) return 'glm';
  return 'unknown';
}

/** Normalize droid usage data into a standard token/cost summary. */
function usageSummary(usage, model) {
  if (!usage) {
    return {
      input_tokens: 0,
      output_tokens: 0,
      cache_read_tokens: 0,
      cache_creation_tokens: 0,
      total_tokens: 0,
      est_cost_usd: 0,
      cost_model_version: COST_MODEL_VERSION,
    };
  }
  const fam = modelFamilyForCost(model);
  const rates = COST_PER_MTOKEN[fam] || COST_PER_MTOKEN.unknown;
  const input = usage.input_tokens || 0;
  const output = usage.output_tokens || 0;
  const cacheRead = usage.cache_read_input_tokens || 0;
  const cacheCreation = usage.cache_creation_input_tokens || 0;
  const total = input + output + cacheRead + cacheCreation;
  const cost = (input * rates.input + output * rates.output + cacheRead * rates.cache_read + cacheCreation * rates.cache_creation) / 1e6;
  return {
    input_tokens: input,
    output_tokens: output,
    cache_read_tokens: cacheRead,
    cache_creation_tokens: cacheCreation,
    total_tokens: total,
    est_cost_usd: +cost.toFixed(6),
    cost_model_version: COST_MODEL_VERSION,
  };
}

/** Merge an array of usageSummary outputs into aggregate totals. */
function aggregateUsage(summaries) {
  return summaries.reduce((acc, s) => ({
    input_tokens: acc.input_tokens + (s.input_tokens || 0),
    output_tokens: acc.output_tokens + (s.output_tokens || 0),
    cache_read_tokens: acc.cache_read_tokens + (s.cache_read_tokens || 0),
    cache_creation_tokens: acc.cache_creation_tokens + (s.cache_creation_tokens || 0),
    total_tokens: acc.total_tokens + (s.total_tokens || 0),
    est_cost_usd: +((acc.est_cost_usd || 0) + (s.est_cost_usd || 0)).toFixed(6),
    cost_model_version: COST_MODEL_VERSION,
    cost_rates_assumption: 'per-1M-token list prices; not actual billing',
  }), {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_tokens: 0,
    cache_creation_tokens: 0,
    total_tokens: 0,
    est_cost_usd: 0,
    cost_model_version: COST_MODEL_VERSION,
    cost_rates_assumption: 'per-1M-token list prices; not actual billing',
  });
}

module.exports = {
  // Canonical API
  executeDroid,
  DroidFailure,
  // Legacy exports
  droidExec,
  droidJson,
  droidExecAsync,
  droidJsonAsync,
  droidExecWithRetry: droidExecWithRetryGated,
  extractJson,
  parseDroidStdout,
  maxEffortFor,
  MAX_EFFORT,
  usageSummary,
  aggregateUsage,
  modelFamilyForCost,
  COST_PER_MTOKEN,
  COST_MODEL_VERSION,
  // Isolation surface for tests / runners
  ISOLATION_DISABLED_TOOLS,
  ISOLATION_ENV_ALLOWLIST,
  ISOLATION_PROMPT,
  isolationMetadata,
  // Test hooks
  _acquireSlot,
  _releaseSlot,
  _getInflight: () => _inflight,
  MAX_DROID_INFLIGHT,
};
