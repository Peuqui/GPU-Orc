/**
 * Client for GPU-Orc's HTTP interface (docs/interface.md). No dependencies (global fetch, Node 18+).
 *
 * This file is copied to the programs that call GPU-Orc by GPU-Orc/scripts/sync_clients.py. Edit it only in
 * GPU-Orc/clients/js/, never in a copy. Same semantics as clients/python/gpu_orc_client.py.
 *
 * The client knows the protocol and nothing else: it has no "off" state (a caller that runs without GPU-Orc
 * does not call it) and it decides nothing about what a problem means for the caller.
 */

const QUICK_TIMEOUT_MS = 5000; // GPU-Orc answers everything but /free at once
// Outcomes after which the cards are as the caller wanted them; any other outcome is a problem.
const FINE_OUTCOMES = new Set(['released', 'released_forced', 'kept']);

/** GPU-Orc is not reachable, answered an HTTP error, or answered in a form this client does not know. */
export class GpuOrcError extends Error {
  constructor(message) {
    super(message);
    this.name = 'GpuOrcError';
  }
}

/**
 * Clear the cards `model` is about to load on. Waits as long as GPU-Orc's own time limit.
 * @returns {Promise<{free: boolean, alreadyRunning: boolean, actions: object[], problems: string[]}>}
 *   problems: one line each, empty when the cards are as wanted
 */
export async function freeForModel(url, token, model, reason, keep = []) {
  const health = await call(url, token, 'GET', '/health', QUICK_TIMEOUT_MS);
  if (typeof health.caller_timeout_s !== 'number') throw new GpuOrcError('GPU-Orc /health names no caller_timeout_s');
  const answer = await call(url, token, 'POST', '/free', health.caller_timeout_s * 1000, { model, reason, keep });
  return freeResult(answer);
}

/** The traffic light: {loading: string[], held: {since, reason}|null, red: boolean}. */
export async function loading(url, token) {
  const answer = await call(url, token, 'GET', '/loading', QUICK_TIMEOUT_MS);
  if (!Array.isArray(answer.loading) || !('held' in answer)) throw new GpuOrcError('GPU-Orc /loading has an unknown form');
  return { loading: answer.loading, held: answer.held, red: answer.loading.length > 0 || answer.held !== null };
}

function freeResult(answer) {
  if (typeof answer.free !== 'boolean' || typeof answer.already_running !== 'boolean' || !Array.isArray(answer.actions)) {
    throw new GpuOrcError('GPU-Orc /free has an unknown form');
  }
  const actions = answer.actions.map(a => ({
    consumer: a.consumer, item: a.item, outcome: a.outcome, detail: a.detail ?? '', waitedS: a.waited_s ?? null,
  }));
  const problems = answer.already_running ? [] : actions
    .filter(a => !FINE_OUTCOMES.has(a.outcome))
    .map(a => `${a.consumer} ${a.item}: ${a.outcome}${a.detail ? ` (${a.detail})` : ''}`);
  if (!answer.already_running && !answer.free) problems.push('the cards are not free');
  return { free: answer.free, alreadyRunning: answer.already_running, actions, problems };
}

async function call(url, token, method, path, timeoutMs, body) {
  const headers = {};
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (token) headers.Authorization = `Bearer ${token}`;
  let response;
  try {
    response = await fetch(url.replace(/\/+$/, '') + path, {
      method, headers, body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (error) {
    throw new GpuOrcError(`GPU-Orc not reachable at ${url} (${path}): ${error.cause?.code ?? error.message}`);
  }
  const text = await response.text();
  let answer;
  try {
    answer = JSON.parse(text);
  } catch {
    throw new GpuOrcError(`GPU-Orc sent no JSON on ${path} (HTTP ${response.status})`);
  }
  if (!response.ok) throw new GpuOrcError(`GPU-Orc refused ${path} (HTTP ${response.status}): ${answer?.error ?? text}`);
  if (answer === null || typeof answer !== 'object' || Array.isArray(answer)) {
    throw new GpuOrcError(`GPU-Orc answered ${path} with no JSON object`);
  }
  return answer;
}
