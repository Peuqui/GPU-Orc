// Calls the JS client against a running GPU-Orc and prints the result as JSON (used by tests/test_client_js.py).
// Usage: node check_against_server.mjs <url> loading | free <model> [keep...]   (token: env GPU_ORC_TOKEN)
import { freeForModel, loading } from './gpu-orc-client.js';

const [url, scenario, model, ...keep] = process.argv.slice(2);
const token = process.env.GPU_ORC_TOKEN || null;
try {
  const result = scenario === 'loading' ? await loading(url, token) : await freeForModel(url, token, model, 'js client check', keep);
  console.log(JSON.stringify(result));
} catch (error) {
  console.log(JSON.stringify({ error: error.name, message: error.message }));
}
