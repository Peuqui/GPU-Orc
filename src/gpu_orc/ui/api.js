// The calls to the service. From this machine no token is needed; from elsewhere every call carries the
// token, kept in this browser (localStorage) once it was entered.
const TOKEN_KEY = 'gpu-orc-token'

export const token = {
  get: () => localStorage.getItem(TOKEN_KEY) ?? '',
  set: (value) => (value ? localStorage.setItem(TOKEN_KEY, value) : localStorage.removeItem(TOKEN_KEY)),
}

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail)
    this.status = status
  }
}

async function json(path) {
  const headers = token.get() ? { Authorization: `Bearer ${token.get()}` } : {}
  const response = await fetch(path, { headers })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new ApiError(response.status, data.error ?? response.statusText)
  return data
}

export const service = {
  status: () => json('../status'),
  profiles: () => json('../profiles'),
  needs: () => json('../needs'),
}
