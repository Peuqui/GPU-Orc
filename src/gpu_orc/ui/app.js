import { ApiError, token } from './api.js'
import { cardsView } from './cards.js'
import { h } from './dom.js'
import { currentLanguage, onLanguageChange, t, toggleLanguage } from './i18n.js'
import { needsView } from './needs.js'
import { profilesView } from './profiles.js'

// The page: three views behind tabs, the token, the language and the colours (built like the pages of the suites).
const message = document.getElementById('message')
const NOTICE_MS = { ok: 4000, error: 10000 }
let hideTimer
function say(text, kind) {
  clearTimeout(hideTimer)
  message.textContent = text
  message.className = kind
  message.hidden = false
  hideTimer = setTimeout(() => { message.hidden = true }, NOTICE_MS[kind])
}
message.addEventListener('click', () => { message.hidden = true })
const show = (text) => say(text, 'ok')
const fail = (error) => say(error instanceof ApiError || error instanceof Error ? error.message : String(error), 'error')

const views = {
  cards: cardsView({ fail }),
  profiles: profilesView({ fail }),
  needs: needsView({ fail }),
}
const TAB_KEYS = { cards: 'tabCards', profiles: 'tabProfiles', needs: 'tabNeeds' }
let current = 'cards'

for (const [name, view] of Object.entries(views)) document.getElementById(`view-${name}`).append(view.root)

function select(name) {
  current = name
  history.replaceState(null, '', `#${name}`)
  message.hidden = true
  for (const key of Object.keys(views)) document.getElementById(`view-${key}`).hidden = key !== name
  drawTabs()
  views[name].load()
}

function drawTabs() {
  document.getElementById('tabs').replaceChildren(...Object.keys(views).map((name) =>
    h('button', { type: 'button', role: 'tab', 'aria-selected': name === current ? 'true' : 'false', onclick: () => select(name) }, t(TAB_KEYS[name]))))
}

const THEMES = ['auto', 'light', 'dark']
const THEME_KEYS = { auto: 'themeAuto', light: 'themeLight', dark: 'themeDark' }
function applyTheme() {
  const theme = localStorage.getItem('gpu-orc-theme') ?? 'auto'
  if (theme === 'auto') document.documentElement.removeAttribute('data-theme')
  else document.documentElement.dataset.theme = theme
  document.getElementById('theme').textContent = t(THEME_KEYS[theme])
}

function drawToken() {
  document.getElementById('token').textContent = t(token.get() ? 'tokenSet' : 'token')
}

function translateStatic() {
  document.documentElement.lang = currentLanguage()
  document.getElementById('language').textContent = t('language')
  applyTheme()
  drawToken()
  drawTabs()
  views[current].render()
}

document.getElementById('token').addEventListener('click', () => {
  const value = prompt(t('tokenPrompt'), token.get())
  if (value === null) return
  token.set(value.trim())
  drawToken()
  show(t(value.trim() ? 'tokenSaved' : 'tokenRemoved'))
  views[current].load()
})
document.getElementById('language').addEventListener('click', toggleLanguage)
document.getElementById('theme').addEventListener('click', () => {
  const now = localStorage.getItem('gpu-orc-theme') ?? 'auto'
  localStorage.setItem('gpu-orc-theme', THEMES[(THEMES.indexOf(now) + 1) % THEMES.length])
  applyTheme()
})
onLanguageChange(translateStatic)
translateStatic()
select(location.hash.slice(1) in views ? location.hash.slice(1) : 'cards')
