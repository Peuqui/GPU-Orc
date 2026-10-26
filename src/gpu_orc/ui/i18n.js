// The texts of the page in German and English; the language follows the browser until it is switched.
const TEXTS = {
  de: {
    language: 'English', themeLight: 'Hell', themeDark: 'Dunkel', themeAuto: 'Automatisch',
    tabCards: 'Karten', tabProfiles: 'Profile', tabNeeds: 'Bedarfe',
    token: 'Token', tokenSet: 'Token gesetzt', tokenPrompt: 'Token für den Zugriff von einem anderen Rechner (leer = löschen):',
    tokenSaved: 'Token gespeichert.', tokenRemoved: 'Token gelöscht.',
    reload: 'Neu laden', filter: 'Filter',
    cardsHint: 'Was auf jeder Karte liegt, wie die Dienste es selbst melden. „Verfügbar“ ist, was ein weiteres Programm noch belegen darf: Ein vLLM-Modell darf seinen Anteil füllen, auch wenn nvidia-smi ihn gerade als frei zeigt.',
    card: 'Karte', used: 'Belegt (MiB)', available: 'Verfügbar (MiB)', holders: 'Halter', unattributed: 'Nicht zugeordnet (MiB)',
    queue: 'Warteschlange: {state}, {waiting} wartend', queueRunning: 'eine Freigabe läuft seit {seconds} s', queueIdle: 'frei',
    nothing: '–', share: 'Anteil {fraction}', busy: 'arbeitet', persistent: 'dauerhaft',
    profilesHint: 'Jeder Eintrag der Modellserver-Konfiguration. „Nimmt“ ist, was das Profil je Karte belegt: gemessen bei der Kalibrierung (beim Bildbeschreiber im Burn-in seines Modells), sonst bei vLLM sein Anteil der Karte, sonst unbekannt. Die Variante sagt, für welche Zusatzmodelle das Profil Platz lässt.',
    profile: 'Profil', backend: 'Backend', cards: 'Karten', context: 'Kontext', kvBlocks: 'KV-Blöcke', variant: 'Variante', claim: 'Nimmt (MiB)', claimSource: 'Quelle',
    source_measured: 'gemessen', source_fraction: 'Anteil', source_no_cards: 'keine Karte (CPU)', source_unknown: 'unbekannt', source_configured: 'eingestellt', source_estimate: 'Schätzung',
    variantBase: 'Basis', variantTts: 'TTS {engine}', variantVlm: 'Bild {key}', variantSpeed: 'Speed', variantDescriber: 'Beschreiber', variantDescriberGpu: 'Beschreiber Karte {gpu}',
    measuredAt: 'gemessen {date}', measuredAtAs: 'gemessen {date} als {profile} (gleicher Eintrag)', problem: 'Problem: {text}',
    needsHint: 'Was die Zusatzmodelle auf einer Karte brauchen: die Engines der Suiten, wie sie es melden, und die Bildmodelle der Modellserver, wie GPU-Orc sie gemessen hat.',
    consumer: 'Dienst', item: 'Modell', need: 'Bedarf (MiB)', isDefault: 'Standard der GPU',
    errorsOf: 'Konnte nicht melden: {names}',
  },
  en: {
    language: 'Deutsch', themeLight: 'Light', themeDark: 'Dark', themeAuto: 'Automatic',
    tabCards: 'Cards', tabProfiles: 'Profiles', tabNeeds: 'Needs',
    token: 'Token', tokenSet: 'Token set', tokenPrompt: 'Token for access from another machine (empty = remove):',
    tokenSaved: 'Token saved.', tokenRemoved: 'Token removed.',
    reload: 'Reload', filter: 'Filter',
    cardsHint: 'What is on each card, as the services report it themselves. “Available” is what another program may still take: a vLLM model may fill its share even where nvidia-smi shows it as free right now.',
    card: 'Card', used: 'Used (MiB)', available: 'Available (MiB)', holders: 'Holders', unattributed: 'Unattributed (MiB)',
    queue: 'Queue: {state}, {waiting} waiting', queueRunning: 'a release runs for {seconds} s', queueIdle: 'idle',
    nothing: '–', share: 'share {fraction}', busy: 'working', persistent: 'persistent',
    profilesHint: 'Every entry of the model server config. “Takes” is what the profile uses per card: measured by the calibration (for a vision describer in the burn-in of its model), else for vLLM its share of the card, else unknown. The variant says which side models the profile leaves room for.',
    profile: 'Profile', backend: 'Backend', cards: 'Cards', context: 'Context', kvBlocks: 'KV blocks', variant: 'Variant', claim: 'Takes (MiB)', claimSource: 'Source',
    source_measured: 'measured', source_fraction: 'share', source_no_cards: 'no card (CPU)', source_unknown: 'unknown', source_configured: 'configured', source_estimate: 'estimate',
    variantBase: 'base', variantTts: 'TTS {engine}', variantVlm: 'vision {key}', variantSpeed: 'speed', variantDescriber: 'describer', variantDescriberGpu: 'describer card {gpu}',
    measuredAt: 'measured {date}', measuredAtAs: 'measured {date} as {profile} (same entry)', problem: 'Problem: {text}',
    needsHint: 'What the side models need on a card: the engines of the suites as they report it, and the vision models of the model servers as GPU-Orc measured them.',
    consumer: 'Service', item: 'Model', need: 'Need (MiB)', isDefault: 'GPU default',
    errorsOf: 'Could not report: {names}',
  },
}

const STORAGE_KEY = 'gpu-orc-language'
let language = localStorage.getItem(STORAGE_KEY) ?? (navigator.language.startsWith('de') ? 'de' : 'en')
const listeners = []

export function currentLanguage() { return language }
export function onLanguageChange(listener) { listeners.push(listener) }
export function toggleLanguage() {
  language = language === 'de' ? 'en' : 'de'
  localStorage.setItem(STORAGE_KEY, language)
  document.documentElement.lang = language
  listeners.forEach((listener) => listener())
}
export function t(key, values = {}) {
  const text = TEXTS[language][key] ?? TEXTS.en[key] ?? key
  return text.replace(/\{(\w+)\}/g, (_, name) => String(values[name] ?? ''))
}
