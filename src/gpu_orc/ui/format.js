import { currentLanguage } from './i18n.js'

// Numbers as the reader of the page writes them: a comma in German, a dot in English. No thousands
// separator, so that 6350 MiB is never read as 6.350.
export function formatNumber(value) {
  return new Intl.NumberFormat(currentLanguage(), { maximumFractionDigits: 3, useGrouping: false }).format(value)
}
