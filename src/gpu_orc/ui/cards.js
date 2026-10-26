import { service } from './api.js'
import { cardTable, h, replaceChildren } from './dom.js'
import { formatNumber } from './format.js'
import { t } from './i18n.js'

// How often the page asks again while it is shown.
const REFRESH_MS = 5000

const mib = (value) => (value === null || value === undefined ? t('nothing') : formatNumber(value))

// One holder of a card as a line: who, what, its memory or share, and whether it works or stays.
function holderLine(holder) {
  const notes = []
  if (holder.memory_mib !== null) notes.push(`${formatNumber(holder.memory_mib)} MiB`)
  if (holder.memory_fraction !== null) notes.push(t('share', { fraction: formatNumber(holder.memory_fraction) }))
  if (holder.busy) notes.push(t('busy'))
  if (holder.persistent) notes.push(t('persistent'))
  return h('div', {}, h('code', {}, `${holder.consumer}: ${holder.item}`), notes.length ? ` (${notes.join(', ')})` : '')
}

// What is on each card and what another program may still take there.
export function cardsView({ fail }) {
  const root = h('div')
  let status = null

  async function load() {
    try {
      status = await service.status()
      render()
    } catch (error) { fail(error) }
  }

  function render() {
    if (status === null) return
    const { queue, gpus, errors } = status
    const state = queue.running ? t('queueRunning', { seconds: formatNumber(Math.round(queue.running_since_s)) }) : t('queueIdle')
    replaceChildren(root,
      h('h2', {}, t('tabCards')),
      h('p', { class: 'muted' }, t('cardsHint')),
      h('div', { class: 'row' },
        h('button', { onclick: load }, t('reload')),
        h('span', { class: 'muted' }, t('queue', { state, waiting: queue.waiting }))),
      errors.length ? h('p', { class: 'status busy' }, t('errorsOf', { names: errors.map((e) => `${e.consumer} (${e.detail})`).join(', ') })) : null,
      cardTable([t('card'), t('used'), t('available'), t('holders'), t('unattributed')],
        gpus.map((gpu) => [
          `${gpu.index}: ${gpu.name}`,
          `${mib(gpu.memory_used_mib)} / ${mib(gpu.memory_total_mib)}`,
          h('strong', {}, mib(gpu.available_mib)),
          gpu.holders.length ? gpu.holders.map(holderLine) : t('nothing'),
          mib(gpu.unattributed_mib),
        ])),
    )
  }

  setInterval(() => {
    if (!root.closest('[hidden]') && !document.hidden) load()
  }, REFRESH_MS)

  return { root, load, render }
}
