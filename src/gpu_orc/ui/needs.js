import { service } from './api.js'
import { cardTable, h, replaceChildren } from './dom.js'
import { formatNumber } from './format.js'
import { t } from './i18n.js'

// What the side models need on a card: the suites' engines and the model servers' vision models.
export function needsView({ fail }) {
  const root = h('div')
  let needs = null
  let errors = []

  async function load() {
    try {
      const answer = await service.needs()
      needs = answer.needs
      errors = answer.errors
      render()
    } catch (error) { fail(error) }
  }

  function render() {
    if (needs === null) return
    replaceChildren(root,
      h('h2', {}, t('tabNeeds')),
      h('p', { class: 'muted' }, t('needsHint')),
      h('div', { class: 'row' }, h('button', { onclick: load }, t('reload'))),
      errors.length ? h('p', { class: 'status busy' }, t('errorsOf', { names: errors.map((e) => `${e.consumer} (${e.detail})`).join(', ') })) : null,
      cardTable([t('consumer'), t('item'), t('need'), t('claimSource'), t('isDefault')],
        needs.map((need) => [
          need.consumer,
          h('code', {}, need.item),
          need.need_mib === null ? t('nothing') : formatNumber(need.need_mib),
          t(`source_${need.source}`),
          need.default ? '✓' : '',
        ])),
    )
  }

  return { root, load, render }
}
