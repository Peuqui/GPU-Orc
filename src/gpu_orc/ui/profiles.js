import { service } from './api.js'
import { cardTable, h, replaceChildren } from './dom.js'
import { formatNumber } from './format.js'
import { t } from './i18n.js'

// The variant of a profile as tags: base, the side models it leaves room for, speed, describer.
function variantTags(variant) {
  const tags = []
  if (variant.describer) {
    tags.push(variant.describer_gpu === null ? t('variantDescriber') : t('variantDescriberGpu', { gpu: variant.describer_gpu }))
  } else {
    if (variant.tts) tags.push(t('variantTts', { engine: variant.tts }))
    if (variant.vlm) tags.push(t('variantVlm', { key: variant.vlm }))
    if (variant.speed) tags.push(t('variantSpeed'))
    if (!tags.length) tags.push(t('variantBase'))
  }
  return tags.map((tag) => h('span', { class: 'tag' }, tag))
}

// Every profile of the model servers with what it takes of its cards.
export function profilesView({ fail }) {
  const root = h('div')
  let profiles = null
  let errors = []
  let cardIndex = {}
  let filter = ''

  async function load() {
    try {
      const [answer, status] = await Promise.all([service.profiles(), service.status()])
      profiles = answer.profiles
      errors = answer.errors
      cardIndex = Object.fromEntries(status.gpus.map((gpu) => [gpu.uuid, gpu.index]))
      render()
    } catch (error) { fail(error) }
  }

  const card = (uuid) => (uuid in cardIndex ? String(cardIndex[uuid]) : uuid)

  function claimText(profile) {
    if (profile.claim_mib === null) return t('nothing')
    return profile.cards.map((uuid) => `${card(uuid)}: ${formatNumber(profile.claim_mib[uuid])}`).join(', ')
  }

  // Only a measurement has a date; it says "measured" itself.
  function sourceText(profile) {
    const date = profile.measured_at?.slice(0, 10)
    if (profile.measured_as) return t('measuredAtAs', { date, profile: profile.measured_as })
    if (date) return t('measuredAt', { date })
    return t(`source_${profile.claim_source}`)
  }

  function render() {
    if (profiles === null) return
    const shown = profiles.filter((profile) => profile.name.toLowerCase().includes(filter.toLowerCase()))
    replaceChildren(root,
      h('h2', {}, t('tabProfiles')),
      h('p', { class: 'muted' }, t('profilesHint')),
      h('div', { class: 'row' },
        h('button', { onclick: load }, t('reload')),
        h('label', { class: 'row' }, t('filter'), h('input', { type: 'text', value: filter, oninput: (event) => { filter = event.target.value; render() } }))),
      errors.length ? h('p', { class: 'status busy' }, t('errorsOf', { names: errors.map((e) => `${e.consumer} (${e.detail})`).join(', ') })) : null,
      cardTable([t('profile'), t('variant'), t('backend'), t('cards'), t('context'), t('kvBlocks'), t('claim'), t('claimSource')],
        shown.map((profile) => [
          h('div', {}, h('code', {}, profile.name), profile.persistent ? h('span', { class: 'tag' }, t('persistent')) : null,
            profile.problem ? h('div', { class: 'status busy' }, t('problem', { text: profile.problem })) : null),
          h('div', {}, variantTags(profile.variant)),
          profile.backend,
          profile.cards.map(card).join(', ') || t('nothing'),
          profile.context === null ? t('nothing') : formatNumber(profile.context),
          profile.kv_blocks === null ? t('nothing') : formatNumber(profile.kv_blocks),
          claimText(profile),
          sourceText(profile),
        ])),
    )
  }

  return { root, load, render }
}
