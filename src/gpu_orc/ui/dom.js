// Builds elements without ever putting data into HTML text: what the service reports is only text.
export function h(tag, attributes = {}, ...children) {
  const element = document.createElement(tag)
  for (const [name, value] of Object.entries(attributes)) {
    if (value === false || value === null || value === undefined) continue
    if (name.startsWith('on')) element.addEventListener(name.slice(2), value)
    else if (name === 'value' || name === 'checked' || name === 'disabled' || name === 'selected') element[name] = value
    else element.setAttribute(name, value === true ? '' : value)
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue
    element.append(child instanceof Node ? child : document.createTextNode(String(child)))
  }
  return element
}

export function replaceChildren(parent, ...children) {
  parent.replaceChildren(...children.flat().filter((child) => child !== null && child !== undefined && child !== false))
}

// A table whose rows turn into cards on a narrow screen: every cell carries its heading in data-label
// (style.css). `headings` are the texts of the header cells ('' = none), `rows` arrays of cell content.
export function cardTable(headings, rows) {
  return h('div', { class: 'table-scroll' }, h('table', { class: 'cards' },
    h('thead', {}, h('tr', {}, headings.map((heading) => h('th', {}, heading)))),
    h('tbody', {}, rows.map((cells) => h('tr', {}, cells.map((cell, index) => h('td', { 'data-label': headings[index] || null }, cell)))))))
}
