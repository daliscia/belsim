// Party programmes of the 2024 costing: the list, one party in detail, all parties side by side.
import { h } from '../dom.js';
import { state } from '../store.js';
import { t } from '../i18n.js';
import { listCard, detailCard, allBlock, partyList } from '../parties.js';

export function render(root) {
  if (!state.parties || !partyList().length) {
    root.append(h('div', { class: 'empty' }, h('b', null, t('parties.title')), ' ', t('parties.none')));
    return;
  }
  root.append(h('div', { class: 'stack' }, listCard(), state.ui.partiesAll ? allBlock() : null, detailCard()));
}
