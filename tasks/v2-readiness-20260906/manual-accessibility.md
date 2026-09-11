# Manual accessibility and device boundary

Native Google Chrome/macOS controls were operated through CUA on a dedicated local candidate tab. Chrome's own accessibility tree explicitly displayed **Zoom: 200%**, after Cmd+0 then five Cmd+Plus steps. This was browser page zoom, not CSS zoom, a smaller viewport, or pinch-scale emulation. The current window was about1200px wide; no claim of every display geometry follows.

Observed directly at200%:

- Shop: readable header, collection tabs, filters and sorting; two-column card start; no visually clipped horizontal content in inspected viewport.
- SG005 PDP: native size menu selectedM by keyboard. Confirmation appeared. Tab moved to Clear options with a visible focus ring; quantity and Add to Cart remained readable.
- Responsive menu: opening moved focus to Collections; Search opened with the search field focused.
- Search: typedSG005, received one product preview; twoTabs moved focus to that result, with a visible rose focus ring. Internal vertical scrolling kept result and shortcuts reachable. Escape closed the dialog and restored focus to the visible menu control (the Search opener had been in the closed menu).
- Bag: empty local session, Close Bag received focus; twoTabs cycled through Explore Shop and back to Close Bag. Escape restored focus to Bag.
- Ask Skyy: question field received focus. Typingshipping and Enter returned the local guide response and Contact the house link. Character, pause control, suggested questions and input remained readable with dialog scrolling. Escape restored focus to Ask Skyy.

Evidence is the actual CUA accessibility-tree and screenshot output in this task's tool history. These native UI screenshots were inspected directly; no separate filesystem screenshot capture is claimed. Zoom was reset to100% and only the dedicated test tab was closed. No checkout submission, order, account or external message was created.

The automated keyboard/reduced-motion and Chromium/Firefox/WebKit evidence is reported separately; synthetic automation is not described as a human screen-reader audit. No representative physical iPhone/Android was tested. No VoiceOver/TalkBack/NVDA listening session was performed. Physical-device GPU/memory, touch keyboard and assistive-technology certification remain **UNVERIFIED**.
