# Product card shopping journey — throwaway UI prototype

**Question:** Which action hierarchy lets the collection-owned stone frame lead with verified product proof while preserving a truthful, low-friction path to bag?

This is a disposable comparison only. It contains no product imagery, SKU facts,
prices, availability, or cart writes. The three options are selected with
`?variant=direct`, `?variant=ledger`, or `?variant=editorial`.

Run it with:

```bash
python3 -m http.server 4173 --directory docs/design/v2-remodel/product-card-shopping-journey/_prototype
```

Then open `http://127.0.0.1:4173/?variant=direct`. Once a decision is recorded
in the parent contract, delete this directory; only the decision belongs in the
production component.
