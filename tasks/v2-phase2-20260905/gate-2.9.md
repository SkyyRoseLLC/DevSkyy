# Gate 2.9 — Runtime/editor token contract

The existing assets/css/design-tokens.css is the explicit primitive authority. No frontend collection palette or typeface was redesigned. sync-token-contract.cjs derives mapped theme.json settings during build and checks them during verification; missing primitives or drift fail closed.

Mappings: rose-gold/Signature gold/Black Rose silver/Love Hurts crimson -> existing rose/gold/silver/crimson. Love Hurts editor changes #DC143C to runtime #EB4666. Editor 2XS/XS spacing aligns to .25/.5rem; other mapped sizes already agree. Archivo, Hanken Grotesk, Anton and Cinzel identifiers/fallbacks come from runtime declarations. Motion becomes 180/420/900ms with the existing house easing. Runtime header/skip/desktop-guide layers are explicit 100/999/80; native dialog top-layer ownership remains separate. Mobile guide remains in flow.

The generator changes only mapped settings, preserving other theme configuration. theme.json is documented as a partially generated tracked release output. Build and full V2 verify PASS; the exact generator is hash-bound in the build contract. No PHP/runtime commerce logic changed.

Staging WordPress theme JSON resolver confirms synchronized settings (token-runtime.json). Browser computed primitives confirm crimson #eb4666, normal420ms, guide80 on Signature. Historic minified CSS preimages matched the preserved recovery commit before guarded writes (tokens-staging-receipts.json).
