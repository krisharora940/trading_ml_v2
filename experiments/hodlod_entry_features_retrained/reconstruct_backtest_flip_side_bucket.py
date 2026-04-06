import argparse
from pathlib import Path
import pandas as pd


def summarize(df: pd.DataFrame) -> dict:
    win = df['pnl'] > 0
    return {
        'trades': int(len(df)),
        'wr': round(float(win.mean() * 100), 1) if len(df) else 0.0,
        'pnl': round(float(df['pnl'].sum()), 2) if len(df) else 0.0,
        'avg_win': round(float(df.loc[win, 'pnl'].mean()), 2) if win.any() else 0.0,
        'avg_loss': round(float(df.loc[~win, 'pnl'].mean()), 2) if (~win).any() else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Reconstruct a fixed-ledger opposite-side variant for a contract bucket.')
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--min-contracts', type=int, required=True)
    ap.add_argument('--max-contracts', type=int, default=None)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    df = pd.read_csv(inp)
    if 'contracts' not in df.columns or 'pnl' not in df.columns or 'direction' not in df.columns:
        raise ValueError('Input CSV missing required columns: contracts, pnl, direction')

    original = df.copy()

    mask = df['contracts'] >= args.min_contracts
    if args.max_contracts is not None:
        mask &= df['contracts'] <= args.max_contracts

    flip_dir = {'long': 'short', 'short': 'long'}
    df.loc[mask, 'direction'] = df.loc[mask, 'direction'].map(flip_dir).fillna(df.loc[mask, 'direction'])
    df.loc[mask, 'pnl'] = -df.loc[mask, 'pnl']

    if 'outcome' in df.columns:
        mapped = df.loc[mask, 'outcome'].map({'win': 'loss', 'loss': 'win'})
        df.loc[mask, 'outcome'] = mapped.fillna(df.loc[mask, 'outcome'])

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print('input', inp)
    print('output', out)
    print('all_original', summarize(original))
    print('all_reconstructed', summarize(df))
    print('bucket_original', summarize(original[mask]))
    print('bucket_reconstructed', summarize(df[mask]))
    print('assert_trade_count_unchanged', len(original) == len(df))
    print('assert_bucket_count_unchanged', int(mask.sum()) == int((df['contracts'] >= args.min_contracts).sum()) if args.max_contracts is None else int(mask.sum()) == int(((df['contracts'] >= args.min_contracts) & (df['contracts'] <= args.max_contracts)).sum()))


if __name__ == '__main__':
    main()
