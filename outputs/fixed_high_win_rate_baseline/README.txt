Fixed high win rate baseline
Engine: correct_baseline_full_exit_at_1p2R.py
Assumptions included:
- 1 tick slippage (0.25)
- tick-rounded entry/stop/target/exit prices
- minimum target distance = max(1.2R, 2.5 points)
- exported PnL stored in true MNQ dollars
- generic exports below include no commissions and no fees
