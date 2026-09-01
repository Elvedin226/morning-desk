"""Command-line entry point for the backtester.

    python main.py --list
    python main.py --strategy ma_cross --param fast=10 --param slow=40
    python main.py --strategy bollinger --symbol ETH/USDT --since 2022-01-01
    python main.py --compare
"""

from __future__ import annotations

import argparse

import numpy as np

import backtest as B
import strategy as S
import validate as V
from data import DEFAULT_EXCHANGE, load_ohlcv


def _coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_params(pairs: list[str]) -> dict:
    params = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--param expects key=value, got '{pair}'")
        key, _, value = pair.partition("=")
        params[key.strip()] = _coerce(value.strip())
    return params


def _format(result: B.Result) -> str:
    m = result.metrics
    return "\n".join(
        [
            f"  total return   {m['total_return'] * 100:>10.2f}%",
            f"  CAGR           {m['cagr'] * 100:>10.2f}%",
            f"  Sharpe         {m['sharpe']:>10.2f}",
            f"  max drawdown   {m['max_drawdown'] * 100:>10.2f}%",
            f"  exposure       {m['exposure'] * 100:>10.1f}%",
            f"  trades         {m['num_trades']:>10}",
            f"  win rate       {m['win_rate'] * 100:>10.1f}%",
            f"  avg trade      {m['avg_trade'] * 100:>10.2f}%",
            f"  final equity   {m['final_equity']:>10,.2f}",
        ]
    )


def _walk_forward_report(df, name, args) -> None:
    wf = V.walk_forward(df, name, args.train, args.test, args.timeframe, args.fee, args.slippage)
    ins, oos = wf.in_sample_metrics, wf.oos_metrics

    # ASCII only: the Windows console defaults to cp1252 and turns an em-dash into mojibake.
    print(f"{name}  ::  {len(wf.windows)} windows, train {args.train} / test {args.test} bars")

    if args.windows:
        print()
        print(wf.windows.to_string(index=False))

    fitted = ", ".join(f"{k}={v}" for k, v in wf.in_sample_params.items()) or "-"
    print()
    print(f"  in-sample     sharpe {ins['sharpe']:>6.2f}   return {ins['total_return'] * 100:>9.1f}%   ({fitted})")
    print(f"  out-of-sample sharpe {oos['sharpe']:>6.2f}   return {oos['total_return'] * 100:>9.1f}%   max DD {oos['max_drawdown'] * 100:.1f}%")

    if ins["sharpe"] > 0:
        drop = (1 - oos["sharpe"] / ins["sharpe"]) * 100
        print(f"  degradation   {drop:>6.0f}%  of the in-sample Sharpe did not survive")

    if args.permutations > 0:
        null = V.permutation_null(
            df, name,
            runs=args.permutations, train=args.train, test=args.test,
            timeframe=args.timeframe, fee=args.fee, slippage=args.slippage,
        )
        # Share of shuffled (edge-free) series that scored at least as well. This is the
        # number that matters: it is the chance of seeing this result from no edge at all.
        beat = float((null >= oos["sharpe"]).mean())
        print()
        print(f"  permutation baseline ({len(null)} shuffled series, no exploitable structure)")
        print(f"    null sharpe   median {np.median(null):.2f}   90th pct {np.percentile(null, 90):.2f}")
        print(f"    strategy scored better than {(1 - beat) * 100:.0f}% of them  (p = {beat:.2f})")
        verdict = "indistinguishable from noise" if beat > 0.10 else "survives the noise baseline"
        print(f"    verdict: {verdict}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest a trading strategy on OHLCV data.")
    p.add_argument("--strategy", default="ma_cross", help="strategy name (see --list)")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VAL",
                   help="strategy parameter, repeatable")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1d")
    p.add_argument("--since", default="2020-01-01")
    p.add_argument("--exchange", default=DEFAULT_EXCHANGE)
    p.add_argument("--fee", type=float, default=0.001, help="per-side fee, 0.001 = 0.1%%")
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--cash", type=float, default=10_000.0)
    p.add_argument("--refresh", action="store_true", help="ignore cached data, refetch")
    p.add_argument("--list", action="store_true", help="list strategies and their parameters")
    p.add_argument("--compare", action="store_true", help="run every strategy on the same data")
    p.add_argument("--walkforward", action="store_true",
                   help="fit parameters on a rolling training window, score only unseen data")
    p.add_argument("--train", type=int, default=365, help="training window, in bars")
    p.add_argument("--test", type=int, default=90, help="test window, in bars")
    p.add_argument("--permutations", type=int, default=50,
                   help="shuffled-series runs for the no-edge baseline (0 to skip)")
    p.add_argument("--windows", action="store_true", help="print the per-window table")
    args = p.parse_args()

    if args.list:
        print("Available strategies:\n")
        for name, cls in sorted(S.STRATEGIES.items()):
            params = ", ".join(f"{k}={v}" for k, v in cls.defaults.items()) or "(no parameters)"
            print(f"  {name:<14} {params}")
            print(f"  {'':<14} {cls.__doc__.splitlines()[0]}\n")
        return

    df = load_ohlcv(args.symbol, args.timeframe, args.since, args.exchange, args.refresh)
    header = f"{args.symbol}  {args.timeframe}  {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} bars)"
    print(f"\n{header}")
    print(f"fees {args.fee * 100:.2f}%/side + slippage {args.slippage * 100:.2f}%\n")

    names = sorted(S.STRATEGIES) if args.compare else [args.strategy]
    params = {} if args.compare else _parse_params(args.param)
    if args.compare and args.param:
        raise SystemExit("--param applies to a single strategy; drop --compare to use it.")

    if args.walkforward:
        for name in names:
            if name not in S.STRATEGIES:
                raise SystemExit(f"Unknown strategy '{name}'. Available: {sorted(S.STRATEGIES)}")
            _walk_forward_report(df, name, args)
        return

    for name in names:
        try:
            strat = S.build(name, **params)
        except ValueError as err:
            # A misspelled strategy or parameter is the most likely thing to go wrong
            # here; a stack trace tells the user nothing the message doesn't.
            raise SystemExit(str(err))
        result = B.run(df, strat, args.timeframe, args.fee, args.slippage, args.cash)
        print(result.strategy)
        print(_format(result))
        print()


if __name__ == "__main__":
    main()
