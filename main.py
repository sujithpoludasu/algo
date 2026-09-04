import json
import os
import signal
import ssl
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import pandas as pd
import requests
import websocket

# ==========================================
# 1. DATA MODELS & INTERFACES
# ==========================================

@dataclass
class Candle:
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class IndicatorSnapshot:
    candle_time: int
    prev_ema9: float
    prev_ema20: float
    prev_vwap: float
    prev_adx: float
    prev_plus_di: float
    prev_minus_di: float
    curr_open: float
    curr_ltp: float

class PositionState(Enum):
    NO_POSITION = 1
    IN_POSITION = 2

class BaseDataFeed(ABC):
    @abstractmethod
    def start(self):
        pass

class BaseExecutionHandler(ABC):
    @abstractmethod
    def place_order(self, symbol: str, side: str, price: float):
        pass

# ==========================================
# 2. INDICATOR ENGINE
# ==========================================

class IndicatorEngine:
    def __init__(self, ema_fast=9, ema_slow=20, adx_len=14):
        self.ema_fast_len = ema_fast
        self.ema_slow_len = ema_slow
        self.adx_len = adx_len
        
        self.alpha_fast = 2 / (ema_fast + 1)
        self.alpha_slow = 2 / (ema_slow + 1)
        
        # Closed candle state
        self.last_closed_time = None
        self.prev_ema9 = None
        self.prev_ema20 = None
        self.stored_vwap = 0.0
        self.last_adx = 0.0
        self.last_plus_di = 0.0
        self.last_minus_di = 0.0
        
        # Active candle tracking
        self.curr_open = None
        self.curr_ltp = None

    def bootstrap(self, df: pd.DataFrame):
        """Initializes state from historical REST data."""
        df = df.copy()
        df['ema9'] = df['close'].ewm(span=self.ema_fast_len, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=self.ema_slow_len, adjust=False).mean()
        
        # VWAP
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['pv'] = df['hlc3'] * df['volume']
        df['session'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.date
        cum_pv = df.groupby('session')['pv'].cumsum()
        cum_vol = df.groupby('session')['volume'].cumsum()
        df['vwap'] = cum_pv / cum_vol
        
        # ADX / DI
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        prev_close = df['close'].shift(1)
        
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        
        up_move = df['high'] - prev_high
        down_move = prev_low - df['low']
        
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
        
        tr_smooth = tr.ewm(alpha=1/self.adx_len, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1/self.adx_len, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/self.adx_len, adjust=False).mean()
        
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        di_sum = plus_di + minus_di
        dx = 100 * ((plus_di - minus_di).abs() / di_sum).fillna(0)
        adx = dx.ewm(alpha=1/self.adx_len, adjust=False).mean()
        
        last = df.iloc[-1]
        self.prev_ema9 = float(last['ema9'])
        self.prev_ema20 = float(last['ema20'])
        self.stored_vwap = float(df['vwap'].iloc[-1])
        self.last_adx = float(adx.iloc[-1])
        self.last_plus_di = float(plus_di.iloc[-1])
        self.last_minus_di = float(minus_di.iloc[-1])
        self.last_closed_time = int(last['time'])

    def update(self, candle: Candle) -> IndicatorSnapshot:
        """Processes tick update and returns state snapshot."""
        if self.last_closed_time is None:
            self.last_closed_time = candle.timestamp
            self.curr_open = candle.open

        if self.prev_ema9 is None:
            self.prev_ema9 = candle.open
        if self.prev_ema20 is None:
            self.prev_ema20 = candle.open

        self.curr_ltp = candle.close

        if candle.timestamp > self.last_closed_time:
            self.prev_ema9 = (self.curr_ltp * self.alpha_fast) + (self.prev_ema9 * (1 - self.alpha_fast))
            self.prev_ema20 = (self.curr_ltp * self.alpha_slow) + (self.prev_ema20 * (1 - self.alpha_slow))
            
            self.last_closed_time = candle.timestamp
            self.curr_open = candle.open

        return IndicatorSnapshot(
            candle_time=self.last_closed_time,
            prev_ema9=self.prev_ema9,
            prev_ema20=self.prev_ema20,
            prev_vwap=self.stored_vwap,
            prev_adx=self.last_adx,
            prev_plus_di=self.last_plus_di,
            prev_minus_di=self.last_minus_di,
            curr_open=self.curr_open,
            curr_ltp=self.curr_ltp
        )

# ==========================================
# 3. STRATEGY ENGINE
# ==========================================

class StrategyEngine:
    def __init__(
        self,
        execution_handler: BaseExecutionHandler,
        symbol: str,
        use_adx_filter: bool = False,
        use_vwap_filter: bool = False,
        use_body_filter: bool = False,
        adx_threshold: float = 20.0,
        min_body_size: float = 5.0,
        tsl_activation: float = 7.0,  
        tsl_lock: float = 6.0,        
        tsl_step: float = 1.0,        
        tsl_trail: float = 1.0        
    ):
        self.state = PositionState.NO_POSITION
        self.executor = execution_handler
        self.symbol = symbol

        self.use_adx_filter = use_adx_filter
        self.use_vwap_filter = use_vwap_filter
        self.use_body_filter = use_body_filter

        self.adx_threshold = float(adx_threshold)
        self.min_body_size = float(min_body_size)

        self.tsl_activation = float(tsl_activation)
        self.tsl_lock = float(tsl_lock)
        self.tsl_step = float(tsl_step)
        self.tsl_trail = float(tsl_trail)

        self.entry_price = 0.0
        self.position_side = None
        self.current_pnl = 0.0
        self.pnl_percent = 0.0

        self.tsl_active = False
        self.stop_loss = 0.0
        self.highest_pnl = 0.0

        self.entry_time = None
        self.trade_history = []

    def update_pnl(self, current_ltp: float):
        if self.state != PositionState.IN_POSITION or self.entry_price == 0.0:
            self.current_pnl = 0.0
            self.pnl_percent = 0.0
            return

        ltp = float(current_ltp)
        if self.position_side == "BUY":
            self.current_pnl = ltp - self.entry_price
        elif self.position_side == "SELL":
            self.current_pnl = self.entry_price - ltp

        self.pnl_percent = (self.current_pnl / self.entry_price) * 100

    def export_trade_summary(self, filename=None):
        if not self.trade_history:
            print("No trades executed in this session.")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trades_{self.symbol}_{timestamp}.csv"

        df = pd.DataFrame(self.trade_history)
        df.to_csv(filename, index=False)
        print(f"[LOG] Session trades saved successfully to: {filename}")
        
        total_trades = len(df)
        winning_trades = len(df[df['pnl_points'] > 0])
        losing_trades = len(df[df['pnl_points'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0
        total_pnl = df['pnl_points'].sum()

        print("\n================ SESSION TRADE SUMMARY ================")
        print(f"Total Trades : {total_trades}")
        print(f"Wins / Losses: {winning_trades} / {losing_trades}")
        print(f"Win Rate     : {win_rate:.2f}%")
        print(f"Total PnL    : {total_pnl:.2f} pts")
        print(f"Saved log to : {filename}")
        print("=======================================================\n")

    def evaluate_exits(self, snapshot) -> bool:
        ltp = float(snapshot.curr_ltp)
        open_price = float(snapshot.curr_open)
        ema9 = float(snapshot.prev_ema9)
        ema20 = float(snapshot.prev_ema20)
        timestamp = getattr(snapshot, "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if self.position_side == "BUY":
            profit_points = ltp - self.entry_price

            if not self.tsl_active and profit_points >= self.tsl_activation:
                self.tsl_active = True
                self.stop_loss = self.entry_price + self.tsl_lock
                self.highest_pnl = profit_points

            elif self.tsl_active and profit_points >= (self.highest_pnl + self.tsl_step):
                steps = int((profit_points - self.highest_pnl) // self.tsl_step)
                self.stop_loss += steps * self.tsl_trail
                self.highest_pnl += steps * self.tsl_step

            if self.tsl_active and ltp <= self.stop_loss:
                self.close_position(ltp, timestamp, reason="TSL HIT")
                return True

        elif self.position_side == "SELL":
            profit_points = self.entry_price - ltp

            if not self.tsl_active and profit_points >= self.tsl_activation:
                self.tsl_active = True
                self.stop_loss = self.entry_price - self.tsl_lock
                self.highest_pnl = profit_points

            elif self.tsl_active and profit_points >= (self.highest_pnl + self.tsl_step):
                steps = int((profit_points - self.highest_pnl) // self.tsl_step)
                self.stop_loss -= steps * self.tsl_trail
                self.highest_pnl += steps * self.tsl_step

            if self.tsl_active and ltp >= self.stop_loss:
                self.close_position(ltp, timestamp, reason="TSL HIT")
                return True

        if self.position_side == "BUY" and (ema9 < ema20) and (ltp < open_price):
            self.close_position(ltp, timestamp, reason="OPPOSITE CROSSOVER")
            return False

        if self.position_side == "SELL" and (ema9 > ema20) and (ltp > open_price):
            self.close_position(ltp, timestamp, reason="OPPOSITE CROSSOVER")
            return False

        return False

    def open_position(self, side, price, timestamp):
        self.state = PositionState.IN_POSITION
        self.position_side = side
        self.entry_price = price
        self.entry_time = timestamp
        self.tsl_active = False
        self.highest_pnl = 0.0
        self.executor.place_order(symbol=self.symbol, side=side, price=price)

    def close_position(self, exit_price, exit_time, reason):
        if self.position_side == "BUY":
            pnl_points = exit_price - self.entry_price
        else:
            pnl_points = self.entry_price - exit_price

        pnl_percent = (pnl_points / self.entry_price) * 100

        trade_log = {
            "symbol": self.symbol,
            "side": self.position_side,
            "entry_time": self.entry_time,
            "entry_price": self.entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": reason,
            "pnl_points": round(pnl_points, 2),
            "pnl_percent": round(pnl_percent, 4)
        }

        self.trade_history.append(trade_log)
        close_side = "SELL" if self.position_side == "BUY" else "BUY"
        self.executor.place_order(symbol=self.symbol, side=close_side, price=exit_price)

        self.state = PositionState.NO_POSITION
        self.position_side = None
        self.entry_price = 0.0
        self.entry_time = None

    def evaluate(self, snapshot):
        ltp = float(snapshot.curr_ltp)
        open_price = float(snapshot.curr_open)
        ema9 = float(snapshot.prev_ema9)
        ema20 = float(snapshot.prev_ema20)

        vwap = float(snapshot.prev_vwap)
        adx = float(snapshot.prev_adx)
        plus_di = float(snapshot.prev_plus_di)
        minus_di = float(snapshot.prev_minus_di)

        if self.state == PositionState.IN_POSITION:
            self.update_pnl(ltp)
            tsl_exited = self.evaluate_exits(snapshot)
            if tsl_exited or self.state == PositionState.NO_POSITION:
                return

        timestamp = datetime.fromtimestamp(snapshot.candle_time).strftime("%Y-%m-%d %H:%M:%S")

        # BUY Signal Evaluation
        buy_primary = (ema9 > ema20) and (ltp > open_price)
        if buy_primary:
            buy_adx_passed = True
            if self.use_adx_filter:
                buy_adx_passed = (adx > self.adx_threshold) and (plus_di > minus_di)

            buy_vwap_passed = True
            if self.use_vwap_filter:
                buy_vwap_passed = (vwap > 0) and (ltp > vwap)

            buy_body_passed = True
            if self.use_body_filter:
                buy_body_passed = (ltp - open_price) >= self.min_body_size

            if buy_adx_passed and buy_vwap_passed and buy_body_passed and self.state == PositionState.NO_POSITION:
                self.open_position("BUY", ltp, timestamp)
                return

        # SELL Signal Evaluation
        sell_primary = (ema9 < ema20) and (ltp < open_price)
        if sell_primary:
            sell_adx_passed = True
            if self.use_adx_filter:
                sell_adx_passed = (adx > self.adx_threshold) and (minus_di > plus_di)

            sell_vwap_passed = True
            if self.use_vwap_filter:
                sell_vwap_passed = (vwap > 0) and (ltp < vwap)

            sell_body_passed = True
            if self.use_body_filter:
                sell_body_passed = (open_price - ltp) >= self.min_body_size

            if sell_adx_passed and sell_vwap_passed and sell_body_passed and self.state == PositionState.NO_POSITION:
                self.open_position("SELL", ltp, timestamp)
                return

# ==========================================
# 4. DATA FEED (DELTA EXCHANGE WEBSOCKET)
# ==========================================

class DeltaDataFeed(BaseDataFeed):
    def __init__(self, symbol: str, on_tick_callback):
        self.symbol = symbol
        self.callback = on_tick_callback
        self.base_url = "https://api.india.delta.exchange"
        self.ws_url = "wss://socket.india.delta.exchange"
        self.ws = None
        self.is_running = True

    def fetch_historical(self, limit=5000) -> pd.DataFrame:
        end_time = int(time.time())
        start_time = end_time - (limit * 60)
        url = f"{self.base_url}/v2/history/candles"
        
        params = {
            "symbol": self.symbol,
            "resolution": "1m",
            "start": str(start_time),
            "end": str(end_time)
        }
        
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        candles = res.json().get("result", [])
        
        if not candles:
            raise ValueError(f"No historical candles returned for {self.symbol}")
        
        df = pd.DataFrame(candles)
        df["time"] = df["time"].astype(int)
        df["close"] = df["close"].astype(float)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        
        df = df.sort_values("time").reset_index(drop=True)
        
        current_minute = int(time.time() // 60) * 60
        if int(df.iloc[-1]["time"]) >= current_minute:
            df = df.iloc[:-1].copy()
            
        return df

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)

            if data.get("type") != "candlestick_1m":
                return

            candle = Candle(
                symbol=self.symbol,
                timestamp=int(data["candle_start_time"] / 1_000_000),
                open=float(data["open"]),
                high=float(data["high"]),
                low=float(data["low"]),
                close=float(data["close"]),
                volume=float(data["volume"])
            )

            if self.callback:
                self.callback(candle)

        except Exception as e:
            pass

    def start(self):
        def on_open(ws):
            if not self.is_running:
                return

            payload = {
                "type": "subscribe",
                "payload": {
                    "channels": [
                        {
                            "name": "candlestick_1m",
                            "symbols": [self.symbol]
                        }
                    ]
                }
            }
            ws.send(json.dumps(payload))

        while self.is_running:
            try:
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=on_open,
                    on_message=self._on_message,
                    on_error=lambda ws, err: None,
                    on_close=lambda ws, code, msg: None
                )

                self.ws.run_forever(
                    ping_interval=15,
                    ping_timeout=10,
                    sslopt={"cert_reqs": ssl.CERT_NONE}
                )

            except Exception:
                pass

            if not self.is_running:
                break

            for _ in range(30):
                if not self.is_running:
                    break
                time.sleep(0.1)

    def stop(self):
        self.is_running = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

# ==========================================
# 5. EXECUTION ADAPTOR & CLI DISPLAY
# ==========================================

class PaperExecutionHandler(BaseExecutionHandler):
    def __init__(self):
        self.last_execution = None

    def place_order(self, symbol: str, side: str, price: float):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_execution = f"[{timestamp}] EXECUTED {side} {symbol} @ {price:.2f}"

def render_cli(snapshot: IndicatorSnapshot, strategy: StrategyEngine, last_exec: str):
    os.system("cls" if os.name == "nt" else "clear")
    candle_time = datetime.fromtimestamp(snapshot.candle_time).strftime("%H:%M:%S")

    ltp = float(snapshot.curr_ltp)
    open_p = float(snapshot.curr_open)

    print("=======================================================")
    print("         BTCUSD LIVE STRATEGY ENGINE + DUMMY P&L       ")
    print("=======================================================")
    print(f" Active Candle    : {candle_time}")
    print(f" Current LTP      : {ltp:.2f}")
    print(f" Candle Open      : {open_p:.2f}")
    print("-------------------------------------------------------")
    print(f" Position State   : {strategy.state.name}")
    
    if strategy.state == PositionState.IN_POSITION:
        pnl_color = "+" if strategy.current_pnl >= 0 else ""
        print(f" Position Side    : {strategy.position_side}")
        print(f" Entry Price      : {strategy.entry_price:.2f}")
        print(f" Live UnPnL ($)   : {pnl_color}{strategy.current_pnl:.2f} USD")
        print(f" Live Return (%)  : {pnl_color}{strategy.pnl_percent:.2f}%")
    else:
        print(" Position Side    : FLAT")
        print(" Live UnPnL ($)   : $0.00")
        
    print("-------------------------------------------------------")
    print(f" Last Execution   : {last_exec or 'None'}")
    print("=======================================================")

# ==========================================
# 6. APPLICATION BOOTSTRAP
# ==========================================

def main():
    symbol = "BTCUSD"
    executor = PaperExecutionHandler()
    strategy = StrategyEngine(execution_handler=executor, symbol=symbol)
    indicators = IndicatorEngine(ema_fast=9, ema_slow=20)
    feed = DeltaDataFeed(symbol=symbol, on_tick_callback=None)

    # Signal hook to exit gracefully on Ctrl+C without traceback
    def signal_handler(sig, frame):
        feed.stop()
        strategy.export_trade_summary()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Fetching historical candles to seed indicators...")
    try:
        df_hist = feed.fetch_historical(limit=5000)
        indicators.bootstrap(df_hist)
        print("Indicator Engine Initialized successfully.")
    except Exception as e:
        print(f"Historical fetch failed: {e}")

    def handle_tick(candle: Candle):
        snapshot = indicators.update(candle)
        strategy.evaluate(snapshot)
        render_cli(snapshot, strategy, executor.last_execution)

    feed.callback = handle_tick
    print("Connecting to live WebSocket...")
    feed.start()

if __name__ == "__main__":
    main()
