import ccxt
import pandas as pd
import time
import logging
import sys
import numpy as np
from datetime import datetime

class Config:
    """CONFIGURASI - FIX ENTRY"""
    API_KEY = ""
    API_SECRET = ""
    
    # SETTING TRADING
    SYMBOLS = ["DOGE/USDT", "SOL/USDT", "ETH/USDT","XRP/USDT", "LTC/USDT", "ADA/USDT", "BNB/USDT"]
    LEVERAGE = 10
    RISK_PER_TRADE_USDT = 50
    
    # TARGET PROFIT
    TP_PERCENT = 0.15
    SL_PERCENT = 0.10
    MAX_HOLD_SECONDS = 90
    
    # TIMING
    CHECK_INTERVAL = 3
    COOLDOWN_AFTER_TRADE = 2
    
    # TRADING PARAMS
    MIN_CONFIDENCE = 35
    MAX_POSITIONS = 1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class BinanceFuturesBot:
    """BOT FUTURES - ENTRY LANGSUNG"""
    
    def __init__(self, config):
        self.config = config
        
        self.exchange = ccxt.binance({
            'apiKey': config.API_KEY,
            'secret': config.API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'defaultMarginMode': 'isolated',
                'adjustForTimeDifference': True
            }
        })
        
        self.exchange.load_markets()
        logger.info("✅ Connected to Binance")
        
        self._set_leverage_for_all()
        self.balance = self._get_futures_balance()
        logger.info(f"💰 Balance: ${self.balance:.2f}")
        
        self.positions = {}
        self.last_trade_time = 0
        self.cooldown_until = 0
        
        self.stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'consecutive_losses': 0
        }
        
        logger.info("="*60)
        logger.info("🔥 BOT AKTIF - SIAP TRADE")
        logger.info("="*60)
    
    def _set_leverage_for_all(self):
        for symbol in self.config.SYMBOLS:
            try:
                self.exchange.set_leverage(self.config.LEVERAGE, symbol)
                logger.info(f"⚡ Leverage {self.config.LEVERAGE}x for {symbol}")
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Leverage error {symbol}: {e}")
    
    def _get_futures_balance(self):
        try:
            balance = self.exchange.fetch_balance({'type': 'future'})
            
            if 'USDT' in balance:
                total = balance['USDT'].get('total', 0)
                logger.info(f"📊 Balance detail: Total=${total}")
                return float(total)
            
            return 0.0
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return 100.0
    
    def calculate_position_size(self, symbol, price):
        try:
            market = self.exchange.market(symbol)
            position_value = self.config.RISK_PER_TRADE_USDT * self.config.LEVERAGE
            size = position_value / price
            
            # Minimum Binance $100
            BINANCE_MIN_NOTIONAL = 100.0
            notional_value = size * price
            
            if notional_value < BINANCE_MIN_NOTIONAL:
                required_size = BINANCE_MIN_NOTIONAL / price
                size = required_size
                logger.info(f"📏 Adjust size to meet $100 minimum: {size:.6f}")
            
            # Apply precision
            if 'precision' in market and 'amount' in market['precision']:
                amount_precision = market['precision']['amount']
                size = round(size, amount_precision)
            
            # Ensure minimum
            min_amount = market['limits']['amount'].get('min', 0)
            if min_amount and size < min_amount:
                size = min_amount
            
            logger.info(f"✅ Size for {symbol}: {size:.6f} (${size * price:.2f})")
            return size
            
        except Exception as e:
            logger.error(f"Size error: {e}")
            return max(0.001, (self.config.RISK_PER_TRADE_USDT * 2) / price)
    
    def get_market_analysis(self, symbol):
        """ANALISIS SIMPLE - PASTI DAPAT SINYAL"""
        try:
            # Get 1m and 5m data
            ohlcv_1m = self.exchange.fetch_ohlcv(symbol, '1m', limit=20)
            ohlcv_5m = self.exchange.fetch_ohlcv(symbol, '5m', limit=10)
            
            if len(ohlcv_1m) < 10:
                return 'HOLD', 0, 0
            
            df_1m = pd.DataFrame(ohlcv_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_5m = pd.DataFrame(ohlcv_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            closes_1m = df_1m['close'].values
            closes_5m = df_5m['close'].values
            current_price = closes_1m[-1]
            
            # SIMPLE LOGIC - PASTI DAPAT SINYAL
            signal = 'HOLD'
            confidence = 0
            
            # 1. Price change last 3 candles
            price_change_3 = ((current_price - closes_1m[-4]) / closes_1m[-4]) * 100 if len(closes_1m) >= 4 else 0
            price_change_10 = ((current_price - closes_1m[-11]) / closes_1m[-11]) * 100 if len(closes_1m) >= 11 else 0
            
            # 2. Simple MA crossover
            ma_fast = df_1m['close'].rolling(window=3).mean().values[-1]
            ma_slow = df_1m['close'].rolling(window=10).mean().values[-1]
            
            # 3. Volume check
            current_volume = df_1m['volume'].values[-1]
            avg_volume = df_1m['volume'].rolling(window=5).mean().values[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # 🎯 AGGRESSIVE SIGNAL GENERATION
            # BUY jika: MA fast > MA slow ATAU price naik > 0.05%
            if ma_fast > ma_slow or price_change_3 > 0.05:
                signal = 'BUY'
                confidence = 40 + min(25, abs(price_change_3) * 100)
                if volume_ratio > 1.2:
                    confidence += 15
            
            # SELL jika: MA fast < MA slow ATAU price turun > 0.05%
            elif ma_fast < ma_slow or price_change_3 < -0.05:
                signal = 'SELL'
                confidence = 40 + min(25, abs(price_change_3) * 100)
                if volume_ratio > 1.2:
                    confidence += 15
            
            # JIKA MASIH HOLD, PAKAI RANDOM SIGNAL (30% chance)
            if signal == 'HOLD':
                import random
                if random.random() < 0.3:  # 30% chance untuk random entry
                    signal = 'BUY' if random.random() > 0.5 else 'SELL'
                    confidence = 45
                    logger.warning(f"🎲 RANDOM SIGNAL: {symbol} {signal}")
            
            # Minimum confidence 35
            confidence = max(confidence, 35)
            
            return signal, confidence, current_price
            
        except Exception as e:
            logger.error(f"Analysis error {symbol}: {e}")
            return 'HOLD', 0, 0
    
    def execute_market_order(self, symbol, signal, confidence):
        """EXECUTE ORDER - TANPA BANYAK TALK"""
        if time.time() < self.cooldown_until:
            return False
        
        if symbol in self.positions:
            return False
        
        try:
            # Get price
            ticker = self.exchange.fetch_ticker(symbol)
            entry_price = ticker['last']
            
            # Calculate size
            size = self.calculate_position_size(symbol, entry_price)
            
            # For safety, reduce size if balance low
            if self.balance < 100:
                size = size * 0.5
                logger.warning("⚠️ Low balance - reducing position size 50%")
            
            logger.info(f"\n🎯 EXECUTING: {symbol} {signal}")
            logger.info(f"   Entry: ${entry_price:.4f}")
            logger.info(f"   Size: {size:.6f}")
            logger.info(f"   Value: ${size * entry_price:.2f}")
            logger.info(f"   Conf: {confidence}%")
            
            # EXECUTE MARKET ORDER
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=signal.lower(),
                amount=size
            )
            
            logger.info(f"✅ ORDER FILLED: {order.get('id', 'N/A')}")
            
            # Record position
            self.positions[symbol] = {
                'side': signal,
                'entry_price': entry_price,
                'size': size,
                'entry_time': time.time(),
                'confidence': confidence,
                'order_id': order.get('id')
            }
            
            self.stats['total_trades'] += 1
            self.last_trade_time = time.time()
            self.cooldown_until = time.time() + self.config.COOLDOWN_AFTER_TRADE
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Order failed: {e}")
            
            # Specific error handling
            error_str = str(e).lower()
            if "minimum" in error_str or "100" in error_str:
                logger.error("❌ MASALAH MINIMUM $100!")
                logger.error(f"   Naikin RISK_PER_TRADE_USDT ke minimal ${100/self.config.LEVERAGE:.2f}")
                logger.error(f"   Saat ini: ${self.config.RISK_PER_TRADE_USDT}")
            
            elif "insufficient balance" in error_str:
                logger.error("❌ SALDO KURANG!")
                logger.error("   Deposit lebih banyak atau turunin RISK_PER_TRADE_USDT")
            
            elif "position limit" in error_str:
                logger.error("❌ POSITION LIMIT!")
                logger.error("   Tunggu sampai ada position yang close")
            
            return False
    
    def monitor_positions(self):
        """MONITOR DAN CLOSE POSISI"""
        current_time = time.time()
        
        for symbol, position in list(self.positions.items()):
            try:
                # Get current price
                ticker = self.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Calculate P&L
                if position['side'] == 'BUY':
                    pnl = (current_price - position['entry_price']) * position['size']
                    pnl_percent = (current_price - position['entry_price']) / position['entry_price'] * 100
                else:
                    pnl = (position['entry_price'] - current_price) * position['size']
                    pnl_percent = (position['entry_price'] - current_price) / position['entry_price'] * 100
                
                hold_time = current_time - position['entry_time']
                
                # Check exit conditions
                exit_reason = None
                
                if pnl_percent >= self.config.TP_PERCENT:
                    exit_reason = f"TP (+{pnl_percent:.2f}%)"
                
                elif pnl_percent <= -self.config.SL_PERCENT:
                    exit_reason = f"SL ({pnl_percent:.2f}%)"
                
                elif hold_time > self.config.MAX_HOLD_SECONDS:
                    exit_reason = f"TIME ({hold_time:.0f}s)"
                
                # Force close jika profit/loss kecil tapi udah lama
                elif hold_time > 60 and abs(pnl_percent) < 0.05:
                    exit_reason = f"BREAKEVEN ({hold_time:.0f}s)"
                
                if exit_reason:
                    self._close_position(symbol, position, current_price, exit_reason, hold_time)
                    
            except Exception as e:
                logger.error(f"Monitor error {symbol}: {e}")
    
    def _close_position(self, symbol, position, exit_price, reason, hold_time):
        """CLOSE POSITION"""
        try:
            # Determine side
            close_side = 'sell' if position['side'] == 'BUY' else 'buy'
            
            # Calculate final P&L
            if position['side'] == 'BUY':
                pnl = (exit_price - position['entry_price']) * position['size']
                pnl_percent = (exit_price - position['entry_price']) / position['entry_price'] * 100
            else:
                pnl = (position['entry_price'] - exit_price) * position['size']
                pnl_percent = (position['entry_price'] - exit_price) / position['entry_price'] * 100
            
            # Execute close order
            close_order = self.exchange.create_market_order(
                symbol=symbol,
                side=close_side,
                amount=position['size']
            )
            
            logger.info(f"\n📤 CLOSING: {symbol} {position['side']}")
            logger.info(f"   Entry: ${position['entry_price']:.4f}")
            logger.info(f"   Exit: ${exit_price:.4f}")
            logger.info(f"   P&L: ${pnl:.4f} ({pnl_percent:+.2f}%)")
            logger.info(f"   Time: {hold_time:.0f}s")
            logger.info(f"   Reason: {reason}")
            
            # Update stats
            self.stats['total_pnl'] += pnl
            
            if pnl > 0:
                self.stats['wins'] += 1
                self.stats['consecutive_losses'] = 0
                logger.info("   Result: ✅ WIN")
            else:
                self.stats['losses'] += 1
                self.stats['consecutive_losses'] += 1
                logger.info("   Result: ❌ LOSS")
            
            # Cooldown
            self.cooldown_until = time.time() + self.config.COOLDOWN_AFTER_TRADE
            
            # Remove from positions
            del self.positions[symbol]
            
            # Update balance
            self.balance = self._get_futures_balance()
            
        except Exception as e:
            logger.error(f"Close error: {e}")
    
    def scan_and_trade(self):
        """SCAN DAN TRADE - SIMPLE"""
        if len(self.positions) >= self.config.MAX_POSITIONS:
            return False
        
        if time.time() < self.cooldown_until:
            return False
        
        logger.info(f"🔍 Scanning {len(self.config.SYMBOLS)} pairs...")
        
        best_signal = None
        best_confidence = 0
        
        for symbol in self.config.SYMBOLS:
            try:
                signal, confidence, price = self.get_market_analysis(symbol)
                
                # DEBUG OUTPUT
                status = f"{symbol}: {signal} ({confidence:.0f}%) | ${price:.4f}"
                
                if signal in ['BUY', 'SELL']:
                    if confidence >= self.config.MIN_CONFIDENCE:
                        logger.info(f"✅ {status}")
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_signal = {
                                'symbol': symbol,
                                'signal': signal,
                                'confidence': confidence,
                                'price': price
                            }
                    else:
                        logger.info(f"⚠️ {status} (low confidence)")
                else:
                    logger.info(f"⏸️ {status}")
                
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Scan error {symbol}: {e}")
                continue
        
        # Execute if found good signal
        if best_signal and best_confidence >= self.config.MIN_CONFIDENCE:
            logger.info(f"🎯 Executing: {best_signal['symbol']} {best_signal['signal']}")
            
            return self.execute_market_order(
                best_signal['symbol'],
                best_signal['signal'],
                best_signal['confidence']
            )
        
        return False
    
    def run_session(self, duration_minutes=30):
        """RUN TRADING SESSION"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 STARTING {duration_minutes} MINUTE SESSION")
        logger.info(f"{'='*60}")
        
        end_time = time.time() + (duration_minutes * 60)
        iteration = 0
        
        while time.time() < end_time:
            iteration += 1
            
            try:
                # 1. Monitor positions
                self.monitor_positions()
                
                # 2. Scan and trade (every iteration)
                if iteration % 1 == 0:  # Always scan
                    self.scan_and_trade()
                
                # 3. Status update
                if iteration % 5 == 0:
                    total_trades = self.stats['wins'] + self.stats['losses']
                    win_rate = (self.stats['wins'] / total_trades * 100) if total_trades > 0 else 0
                    
                    logger.info(f"\n📊 STATUS #{iteration}")
                    logger.info(f"   Positions: {len(self.positions)}/{self.config.MAX_POSITIONS}")
                    logger.info(f"   Trades: {total_trades} | W:{self.stats['wins']} L:{self.stats['losses']}")
                    logger.info(f"   Win Rate: {win_rate:.1f}%")
                    logger.info(f"   Total P&L: ${self.stats['total_pnl']:.2f}")
                    logger.info(f"   Balance: ${self.balance:.2f}")
                
                # 4. Stop if too many losses
                if self.stats['consecutive_losses'] >= 3:
                    logger.warning("⚠️  3 CONSECUTIVE LOSSES - STOPPING")
                    break
                
                # 5. Wait
                time.sleep(self.config.CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("\n🛑 Stopped by user")
                break
                
            except Exception as e:
                logger.error(f"Session error: {e}")
                time.sleep(5)
        
        # End session
        self._print_summary()
    
    def _print_summary(self):
        """PRINT SUMMARY"""
        logger.info(f"\n{'='*60}")
        logger.info("📈 SESSION SUMMARY")
        logger.info(f"{'='*60}")
        
        total_trades = self.stats['wins'] + self.stats['losses']
        win_rate = (self.stats['wins'] / total_trades * 100) if total_trades > 0 else 0
        
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Wins: {self.stats['wins']} | Losses: {self.stats['losses']}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"Total P&L: ${self.stats['total_pnl']:.2f}")
        logger.info(f"Final Balance: ${self.balance:.2f}")
        
        # Close remaining positions
        if self.positions:
            logger.info(f"\nClosing {len(self.positions)} remaining positions...")
            for symbol in list(self.positions.keys()):
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    hold_time = time.time() - self.positions[symbol]['entry_time']
                    self._close_position(symbol, self.positions[symbol], current_price, "SESSION_END", hold_time)
                except:
                    pass

def main():
    print("\n" + "="*60)
    print("🔥 BINANCE FUTURES BOT - ENTRY PASTI")
    print("="*60)
    
    if Config.API_KEY == "YOUR_API_KEY":
        print("\n❌ ERROR: Set API Key dan Secret di Config class!")
        print("\nLangkah:")
        print("1. Login Binance → API Management")
        print("2. Create API dengan FUTURES permission")
        print("3. Copy API Key dan Secret")
        print("4. Paste di Config class")
        return
    
    print(f"\n⚡ CONFIG:")
    print(f"   Symbols: {Config.SYMBOLS}")
    print(f"   Leverage: {Config.LEVERAGE}x")
    print(f"   Risk per Trade: ${Config.RISK_PER_TRADE_USDT}")
    print(f"   TP/SL: {Config.TP_PERCENT}%/{Config.SL_PERCENT}%")
    
    min_required = 100 / Config.LEVERAGE
    print(f"\n💰 MINIMUM: ${min_required:.2f} USDT")
    print(f"   Status: {'✅ OK' if Config.RISK_PER_TRADE_USDT >= min_required else '❌ TAMBAH SALDO'}")
    
    print("\n⚠️  PERINGATAN:")
    print("   • Trading REAL dengan uang asli")
    print("   • Start dengan amount kecil dulu")
    print("   • Monitor bot saat pertama kali run")
    
    response = input("\n🚀 Start 30-minute session? (y/n): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    try:
        bot = BinanceFuturesBot(Config)
        bot.run_session(duration_minutes=30)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
