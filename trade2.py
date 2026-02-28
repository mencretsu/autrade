import ccxt
import pandas as pd
import time
import logging
import random
import requests
import sys
from datetime import datetime

# ============ KONFIGURASI ============
class Config:
    # API SETTINGS - WAJIB DIGANTI!
    API_KEY = ""
    API_SECRET = ""
    
    # TELEGRAM SETTINGS - WAJIB DIGANTI!
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = "-"
    
    # TRADING SETTINGS
    SYMBOLS =  [
        "ETH/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "ADA/USDT",
        "XLM/USDT",
        "BNB/USDT",
        "DOGE/USDT",
        "LTC/USDT",
        "DOT/USDT",
        "ZEC/USDT",
        "SUI/USDT",
        "NEAR/USDT",
        "ETC/USDT",
        "BCH/USDT",
        "IOTA/USDT",
        "DASH/USDT",
        "XMR/USDT",
        "ATOM/USDT",
        "LINK/USDT",
        "XTZ/USDT"
    ] # Kurangi dulu untuk test
    
    LEVERAGE = 3  # Lebih aman untuk test
    POSITION_SIZE_USDT = 10  # Kecilkan untuk test
    MAX_POSITIONS = 20
    
    # RISK MANAGEMENT
    STOP_LOSS_PERCENT = 1.0
    TAKE_PROFIT_PERCENT = 2.0
    CHECK_INTERVAL = 3
    
    # TRADING MODE
    USE_RANDOM_SIGNALS = True  # PAKAI RANDOM DULU BUAT TEST
    TRADE_CHANCE_PERCENT = 30

# ============ TELEGRAM FIX ============
class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.test_connection()
    
    def test_connection(self):
        """Test koneksi ke Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ Telegram bot connected: {response.json()['result']['username']}")
                return True
            else:
                logger.error(f"❌ Telegram bot error: {response.text}")
                return False
        except Exception as e:
            logger.error(f"❌ Telegram connection failed: {e}")
            return False
    
    def send_message(self, message, is_error=False):
        """Kirim pesan ke Telegram"""
        try:
            # Format pesan
            if is_error:
                prefix = "❌ "
                parse_mode = None
            else:
                prefix = "📊 "
                parse_mode = "HTML"
            
            full_message = f"{prefix}{message}"
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": full_message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"📨 Telegram message sent")
                return True
            else:
                logger.error(f"❌ Telegram send failed: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.error("❌ Telegram timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")
            return False

# ============ LOGGING SETUP ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============ SIMPLE ANALYZER ============
class SimpleAnalyzer:
    def __init__(self, exchange):
        self.exchange = exchange
    
    def get_signal(self, symbol):
        """Generate random signal untuk testing"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker['last']
            
            # Random signal
            if random.randint(1, 100) <= Config.TRADE_CHANCE_PERCENT:
                signal = random.choice(['BUY', 'SELL'])
                confidence = random.randint(40, 70)
            else:
                signal = 'HOLD'
                confidence = 0
            
            return {
                'signal': signal,
                'confidence': confidence,
                'price': price
            }
        except:
            return {'signal': 'HOLD', 'confidence': 0, 'price': 0}

# ============ FAST TRADER ============
class FastTrader:
    def __init__(self, config):
        self.config = config
        
        # ✅ Inisialisasi Telegram DULU
        self.telegram = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
        
        # ✅ Test koneksi Telegram
        if not self.telegram.test_connection():
            logger.warning("⚠️ Telegram connection failed, continuing without notifications")
        
        # Setup exchange
        self.exchange = ccxt.binance({
            'apiKey': config.API_KEY,
            'secret': config.API_SECRET,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
            'timeout': 5000
        })
        
        try:
            self.exchange.load_markets()
            logger.info("✅ Exchange connected")
        except Exception as e:
            logger.error(f"❌ Exchange connection failed: {e}")
            sys.exit(1)
        
        # Get balance
        self.balance = self.get_balance()
        
        # Inisialisasi komponen
        self.analyzer = SimpleAnalyzer(self.exchange)
        self.positions = {}
        self.stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0
        }
        
        # Setup leverage
        self.setup_leverage()
        
        # ✅ Kirim notifikasi startup
        startup_msg = f"""
🚀 <b>TRADING BOT STARTED</b>
━━━━━━━━━━━━━━
• Balance: <code>${self.balance:.2f}</code>
• Mode: {'RANDOM' if config.USE_RANDOM_SIGNALS else 'ANALYSIS'}
• Symbols: {len(config.SYMBOLS)}
• Leverage: {config.LEVERAGE}x
━━━━━━━━━━━━━━
Bot is now running...
        """
        self.telegram.send_message(startup_msg)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"💰 Balance: ${self.balance:.2f}")
        logger.info(f"📊 Mode: {'RANDOM' if config.USE_RANDOM_SIGNALS else 'ANALYSIS'}")
        logger.info(f"🎯 Symbols: {config.SYMBOLS}")
        logger.info(f"{'='*50}")
    
    def get_balance(self):
        """Ambil balance dari Binance"""
        try:
            balance = self.exchange.fetch_balance({'type': 'future'})
            
            # Coba berbagai format balance
            if 'USDT' in balance.get('total', {}):
                return float(balance['total']['USDT'])
            elif 'free' in balance.get('USDT', {}):
                return float(balance['USDT']['free'])
            else:
                logger.warning("Using default balance 100.0")
                return 100.0
                
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return 100.0
    
    def setup_leverage(self):
        """Set leverage untuk symbol"""
        logger.info("Setting leverage...")
        for symbol in self.config.SYMBOLS:
            try:
                self.exchange.set_leverage(self.config.LEVERAGE, symbol)
                logger.info(f"  ✅ {symbol}: {self.config.LEVERAGE}x")
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"  ⚠️ {symbol}: {e}")
    
    def calculate_size(self, symbol, price):
        """Hitung ukuran posisi"""
        try:
            # Position value dengan leverage
            position_value = self.config.POSITION_SIZE_USDT * self.config.LEVERAGE
            
            # Amount
            amount = position_value / price
            
            # Rounding berdasarkan harga
            if price > 1000:
                amount = round(amount, 4)
            elif price > 100:
                amount = round(amount, 3)
            elif price > 10:
                amount = round(amount, 2)
            elif price > 1:
                amount = round(amount, 1)
            else:
                amount = round(amount, 0)
            
            # Minimum amount
            if amount < 0.001:
                amount = 0.001
            
            return amount
            
        except Exception as e:
            logger.error(f"Size calc error: {e}")
            return 0.01
    
    def execute_trade(self, symbol, signal):
        """Eksekusi trade instan"""
        try:
            # Cek apakah bisa trade
            if symbol in self.positions:
                logger.info(f"Already have position in {symbol}")
                return False
            
            if len(self.positions) >= self.config.MAX_POSITIONS:
                logger.info("Max positions reached")
                return False
            
            # Ambil harga
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker['last']
            
            # Hitung size
            size = self.calculate_size(symbol, price)
            if size <= 0:
                return False
            
            # ✅ LOG SEBELUM ORDER
            logger.info(f"\n🎯 ATTEMPTING TRADE:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Signal: {signal}")
            logger.info(f"   Price: ${price:.4f}")
            logger.info(f"   Size: {size}")
            
            # Eksekusi MARKET ORDER
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=signal.lower(),
                amount=size
            )
            
            # Hitung SL dan TP
            if signal == 'BUY':
                sl_price = price * (1 - self.config.STOP_LOSS_PERCENT / 100)
                tp_price = price * (1 + self.config.TAKE_PROFIT_PERCENT / 100)
            else:  # SELL
                sl_price = price * (1 + self.config.STOP_LOSS_PERCENT / 100)
                tp_price = price * (1 - self.config.TAKE_PROFIT_PERCENT / 100)
            
            # Simpan posisi
            self.positions[symbol] = {
                'side': signal,
                'entry_price': price,
                'size': size,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'entry_time': time.time(),
                'order_id': order.get('id', 'N/A')
            }
            
            self.stats['trades'] += 1
            
            # ✅ KIRIM NOTIFIKASI TELEGRAM
            telegram_msg = f"""
📈 <b>TRADE OPENED</b>
━━━━━━━━━━━━━━
• Symbol: <code>{symbol}</code>
• Side: <b>{signal}</b>
• Entry: <code>${price:.4f}</code>
• Size: <code>{size}</code>
• SL: <code>${sl_price:.4f}</code>
• TP: <code>${tp_price:.4f}</code>
━━━━━━━━━━━━━━
Total trades: {self.stats['trades']}
            """
            self.telegram.send_message(telegram_msg)
            
            logger.info(f"✅ Trade executed successfully!")
            logger.info(f"   Order ID: {order.get('id', 'N/A')}")
            
            return True
            
        except Exception as e:
            error_msg = f"Trade failed for {symbol}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.telegram.send_message(f"❌ {error_msg}", is_error=True)
            return False
    
    def check_positions(self):
        """Cek dan close posisi"""
        for symbol, pos in list(self.positions.items()):
            try:
                # Ambil harga terkini
                ticker = self.exchange.fetch_ticker(symbol)
                current = ticker['last']
                
                # Hitung P&L
                if pos['side'] == 'BUY':
                    pnl = (current - pos['entry_price']) * pos['size']
                    pnl_pct = (current - pos['entry_price']) / pos['entry_price'] * 100
                else:
                    pnl = (pos['entry_price'] - current) * pos['size']
                    pnl_pct = (pos['entry_price'] - current) / pos['entry_price'] * 100
                
                # Cek exit conditions
                exit_reason = None
                
                if pos['side'] == 'BUY':
                    if current <= pos['sl_price']:
                        exit_reason = "STOP LOSS"
                    elif current >= pos['tp_price']:
                        exit_reason = "TAKE PROFIT"
                else:
                    if current >= pos['sl_price']:
                        exit_reason = "STOP LOSS"
                    elif current <= pos['tp_price']:
                        exit_reason = "TAKE PROFIT"
                
                # Max hold time
                hold_time = time.time() - pos['entry_time']
                if hold_time > 180 and not exit_reason:  # 3 menit
                    exit_reason = "TIME LIMIT"
                
                # Close position jika ada reason
                if exit_reason:
                    self.close_position(symbol, pos, current, exit_reason, pnl)
                    
            except Exception as e:
                logger.error(f"Position check error {symbol}: {e}")
    
    def close_position(self, symbol, position, exit_price, reason, pnl):
        """Close posisi"""
        try:
            # Tentukan side untuk close
            close_side = 'sell' if position['side'] == 'BUY' else 'buy'
            
            # Market order untuk close
            self.exchange.create_market_order(
                symbol=symbol,
                side=close_side,
                amount=position['size']
            )
            
            # Update stats
            self.stats['pnl'] += pnl
            if pnl > 0:
                self.stats['wins'] += 1
                result_emoji = "✅"
                result_text = "PROFIT"
            else:
                self.stats['losses'] += 1
                result_emoji = "❌"
                result_text = "LOSS"
            
            # Hapus dari positions
            del self.positions[symbol]
            
            # ✅ KIRIM NOTIFIKASI CLOSE
            hold_time = time.time() - position['entry_time']
            telegram_msg = f"""
{result_emoji} <b>TRADE CLOSED - {result_text}</b>
━━━━━━━━━━━━━━
• Symbol: <code>{symbol}</code>
• Side: {position['side']}
• Entry: <code>${position['entry_price']:.4f}</code>
• Exit: <code>${exit_price:.4f}</code>
• P&L: <code>${pnl:.4f}</code>
• Reason: <b>{reason}</b>
• Duration: <code>{hold_time:.1f}s</code>
━━━━━━━━━━━━━━
Total P&L: <code>${self.stats['pnl']:.4f}</code>
Wins: {self.stats['wins']} | Losses: {self.stats['losses']}
            """
            self.telegram.send_message(telegram_msg)
            
            logger.info(f"📤 {symbol} closed: {reason}")
            logger.info(f"   P&L: ${pnl:.4f}")
            
        except Exception as e:
            logger.error(f"Close error {symbol}: {e}")
    
    def scan_and_trade(self):
        """Scan untuk trading opportunities"""
        if len(self.positions) >= self.config.MAX_POSITIONS:
            return
        
        # Pilih symbol yang belum ada position
        available = [s for s in self.config.SYMBOLS if s not in self.positions]
        if not available:
            return
        
        # Coba 2 symbol random
        for symbol in random.sample(available, min(2, len(available))):
            try:
                # Generate signal
                analysis = self.analyzer.get_signal(symbol)
                
                # Jika dapat signal trade
                if analysis['signal'] in ['BUY', 'SELL']:
                    logger.info(f"Signal: {symbol} {analysis['signal']} ({analysis['confidence']}%)")
                    
                    # Eksekusi trade
                    if self.execute_trade(symbol, analysis['signal']):
                        break  # Satu trade dulu
                    
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Scan error {symbol}: {e}")
    
    def show_stats(self):
        """Tampilkan statistics"""
        total = self.stats['wins'] + self.stats['losses']
        win_rate = (self.stats['wins'] / total * 100) if total > 0 else 0
        
        stats_text = f"""
📊 <b>STATUS UPDATE</b>
━━━━━━━━━━━━━━
• Active Positions: <code>{len(self.positions)}</code>
• Total Trades: <code>{self.stats['trades']}</code>
• Win Rate: <code>{win_rate:.1f}%</code>
• Total P&L: <code>${self.stats['pnl']:.4f}</code>
• Balance: <code>${self.balance:.2f}</code>
━━━━━━━━━━━━━━
        """
        
        logger.info(f"\n{'='*40}")
        logger.info(f"Positions: {len(self.positions)}/{self.config.MAX_POSITIONS}")
        logger.info(f"Trades: {self.stats['trades']}")
        logger.info(f"P&L: ${self.stats['pnl']:.4f}")
        logger.info(f"Win Rate: {win_rate:.1f}%")
        logger.info(f"{'='*40}")
        
        # Kirim ke Telegram setiap 5 trade
        if self.stats['trades'] % 5 == 0:
            self.telegram.send_message(stats_text)
    
    def run(self):
        """Main loop"""
        iteration = 0
        
        while True:
            iteration += 1
            
            try:
                # 1. Cek posisi aktif
                self.check_positions()
                
                # 2. Cari trade baru (setiap 2 iterasi)
                if iteration % 2 == 0:
                    self.scan_and_trade()
                
                # 3. Update stats (setiap 5 iterasi)
                if iteration % 5 == 0:
                    self.show_stats()
                    # Update balance
                    self.balance = self.get_balance()
                
                # 4. Tunggu
                time.sleep(self.config.CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("\n🛑 Bot stopped by user")
                self.telegram.send_message("🛑 Bot manually stopped")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(5)

# ============ MAIN ============
def main():
    print("\n" + "="*60)
    print("🤖 TELEGRAM NOTIFICATION BOT - FIXED")
    print("="*60)
    
    # Validasi config
    if "YOUR_API" in Config.API_KEY:
        print("\n❌ ERROR: Replace API credentials in Config class!")
        print("1. Get Binance API: https://www.binance.com/en/my/settings/api-management")
        print("2. Get Telegram Bot: Talk to @BotFather on Telegram")
        print("3. Get Chat ID: Send /start to @userinfobot")
        return
    
    print(f"\n⚙️  CONFIG:")
    print(f"   Symbols: {Config.SYMBOLS}")
    print(f"   Position: ${Config.POSITION_SIZE_USDT}")
    print(f"   Telegram: {'ENABLED' if Config.TELEGRAM_BOT_TOKEN != 'xxxxxxxxxxxx' else 'NOT SET'}")
    
    print("\n📨 Testing Telegram connection...")
    
    # Test Telegram
    test_bot = TelegramNotifier(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)
    if test_bot.test_connection():
        print("✅ Telegram connected!")
        test_bot.send_message("🔔 <b>Bot test successful!</b>\nConnection established.")
    else:
        print("⚠️  Telegram not connected - check token & chat ID")
        response = input("Continue without Telegram? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            return
    
    print("\n🚀 Starting bot in 3 seconds...")
    time.sleep(3)
    
    try:
        bot = FastTrader(Config)
        bot.run()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
