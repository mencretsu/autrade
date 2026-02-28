"""
MASTER TRADER BOT - Advanced Binance Futures Trading System
============================================================
Features:
- Multi-timeframe analysis
- 5-point confirmation system
- Auto risk management
- Telegram notifications
- Position tracking
- Emergency stop
- Profit/loss logging

Version: 1
"""

import ccxt
import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import requests
import traceback

# ==================== CONFIGURATION ====================

class Config:
    """Bot Configuration - EDIT THIS SECTION"""
    # BINANCE API CREDENTIALS
    API_KEY = ""
    API_SECRET = ""
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = "-"
    # TRADING SETTINGS
    MULTI_SYMBOL_MODE = True  # Enable multi-symbol scanning
    SYMBOLS = [
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
    ]
    TIMEFRAME = "15m"
    TRADING_STYLE = "day"  # scalping, day, swing
    
    # RISK MANAGEMENT
    ACCOUNT_BALANCE = 100  # USD
    RISK_PER_TRADE = 2  # Percentage
    MAX_POSITIONS = 1  # Maximum 1 position at a time
    USE_LEVERAGE = True
    LEVERAGE = 10  # 1-125x
    
    # CONFIRMATIONS REQUIRED
    MIN_CONFIRMATIONS = 4  # Out of 5
    
    # AUTO REFRESH
    CHECK_INTERVAL = 10  # Seconds between checks
    
    # TELEGRAM NOTIFICATIONS (Optional)
    TELEGRAM_ENABLED = True
    

    # SAFETY FEATURES
    EMERGENCY_STOP = False
    MAX_DAILY_LOSS = 50  # USD
    MAX_DAILY_TRADES = 20
    
    # LOGGING
    LOG_TRADES = True
    LOG_FILE = "trading_log.json"


# ==================== LOGGING SETUP ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== TRADING SETTINGS ====================

TRADING_SETTINGS = {
    'scalping': {
        'k_length': 9,
        'k_smooth': 3,
        'd_smooth': 3,
        'overbought': 85,
        'oversold': 15
    },
    'day': {
        'k_length': 14,
        'k_smooth': 3,
        'd_smooth': 3,
        'overbought': 80,
        'oversold': 20
    },
    'swing': {
        'k_length': 14,
        'k_smooth': 5,
        'd_smooth': 3,
        'overbought': 80,
        'oversold': 20
    }
}


# ==================== TELEGRAM NOTIFIER ====================

class TelegramNotifier:
    """Send notifications via Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    def send_message(self, message: str):
        """Send message to Telegram"""
        if not self.enabled:
            return
        
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    def send_trade_alert(self, action: str, symbol: str, price: float, 
                         size: float, confidence: int, stop_loss: float = None,
                         take_profit: float = None, leverage: int = 1,
                         position_value: float = None):
        """Send trade alert"""
        emoji = "🟢" if action == "BUY" else "🔴"
        direction = "LONG" if action == "BUY" else "SHORT"
        
        message = f"""
{emoji} <b>TRADE OPENED - {direction}</b>

<b>Symbol:</b> {symbol}
<b>Entry Price:</b> ${price:,.2f}
<b>Position Size:</b> {size}
<b>Position Value:</b> ${position_value:,.2f}
<b>Leverage:</b> {leverage}x

<b>Stop Loss:</b> ${stop_loss:,.2f}
<b>Take Profit:</b> ${take_profit:,.2f}
<b>Risk/Reward:</b> 1:1.5

<b>Confidence:</b> {confidence}%
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        self.send_message(message)
    
    def send_position_closed_alert(self, action: str, symbol: str, 
                                   entry_price: float, exit_price: float,
                                   size: float, pnl: float, pnl_percent: float,
                                   reason: str, balance: float):
        """Send position closed alert"""
        emoji = "💰" if pnl > 0 else "💸"
        result = "PROFIT" if pnl > 0 else "LOSS"
        
        message = f"""
{emoji} <b>POSITION CLOSED - {result}</b>

<b>Symbol:</b> {symbol}
<b>Direction:</b> {action}

<b>Entry:</b> ${entry_price:,.2f}
<b>Exit:</b> ${exit_price:,.2f}
<b>Size:</b> {size}

<b>P&L:</b> ${pnl:,.2f} ({pnl_percent:+.2f}%)
<b>Reason:</b> {reason}

<b>Current Balance:</b> ${balance:,.2f}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        self.send_message(message)
    
    def send_error_alert(self, error: str):
        """Send error alert"""
        message = f"⚠️ <b>ERROR ALERT</b>\n\n{error}"
        self.send_message(message)
    
    def send_daily_summary(self, trades: int, profit: float, win_rate: float,
                          balance: float, winners: int, losers: int):
        """Send daily summary"""
        emoji = "📈" if profit > 0 else "📉"
        roi = (profit / balance * 100) if balance > 0 else 0
        
        message = f"""
{emoji} <b>DAILY SUMMARY</b>

<b>Total Trades:</b> {trades}
<b>Winners:</b> {winners} 🟢
<b>Losers:</b> {losers} 🔴
<b>Win Rate:</b> {win_rate:.1f}%

<b>Total P&L:</b> ${profit:,.2f}
<b>ROI:</b> {roi:+.2f}%

<b>Current Balance:</b> ${balance:,.2f}
<b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}

{'🎉 Great job!' if profit > 0 else '💪 Keep grinding!'}
        """
        self.send_message(message)
    
    def send_position_update(self, positions: list, total_pnl: float, balance: float):
        """Send active positions update"""
        if not positions:
            return
        
        message = f"📊 <b>ACTIVE POSITIONS ({len(positions)})</b>\n\n"
        
        for i, pos in enumerate(positions, 1):
            emoji = "🟢" if pos['side'] == 'BUY' else "🔴"
            unrealized_pnl = pos.get('unrealized_pnl', 0)
            pnl_emoji = "💰" if unrealized_pnl > 0 else "💸"
            
            message += f"""
{emoji} <b>Position #{i}</b>
Symbol: {pos['symbol']}
Side: {pos['side']}
Entry: ${pos['entry_price']:,.2f}
Size: {pos['size']}
{pnl_emoji} Unrealized P&L: ${unrealized_pnl:,.2f}

"""
        
        message += f"""
<b>Total Unrealized P&L:</b> ${total_pnl:,.2f}
<b>Account Balance:</b> ${balance:,.2f}
        """
        
        self.send_message(message)


# ==================== MARKET ANALYZER ====================

class MarketAnalyzer:
    """Analyze market conditions and generate signals"""
    
    @staticmethod
    def calculate_stochastic(data: pd.DataFrame, k_len: int, k_smooth: int, 
                            d_smooth: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        high_roll = data['high'].rolling(k_len).max()
        low_roll = data['low'].rolling(k_len).min()
        
        k_values = 100 * (data['close'] - low_roll) / (high_roll - low_roll)
        k_smooth_values = k_values.rolling(k_smooth).mean()
        d_values = k_smooth_values.rolling(d_smooth).mean()
        
        return k_smooth_values, d_values
    
    @staticmethod
    def analyze_market_condition(data: pd.DataFrame) -> Dict:
        """Analyze overall market condition"""
        closes = data['close']
        volumes = data['volume']
        
        # Price change
        price_change = ((closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100)
        
        # Volatility (Standard Deviation)
        volatility = (closes.std() / closes.mean() * 100)
        
        # Volume analysis
        avg_volume = volumes.mean()
        current_volume = volumes.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Determine condition
        if abs(price_change) < 1 and volatility < 2:
            condition = 'Ranging'
            detail = 'Low volatility, sideways'
        elif abs(price_change) > 3:
            condition = 'Bull Trend' if price_change > 0 else 'Bear Trend'
            detail = f'Strong trend: {price_change:.2f}%'
        else:
            condition = 'Consolidation'
            detail = 'Medium volatility'
        
        # Trend strength
        trend_strength = 'Strong' if abs(price_change) > 5 else 'Moderate' if abs(price_change) > 2 else 'Weak'
        
        # Volume activity
        if volume_ratio > 2:
            volume_activity = 'Very High'
        elif volume_ratio > 1.5:
            volume_activity = 'High'
        elif volume_ratio < 0.5:
            volume_activity = 'Low'
        else:
            volume_activity = 'Normal'
        
        return {
            'condition': condition,
            'detail': detail,
            'price_change': price_change,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'volume_activity': volume_activity,
            'volume_ratio': volume_ratio,
            'avg_volume': avg_volume,
            'current_volume': current_volume
        }
    
    @staticmethod
    def analyze_price_action(data: pd.DataFrame) -> Dict:
        """Analyze price action and support/resistance"""
        closes = data['close'].values
        highs = data['high'].values
        lows = data['low'].values
        
        # Support and Resistance
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]
        
        resistance_1 = np.max(recent_highs)
        support_1 = np.min(recent_lows)
        
        sorted_highs = np.sort(recent_highs)[::-1]
        sorted_lows = np.sort(recent_lows)
        
        resistance_2 = sorted_highs[1] if len(sorted_highs) > 1 else resistance_1 * 0.995
        support_2 = sorted_lows[1] if len(sorted_lows) > 1 else support_1 * 1.005
        
        # Market structure
        last_5 = closes[-5:]
        if last_5[4] > last_5[2] and last_5[3] > last_5[1]:
            structure = 'Higher Highs & Higher Lows'
            bias = 'Bullish'
        elif last_5[4] < last_5[2] and last_5[3] < last_5[1]:
            structure = 'Lower Highs & Lower Lows'
            bias = 'Bearish'
        else:
            structure = 'Consolidation'
            bias = 'Neutral'
        
        # Check if at key level
        current_price = closes[-1]
        at_key_level = False
        key_level_type = ''
        
        levels = [resistance_1, resistance_2, support_1, support_2]
        for level in levels:
            diff = abs((current_price - level) / level * 100)
            if diff < 0.5:  # Within 0.5%
                at_key_level = True
                key_level_type = 'Resistance' if current_price >= level else 'Support'
                break
        
        return {
            'structure': structure,
            'bias': bias,
            'support_1': support_1,
            'support_2': support_2,
            'resistance_1': resistance_1,
            'resistance_2': resistance_2,
            'at_key_level': at_key_level,
            'key_level_type': key_level_type
        }
    
    @staticmethod
    def analyze_stochastic_signal(k_values: pd.Series, d_values: pd.Series, 
                                  settings: Dict) -> Dict:
        """Analyze Stochastic for trading signals"""
        if len(k_values) < 2:
            return {'signal': 'neutral', 'details': 'Insufficient data', 'strength': 'neutral'}
        
        k_current = k_values.iloc[-1]
        d_current = d_values.iloc[-1]
        k_previous = k_values.iloc[-2]
        d_previous = d_values.iloc[-2]
        
        bullish_cross = k_previous < d_previous and k_current > d_current
        bearish_cross = k_previous > d_previous and k_current < d_current
        
        signal = 'neutral'
        details = ''
        strength = 'weak'
        
        # Strong signals
        if k_current < settings['oversold'] and bullish_cross:
            signal = 'strong_buy'
            details = f"Oversold ({k_current:.1f}) + Bullish Cross"
            strength = 'strong'
        elif k_current > settings['overbought'] and bearish_cross:
            signal = 'strong_sell'
            details = f"Overbought ({k_current:.1f}) + Bearish Cross"
            strength = 'strong'
        # Moderate signals
        elif k_current < settings['oversold']:
            signal = 'buy'
            details = f"Oversold zone ({k_current:.1f})"
            strength = 'moderate'
        elif k_current > settings['overbought']:
            signal = 'sell'
            details = f"Overbought zone ({k_current:.1f})"
            strength = 'moderate'
        # Weak signals
        elif bullish_cross:
            signal = 'buy'
            details = 'Bullish cross in neutral zone'
            strength = 'weak'
        elif bearish_cross:
            signal = 'sell'
            details = 'Bearish cross in neutral zone'
            strength = 'weak'
        else:
            signal = 'neutral'
            details = 'No clear signal'
        
        return {
            'signal': signal,
            'details': details,
            'strength': strength,
            'k_value': k_current,
            'd_value': d_current,
            'is_bullish': k_current > d_current
        }
    
    @staticmethod
    def evaluate_confirmations(market: Dict, price_action: Dict, 
                              stoch: Dict, volume_ratio: float) -> Tuple[int, List[str]]:
        """Evaluate all confirmations (5-point system)"""
        confirmations = 0
        details = []
        
        # 1. Price Action at Key Level
        if price_action['at_key_level']:
            confirmations += 1
            details.append(f"✅ Price at {price_action['key_level_type']} level")
        else:
            details.append("❌ Price not at key level")
        
        # 2. Trend Alignment
        trend_matches = (
            (market['condition'] == 'Bull Trend' and price_action['bias'] == 'Bullish') or
            (market['condition'] == 'Bear Trend' and price_action['bias'] == 'Bearish') or
            (market['condition'] == 'Ranging' and stoch['signal'] in ['buy', 'strong_buy', 'sell', 'strong_sell'])
        )
        
        if trend_matches:
            confirmations += 1
            details.append("✅ Trend alignment confirmed")
        else:
            details.append("❌ Trend misalignment")
        
        # 3. Volume Confirmation
        if volume_ratio > 1.2:
            confirmations += 1
            details.append("✅ Volume confirmation strong")
        else:
            details.append("⚠️ Volume below average")
        
        # 4. Stochastic Signal
        if stoch['strength'] in ['strong', 'moderate']:
            confirmations += 1
            details.append(f"✅ Stochastic: {stoch['details']}")
        else:
            details.append(f"⚠️ Stochastic: {stoch['details']}")
        
        # 5. Market Structure
        if price_action['structure'] != 'Consolidation':
            confirmations += 1
            details.append(f"✅ Market structure: {price_action['structure']}")
        else:
            details.append("⚠️ Market structure unclear")
        
        return confirmations, details
    
    @staticmethod
    def make_final_decision(confirmations: int, market: Dict, stoch: Dict, 
                           price_action: Dict, min_confirmations: int) -> Dict:
        """Make final trading decision"""
        decision = 'HOLD'
        reason = ''
        confidence = 0
        
        if confirmations >= min_confirmations:
            confidence = 80 + (confirmations * 4)
            
            if stoch['signal'] in ['buy', 'strong_buy']:
                decision = 'BUY'
                reason = f"Confirmations: {confirmations}/5. {stoch['details']}. Market: {market['condition']}"
            elif stoch['signal'] in ['sell', 'strong_sell']:
                decision = 'SELL'
                reason = f"Confirmations: {confirmations}/5. {stoch['details']}. Market: {market['condition']}"
        elif confirmations >= 3:
            confidence = 60 + (confirmations * 5)
            decision = 'HOLD'
            reason = f"Moderate confirmations ({confirmations}/5). Wait for better setup."
        else:
            confidence = 20 + (confirmations * 10)
            decision = 'HOLD'
            reason = f"Insufficient confirmations ({confirmations}/5). No trade."
        
        # Adjust for key levels
        if price_action['at_key_level']:
            confidence += 10
        
        confidence = min(95, confidence)
        
        return {
            'decision': decision,
            'reason': reason,
            'confidence': confidence,
            'confirmations': confirmations
        }


# ==================== RISK MANAGER ====================

class RiskManager:
    """Handle position sizing and risk management"""
    
    @staticmethod
    def calculate_position_size(account_balance: float, risk_percent: float,
                               entry_price: float, stop_loss_price: float,
                               leverage: int = 1) -> float:
        """Calculate position size based on risk"""
        risk_amount = account_balance * (risk_percent / 100)
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit == 0:
            return 0
        
        position_size = (risk_amount / risk_per_unit) * leverage
        return round(position_size, 6)
    
    @staticmethod
    def calculate_stop_loss(entry_price: float, volatility: float, 
                          direction: str) -> float:
        """Calculate stop loss based on volatility"""
        sl_percent = min(5, max(1, volatility * 1.5))
        
        if direction == 'BUY':
            stop_loss = entry_price * (1 - sl_percent/100)
        else:  # SELL
            stop_loss = entry_price * (1 + sl_percent/100)
        
        return round(stop_loss, 2)
    
    @staticmethod
    def calculate_take_profit(entry_price: float, stop_loss: float, 
                            direction: str, risk_reward: float = 1.5) -> float:
        """Calculate take profit (1:1.5 risk/reward)"""
        risk_amount = abs(entry_price - stop_loss)
        
        if direction == 'BUY':
            take_profit = entry_price + (risk_amount * risk_reward)
        else:  # SELL
            take_profit = entry_price - (risk_amount * risk_reward)
        
        return round(take_profit, 2)


# ==================== TRADING BOT ====================

class TradingBot:
    """Main Trading Bot Class"""
    
    def __init__(self, config: Config):
        self.config = config
        self.exchange = None
        self.telegram = None
        self.analyzer = MarketAnalyzer()
        self.risk_manager = RiskManager()
        
        # Trading state
        self.active_positions = []
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.trade_history = []
        
        # Safety flags
        self.emergency_stop = config.EMERGENCY_STOP
        self.last_check_date = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize exchange and services"""
        try:
            # Initialize exchange
            self.exchange = ccxt.binance({
                'apiKey': self.config.API_KEY,
                'secret': self.config.API_SECRET,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'
                }
            })
            
            # Set leverage for all symbols if enabled
            if self.config.USE_LEVERAGE:
                if self.config.MULTI_SYMBOL_MODE:
                    for symbol in self.config.SYMBOLS:
                        try:
                            self.exchange.set_leverage(self.config.LEVERAGE, symbol)
                        except Exception as e:
                            logger.warning(f"Could not set leverage for {symbol}: {e}")
                    logger.info(f"Leverage set to {self.config.LEVERAGE}x for all symbols")
                else:
                    self.exchange.set_leverage(self.config.LEVERAGE, self.config.SYMBOL)
                    logger.info(f"Leverage set to {self.config.LEVERAGE}x")
            
            # Initialize Telegram
            if self.config.TELEGRAM_ENABLED:
                self.telegram = TelegramNotifier(
                    self.config.TELEGRAM_BOT_TOKEN,
                    self.config.TELEGRAM_CHAT_ID,
                    True
                )
                symbols_list = ', '.join(self.config.SYMBOLS[:5]) + "..." if self.config.MULTI_SYMBOL_MODE else self.config.SYMBOL
                self.telegram.send_message(
                    f"🤖 <b>Trading Bot Started</b>\n\n"
                    f"Mode: {'Multi-Symbol Scanner' if self.config.MULTI_SYMBOL_MODE else 'Single Symbol'}\n"
                    f"Symbols: {symbols_list}\n"
                    f"Max Positions: {self.config.MAX_POSITIONS}"
                )
            
            logger.info("Bot initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise
    
    def fetch_market_data(self, symbol: str = None) -> pd.DataFrame:
        """Fetch OHLCV data from exchange"""
        try:
            target_symbol = symbol if symbol else self.config.SYMBOL
            ohlcv = self.exchange.fetch_ohlcv(
                target_symbol,
                self.config.TIMEFRAME,
                limit=100
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data for {target_symbol}: {e}")
            raise
    
    def analyze_market(self, data: pd.DataFrame) -> Dict:
        """Run complete market analysis"""
        settings = TRADING_SETTINGS[self.config.TRADING_STYLE]
        
        # Calculate indicators
        k_values, d_values = self.analyzer.calculate_stochastic(
            data, 
            settings['k_length'],
            settings['k_smooth'],
            settings['d_smooth']
        )
        
        # Analyze market condition
        market = self.analyzer.analyze_market_condition(data)
        
        # Analyze price action
        price_action = self.analyzer.analyze_price_action(data)
        
        # Analyze stochastic signal
        stoch = self.analyzer.analyze_stochastic_signal(k_values, d_values, settings)
        
        # Evaluate confirmations
        confirmations, details = self.analyzer.evaluate_confirmations(
            market, price_action, stoch, market['volume_ratio']
        )
        
        # Make final decision
        decision = self.analyzer.make_final_decision(
            confirmations, market, stoch, price_action, 
            self.config.MIN_CONFIRMATIONS
        )
        
        current_price = data['close'].iloc[-1]
        
        return {
            'market': market,
            'price_action': price_action,
            'stoch': stoch,
            'decision': decision,
            'current_price': current_price,
            'confirmations': confirmations,
            'details': details
        }
    
    def scan_all_symbols(self) -> Optional[Tuple[str, Dict]]:
        """Scan all symbols and find best opportunity"""
        if not self.config.MULTI_SYMBOL_MODE:
            return self.config.SYMBOL, None
        
        logger.info(f"\n🔍 Scanning {len(self.config.SYMBOLS)} symbols for opportunities...")
        
        best_symbol = None
        best_analysis = None
        best_score = 0
        
        scan_results = []
        
        for symbol in self.config.SYMBOLS:
            try:
                # Fetch and analyze
                data = self.fetch_market_data(symbol)
                analysis = self.analyze_market(data)
                
                # Calculate score
                decision = analysis['decision']['decision']
                confidence = analysis['decision']['confidence']
                confirmations = analysis['confirmations']
                
                # Only consider BUY or SELL signals
                if decision in ['BUY', 'SELL']:
                    # Score = confidence * (confirmations/5) * 100
                    score = confidence * (confirmations / 5)
                    
                    scan_results.append({
                        'symbol': symbol,
                        'decision': decision,
                        'confidence': confidence,
                        'confirmations': confirmations,
                        'score': score,
                        'price': analysis['current_price']
                    })
                    
                    logger.info(f"  {symbol}: {decision} | Conf: {confidence}% | Confirms: {confirmations}/5 | Score: {score:.1f}")
                    
                    # Update best if higher score
                    if score > best_score:
                        best_score = score
                        best_symbol = symbol
                        best_analysis = analysis
                
                # Small delay to avoid rate limit
                time.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"  {symbol}: Error - {str(e)[:50]}")
                continue
        
        # Send scan summary to Telegram
        if self.telegram and scan_results:
            self._send_scan_summary(scan_results, best_symbol, best_score)
        
        if best_symbol:
            logger.info(f"\n✅ Best Opportunity: {best_symbol} (Score: {best_score:.1f})")
            return best_symbol, best_analysis
        else:
            logger.info("\n⏸️  No clear opportunities found")
            return None, None
    
    def _send_scan_summary(self, results: List[Dict], best_symbol: str, best_score: float):
        """Send scan summary to Telegram"""
        if not results:
            return
        
        # Sort by score
        sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
        
        message = f"🔍 <b>MARKET SCAN COMPLETE</b>\n\n"
        message += f"Scanned: {len(self.config.SYMBOLS)} symbols\n"
        message += f"Signals found: {len(results)}\n\n"
        message += f"<b>TOP 5 OPPORTUNITIES:</b>\n"
        
        for i, result in enumerate(sorted_results, 1):
            emoji = "🟢" if result['decision'] == 'BUY' else "🔴"
            is_best = "⭐" if result['symbol'] == best_symbol else ""
            
            message += f"\n{i}. {emoji} {result['symbol']} {is_best}\n"
            message += f"   Score: {result['score']:.1f} | {result['decision']}\n"
            message += f"   Confidence: {result['confidence']}% | Confirms: {result['confirmations']}/5\n"
        
        if best_symbol:
            message += f"\n✅ <b>SELECTED: {best_symbol}</b>"
        
        self.telegram.send_message(message)
    
    def execute_trade(self, analysis, symbol):
        """Execute trade based on analysis"""
        decision = analysis['decision']['decision']
        
        if decision == 'HOLD':
            logger.info("Decision: HOLD - No trade")
            return
        
        # Safety checks
        if self.emergency_stop:
            logger.warning("Emergency stop active - trade blocked")
            return
        
        if self.daily_trades >= self.config.MAX_DAILY_TRADES:
            logger.warning(f"Daily trade limit reached ({self.config.MAX_DAILY_TRADES})")
            return
        
        if abs(self.daily_pnl) >= self.config.MAX_DAILY_LOSS:
            logger.warning(f"Daily loss limit reached (${self.config.MAX_DAILY_LOSS})")
            self.emergency_stop = True
            return
        
        if len(self.active_positions) >= self.config.MAX_POSITIONS:
            logger.warning(f"Max positions reached ({self.config.MAX_POSITIONS})")
            return
        
        try:
            current_price = analysis['current_price']
            volatility = analysis['market']['volatility']
            
            # Calculate risk parameters
            direction = 'BUY' if decision == 'BUY' else 'SELL'
            stop_loss = self.risk_manager.calculate_stop_loss(current_price, volatility, direction)
            take_profit = self.risk_manager.calculate_take_profit(current_price, stop_loss, direction)
            position_size = self.risk_manager.calculate_position_size(
                self.config.ACCOUNT_BALANCE,
                self.config.RISK_PER_TRADE,
                current_price,
                stop_loss,
                self.config.LEVERAGE if self.config.USE_LEVERAGE else 1
            )
            
            logger.info(f"\nOpening {decision} position:")
            logger.info(f"  Symbol: {symbol}")
            logger.info(f"  Entry: ${current_price:.2f}")
            logger.info(f"  Size: {position_size}")
            logger.info(f"  Stop Loss: ${stop_loss:.2f}")
            logger.info(f"  Take Profit: ${take_profit:.2f}")
            
            # Place market order
            side = 'buy' if decision == 'BUY' else 'sell'
            order = self.exchange.create_market_order(symbol, side, position_size)
            logger.info(f"  Market order placed: {order['id']}")
            
            # Clean symbol for Binance API (remove /)
            symbol_clean = symbol.replace('/', '')
            sl_side = 'SELL' if decision == 'BUY' else 'BUY'
            
            # Place Stop Loss order with stopPrice
            try:
                stop_order = self.exchange.fapiPrivatePostOrder({
                    'symbol': symbol_clean,
                    'side': sl_side,
                    'type': 'STOP_MARKET',
                    'quantity': position_size,
                    'stopPrice': stop_loss,
                    'reduceOnly': 'true'
                })
                logger.info(f"  Stop Loss placed: {stop_order.get('orderId', 'N/A')}")
            except Exception as e:
                logger.error(f"  Stop Loss error: {e}")
                # Continue even if SL fails
            
            # Place Take Profit order with stopPrice
            try:
                tp_order = self.exchange.fapiPrivatePostOrder({
                    'symbol': symbol_clean,
                    'side': sl_side,
                    'type': 'TAKE_PROFIT_MARKET',
                    'quantity': position_size,
                    'stopPrice': take_profit,
                    'reduceOnly': 'true'
                })
                logger.info(f"  Take Profit placed: {tp_order.get('orderId', 'N/A')}")
            except Exception as e:
                logger.error(f"  Take Profit error: {e}")
                # Continue even if TP fails
            
            # Record position
            position = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': decision,
                'entry_price': current_price,
                'size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': analysis['decision']['confidence'],
                'confirmations': analysis['confirmations'],
                'order_id': order['id'],
                'sl_order_id': stop_order.get('orderId', None) if 'stop_order' in locals() else None,
                'tp_order_id': tp_order.get('orderId', None) if 'tp_order' in locals() else None
            }
            
            self.active_positions.append(position)
            self.daily_trades += 1
            
            # Log trade to file
            if self.config.LOG_TRADES:
                self._log_trade(position)
            
            # Send Telegram notification
            if self.telegram:
                position_value = position_size * current_price
                self.telegram.send_trade_alert(
                    decision,
                    symbol,
                    current_price,
                    position_size,
                    analysis['decision']['confidence'],
                    stop_loss,
                    take_profit,
                    self.config.LEVERAGE if self.config.USE_LEVERAGE else 1,
                    position_value
                )
            
            logger.info(f"\n>>> TRADE EXECUTED: {decision} {symbol} @ ${current_price:.2f} <<<\n")
            
        except Exception as e:
            error_msg = f"Trade execution error: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_error_alert(error_msg[:500])
            """Execute trade based on analysis"""
        decision = analysis['decision']['decision']
        
        if decision == 'HOLD':
            logger.info("Decision: HOLD - No trade executed")
            return
        
        # Safety checks
        if self.emergency_stop:
            logger.warning("Emergency stop active - trade blocked")
            return
        
        if self.daily_trades >= self.config.MAX_DAILY_TRADES:
            logger.warning(f"Daily trade limit reached ({self.config.MAX_DAILY_TRADES})")
            return
        
        if abs(self.daily_pnl) >= self.config.MAX_DAILY_LOSS:
            logger.warning(f"Daily loss limit reached (${self.config.MAX_DAILY_LOSS})")
            self.emergency_stop = True
            return
        
        if len(self.active_positions) >= self.config.MAX_POSITIONS:
            logger.warning(f"Max positions reached ({self.config.MAX_POSITIONS})")
            return
        
        try:
            current_price = analysis['current_price']
            volatility = analysis['market']['volatility']
            
            # Calculate risk parameters
            direction = 'BUY' if decision == 'BUY' else 'SELL'
            stop_loss = self.risk_manager.calculate_stop_loss(
                current_price, volatility, direction
            )
            take_profit = self.risk_manager.calculate_take_profit(
                current_price, stop_loss, direction
            )
            position_size = self.risk_manager.calculate_position_size(
                self.config.ACCOUNT_BALANCE,
                self.config.RISK_PER_TRADE,
                current_price,
                stop_loss,
                self.config.LEVERAGE if self.config.USE_LEVERAGE else 1
            )
            
            # Place order
            side = 'buy' if decision == 'BUY' else 'sell'
            order = self.exchange.create_market_order(
                symbol,
                side,
                position_size
            )
            
            # Place stop loss
            sl_side = 'sell' if decision == 'BUY' else 'buy'
            stop_order = self.exchange.create_order(
                symbol,
                'stop_market',
                sl_side,
                position_size,
                stop_loss
            )
            
            # Place take profit
            tp_order = self.exchange.create_order(
                symbol,
                'take_profit_market',
                sl_side,
                position_size,
                take_profit
            )
            
            # Record position
            position = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': decision,
                'entry_price': current_price,
                'size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': analysis['decision']['confidence'],
                'confirmations': analysis['confirmations'],
                'order_id': order['id']
            }
            
            self.active_positions.append(position)
            self.daily_trades += 1
            
            # Log trade
            if self.config.LOG_TRADES:
                self._log_trade(position)
            
            # Send notification
            if self.telegram:
                position_value = position_size * current_price
                self.telegram.send_trade_alert(
                    decision,
                    symbol,
                    current_price,
                    position_size,
                    analysis['decision']['confidence'],
                    stop_loss,
                    take_profit,
                    self.config.LEVERAGE if self.config.USE_LEVERAGE else 1,
                    position_value
                )
            
            logger.info(f"""
╔══════════════════════════════════════╗
║       TRADE EXECUTED                 ║
╠══════════════════════════════════════╣
║ Symbol:       {symbol:>20} ║
║ Action:       {decision:>20} ║
║ Price:        ${current_price:>19.2f} ║
║ Size:         {position_size:>20.6f} ║
║ Stop Loss:    ${stop_loss:>19.2f} ║
║ Take Profit:  ${take_profit:>19.2f} ║
║ Confidence:   {analysis['decision']['confidence']:>19}% ║
╚══════════════════════════════════════╝
            """)
            
        except Exception as e:
            error_msg = f"Trade execution error: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_error_alert(error_msg)
    
    def check_positions(self):
        """Check and update active positions"""
        if not self.active_positions:
            return
        
        try:
            # Get current balance
            balance_info = self.exchange.fetch_balance()
            current_balance = balance_info['USDT']['total']
            
            for position in self.active_positions[:]:
                # Fetch current price
                ticker = self.exchange.fetch_ticker(self.config.SYMBOL)
                current_price = ticker['last']
                
                # Calculate unrealized P&L
                entry_price = position['entry_price']
                size = position['size']
                
                if position['side'] == 'BUY':
                    unrealized_pnl = (current_price - entry_price) * size
                else:
                    unrealized_pnl = (entry_price - current_price) * size
                
                position['unrealized_pnl'] = unrealized_pnl
                position['current_price'] = current_price
                
                # Check if position hit stop loss or take profit
                hit_sl = False
                hit_tp = False
                
                if position['side'] == 'BUY':
                    hit_sl = current_price <= position['stop_loss']
                    hit_tp = current_price >= position['take_profit']
                else:
                    hit_sl = current_price >= position['stop_loss']
                    hit_tp = current_price <= position['take_profit']
                
                # Close position if SL or TP hit
                if hit_sl or hit_tp:
                    reason = "Stop Loss Hit" if hit_sl else "Take Profit Hit"
                    pnl_percent = (unrealized_pnl / (entry_price * size)) * 100
                    
                    # Send close notification
                    if self.telegram:
                        self.telegram.send_position_closed_alert(
                            position['side'],
                            self.config.SYMBOL,
                            entry_price,
                            current_price,
                            size,
                            unrealized_pnl,
                            pnl_percent,
                            reason,
                            current_balance
                        )
                    
                    # Update daily P&L
                    self.daily_pnl += unrealized_pnl
                    
                    # Log to history
                    position['exit_price'] = current_price
                    position['pnl'] = unrealized_pnl
                    position['exit_reason'] = reason
                    position['exit_time'] = datetime.now().isoformat()
                    
                    logger.info(f"""
╔══════════════════════════════════════╗
║       POSITION CLOSED                ║
╠══════════════════════════════════════╣
║ Reason:       {reason:>20} ║
║ Entry:        ${entry_price:>8.2f}            ║
║ Exit:         ${current_price:>8.2f}            ║
║ P&L:          ${unrealized_pnl:>8.2f}            ║
║ P&L%:         {pnl_percent:>7.2f}%            ║
║ Balance:      ${current_balance:>8.2f}            ║
╚══════════════════════════════════════╝
                    """)
                    
                    # Remove from active
                    self.active_positions.remove(position)
            
            # Send position update every 5 minutes
            if hasattr(self, '_last_position_update'):
                time_since_update = (datetime.now() - self._last_position_update).seconds
                if time_since_update >= 300:  # 5 minutes
                    if self.active_positions and self.telegram:
                        total_unrealized = sum(p.get('unrealized_pnl', 0) for p in self.active_positions)
                        self.telegram.send_position_update(
                            self.active_positions,
                            total_unrealized,
                            current_balance
                        )
                    self._last_position_update = datetime.now()
            else:
                self._last_position_update = datetime.now()
                
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
    
    def _calculate_pnl(self, position: Dict, closed_order: Dict) -> float:
        """Calculate profit/loss for closed position"""
        entry = position['entry_price']
        exit_price = closed_order.get('average', entry)
        size = position['size']
        
        if position['side'] == 'BUY':
            pnl = (exit_price - entry) * size
        else:
            pnl = (entry - exit_price) * size
        
        return pnl
    
    def _log_trade(self, trade: Dict):
        """Log trade to file"""
        try:
            self.trade_history.append(trade)
            with open(self.config.LOG_FILE, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    def _reset_daily_counters(self):
        """Reset daily counters"""
        current_date = datetime.now().date()
        
        if self.last_check_date != current_date:
            # Get current balance
            try:
                balance_info = self.exchange.fetch_balance()
                current_balance = balance_info['USDT']['total']
            except:
                current_balance = self.config.ACCOUNT_BALANCE
            
            # Send daily summary
            if self.telegram and self.daily_trades > 0:
                win_rate, winners, losers = self._calculate_win_rate()
                self.telegram.send_daily_summary(
                    self.daily_trades,
                    self.daily_pnl,
                    win_rate,
                    current_balance,
                    winners,
                    losers
                )
            
            # Reset counters
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.emergency_stop = self.config.EMERGENCY_STOP
            self.last_check_date = current_date
            
            logger.info("Daily counters reset")
    
    def _calculate_win_rate(self) -> Tuple[float, int, int]:
        """Calculate win rate from trade history"""
        if not self.trade_history:
            return 0.0, 0, 0
        
        winners = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
        losers = len(self.trade_history) - winners
        win_rate = (winners / len(self.trade_history)) * 100
        
        return win_rate, winners, losers
    
    def run(self):
        """Main bot loop"""
        logger.info("🚀 Bot started - Trading in progress...")
        
        if self.telegram:
            self.telegram.send_message(
                f"🚀 <b>Bot Started</b>\n\n"
                f"Symbol: {self.config.SYMBOLS}\n"
                f"Timeframe: {self.config.TIMEFRAME}\n"
                f"Risk: {self.config.RISK_PER_TRADE}%\n"
                f"Leverage: {self.config.LEVERAGE}x" if self.config.USE_LEVERAGE else "Leverage: 1x"
            )
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                
                # Reset daily counters if new day
                self._reset_daily_counters()
                
                # Check if emergency stop
                if self.emergency_stop:
                    logger.warning("⚠️ Emergency stop active - bot paused")
                    time.sleep(60)
                    continue
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*60}")
                
                try:
                    # Fetch market data
                    logger.info("📊 Fetching market data...")
                    data = self.fetch_market_data()
                    
                    # Analyze market
                    logger.info("🔍 Analyzing market...")
                    analysis = self.analyze_market(data)
                    
                    # Log analysis results
                    logger.info(f"""
┌─────────────────────────────────────┐
│ MARKET ANALYSIS                     │
├─────────────────────────────────────┤
│ Condition:    {analysis['market']['condition']:>15} │
│ Trend:        {analysis['market']['trend_strength']:>15} │
│ Volatility:   {analysis['market']['volatility']:>14.2f}% │
│ Volume:       {analysis['market']['volume_activity']:>15} │
│ Price:        ${analysis['current_price']:>14.2f} │
├─────────────────────────────────────┤
│ CONFIRMATIONS: {analysis['confirmations']}/5                │
├─────────────────────────────────────┤
│ DECISION:     {analysis['decision']['decision']:>15} │
│ Confidence:   {analysis['decision']['confidence']:>14}% │
└─────────────────────────────────────┘
                    """)
                    
                    # Print confirmation details
                    logger.info("Confirmation Details:")
                    for detail in analysis['details']:
                        logger.info(f"  {detail}")
                    
                    # Check existing positions
                    self.check_positions()
                    
                    # Execute trade if conditions met
                    if analysis['decision']['decision'] != 'HOLD':
                        logger.info(f"✅ Trade signal detected: {analysis['decision']['decision']}")
                        logger.info(f"Reason: {analysis['decision']['reason']}")
                        self.execute_trade(analysis)
                    else:
                        logger.info("⏸️  No trade signal - Holding")
                    
                    # Display active positions
                    if self.active_positions:
                        logger.info(f"\n📈 Active Positions: {len(self.active_positions)}")
                        for i, pos in enumerate(self.active_positions, 1):
                            logger.info(f"  Position {i}: {pos['side']} {pos['size']} @ ${pos['entry_price']}")
                    
                    # Display daily stats
                    logger.info(f"\n📊 Daily Stats:")
                    logger.info(f"  Trades: {self.daily_trades}/{self.config.MAX_DAILY_TRADES}")
                    logger.info(f"  P&L: ${self.daily_pnl:.2f}")
                    
                except Exception as e:
                    error_msg = f"Error in trading loop: {e}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    if self.telegram:
                        self.telegram.send_error_alert(error_msg[:500])
                
                # Wait before next iteration
                logger.info(f"\n⏳ Waiting {self.config.CHECK_INTERVAL} seconds...\n")
                time.sleep(self.config.CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user")
            if self.telegram:
                self.telegram.send_message("🛑 <b>Bot Stopped</b>")
        except Exception as e:
            error_msg = f"Critical error: {e}\n{traceback.format_exc()}"
            logger.error(error_msg)
            if self.telegram:
                self.telegram.send_error_alert(error_msg[:500])


# ==================== MAIN FUNCTION ====================

def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║         🏆 MASTER TRADER BOT v2.0 🏆                     ║
║                                                          ║
║     Advanced Binance Futures Trading System              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

Features:
  ✅ Multi-timeframe analysis
  ✅ 5-point confirmation system
  ✅ Auto risk management
  ✅ Telegram notifications
  ✅ Position tracking
  ✅ Emergency stop
  ✅ Daily P&L tracking

WARNING: Trading involves risk. Use at your own risk.
    """)
    
    # Validate configuration
    if Config.API_KEY == "YOUR_BINANCE_API_KEY":
        print("\n❌ ERROR: Please configure your API credentials in Config class")
        print("   Edit the API_KEY and API_SECRET values")
        return
    
    print("\n⚙️  Configuration:")
    print(f"   Symbol: {Config.SYMBOLS}")
    print(f"   Timeframe: {Config.TIMEFRAME}")
    print(f"   Trading Style: {Config.TRADING_STYLE}")
    print(f"   Risk per Trade: {Config.RISK_PER_TRADE}%")
    print(f"   Leverage: {Config.LEVERAGE}x" if Config.USE_LEVERAGE else "   Leverage: 1x")
    print(f"   Min Confirmations: {Config.MIN_CONFIRMATIONS}/5")
    print(f"   Check Interval: {Config.CHECK_INTERVAL}s")
    
    response = input("\n▶️  Start bot? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("Bot startup cancelled")
        return
    
    try:
        # Initialize and run bot
        bot = TradingBot(Config)
        bot.run()
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
