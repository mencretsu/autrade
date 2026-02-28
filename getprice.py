import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import csv

# ============================================
# AMBIL DATA BTC DARI BINANCE
# ============================================
def get_btc_history(start_date, end_date, interval='1d'):
    def to_ms(date_str):
        return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp() * 1000)
    
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    start_ms = to_ms(start_date)
    end_ms = to_ms(end_date)
    
    while start_ms < end_ms:
        params = {
            "symbol": "BTCUSDT",
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        start_ms = data[-1][6] + 1
    
    df = pd.DataFrame(all_data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base_vol",
        "taker_quote_vol","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
    
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col])
    
    return df[["open_time","open","high","low","close","volume"]]

# ============================================
# VERSI 1: ANGKA PERSIS SAMA
# ============================================
def find_exact_levels(df, col="high", min_occurrence=2):
    levels = defaultdict(list)
    
    for idx, row in df.iterrows():
        price = round(float(row[col]), 2)
        levels[price].append(row['open_time'])
    
    result = {}
    for price, dates in levels.items():
        if len(dates) >= min_occurrence:
            result[price] = dates
    
    return result

# ============================================
# VERSI 2: CLUSTERING (IMPROVED)
# ============================================
def find_clustered_levels(df, col="high", tolerance_percent=0.5, min_touches=2):
    def within_tolerance(a, b):
        return abs(a - b) <= a * (tolerance_percent / 100)
    
    levels = []
    visited = set()
    
    for i in range(len(df)):
        if i in visited:
            continue
        
        current_price = float(df.iloc[i][col])
        current_date = df.iloc[i]['open_time']
        cluster_prices = [current_price]
        cluster_dates = [current_date]
        
        for j in range(i + 1, len(df)):
            if j in visited:
                continue
            compare_price = float(df.iloc[j][col])
            if within_tolerance(current_price, compare_price):
                cluster_prices.append(compare_price)
                cluster_dates.append(df.iloc[j]['open_time'])
                visited.add(j)
        
        if len(cluster_prices) >= min_touches:
            avg_price = sum(cluster_prices) / len(cluster_prices)
            levels.append({
                'price': avg_price,
                'dates': sorted(cluster_dates),
                'count': len(cluster_prices)
            })
    
    # 🔥 SORTING: DARI TOUCH TERBANYAK KE TERKECIL
    return sorted(levels, key=lambda x: -x['count'])

# ============================================
# GET TOP TWO WITH 3.5% - 4% PRICE DIFFERENCE
# ============================================
def get_top_two_smart(support_levels, resistance_levels, min_diff_percent=3.5, max_diff_percent=4.0):
    """
    Cari 2 level terbaik dari kombinasi support/resistance
    dengan syarat: selisih harga antara 3.5% - 4%
    
    Priority:
    1. 2 support terbanyak (jika 3.5% <= diff <= 4%)
    2. 2 resistance terbanyak (jika 3.5% <= diff <= 4%)
    3. 1 support + 1 resistance terbanyak (jika 3.5% <= diff <= 4%)
    """
    all_candidates = []
    
    # Gabungkan semua level dengan label
    for s in support_levels:
        all_candidates.append({
            'price': s['price'],
            'count': s['count'],
            'dates': s['dates'],
            'type': 'SUPPORT'
        })
    
    for r in resistance_levels:
        all_candidates.append({
            'price': r['price'],
            'count': r['count'],
            'dates': r['dates'],
            'type': 'RESISTANCE'
        })
    
    # Sort by touch count (descending)
    all_candidates.sort(key=lambda x: -x['count'])
    
    if len(all_candidates) < 2:
        return None, None
    
    # Cari pasangan terbaik dengan diff antara 3.5% - 4%
    top1 = all_candidates[0]
    top2 = None
    
    for candidate in all_candidates[1:]:
        price_diff_percent = abs(candidate['price'] - top1['price']) / top1['price'] * 100
        if min_diff_percent <= price_diff_percent <= max_diff_percent:
            top2 = candidate
            break
    
    return top1, top2

# ============================================
# FUNGSI UTAMA
# ============================================
def run_analysis():
    print("\n" + "="*70)
    print("   BTC SUPPORT & RESISTANCE FINDER")
    print("="*70)
    
    print("\nPilih Timeframe:")
    print("1. 1m  (1 menit)")
    print("2. 5m  (5 menit)")
    print("3. 15m (15 menit)")
    print("4. 1h  (1 jam)")
    print("5. 4h  (4 jam)")
    print("6. 1d  (1 hari)")
    print("7. 1w  (1 minggu)")
    
    timeframe_map = {
        "1": "1m", "2": "5m", "3": "15m", "4": "1h",
        "5": "4h", "6": "1d", "7": "1w"
    }
    
    choice = input("\nMasukkan pilihan (1-7): ").strip()
    interval = timeframe_map.get(choice, "1d")
    
    print("\nPilih Periode:")
    print("1. 1 Bulan terakhir")
    print("2. 2 Bulan terakhir")
    print("3. 3 Bulan terakhir")
    print("4. 6 Bulan terakhir")
    print("5. 12 Bulan terakhir")
    
    period_choice = input("\nMasukkan pilihan (1-5): ").strip()
    period_map = {"1": 1, "2": 2, "3": 3, "4": 6, "5": 12}
    months = period_map.get(period_choice, 1)
    
    end_date_obj = datetime.now()
    start_date_obj = end_date_obj - relativedelta(months=months)
    
    start_date = start_date_obj.strftime("%Y-%m-%d")
    end_date = end_date_obj.strftime("%Y-%m-%d")
    
    print("\nPilih metode:")
    print("1. Angka PERSIS SAMA (exact match)")
    print("2. Clustering dengan toleransi (area mirip)")
    method = input("\nPilihan (1/2): ").strip()
    
    print(f"\n🔄 Mengambil data BTC dengan timeframe {interval}...")
    df = get_btc_history(start_date, end_date, interval=interval)
    
    print(f"\n📊 Total data: {len(df)} candles")
    print(f"📅 Dari: {df['open_time'].iloc[0]}")
    print(f"📅 Sampai: {df['open_time'].iloc[-1]}")
    print(f"💰 Harga terakhir: ${df['close'].iloc[-1]:,.2f}")
    
    print("\n🔍 Mencari level support & resistance...\n")
    
    # =====================================================================================
    # EXACT MATCH
    # =====================================================================================
    if method == "1":
        print("="*70)
        print("   📈 RESISTANCE LEVELS (Exact - Sorted by Touch)")
        print("="*70)
        resistance_exact = find_exact_levels(df, col="high", min_occurrence=2)
        if resistance_exact:
            sorted_res = sorted(resistance_exact.items(), key=lambda x: -len(x[1]))
            for price, dates in sorted_res:
                dates_str = ", ".join([d.strftime("%Y-%m-%d %H:%M") for d in dates[:5]])
                if len(dates) > 5:
                    dates_str += f" ... (+{len(dates)-5} more)"
                print(f"🔴 ${price:,.2f}  [{dates_str}]  ({len(dates)}x)")
        else:
            print("Tidak ada resistance exact match.")
        
        print("\n" + "="*70)
        print("   📉 SUPPORT LEVELS (Exact - Sorted by Touch)")
        print("="*70)
        support_exact = find_exact_levels(df, col="low", min_occurrence=2)
        if support_exact:
            sorted_sup = sorted(support_exact.items(), key=lambda x: -len(x[1]))
            for price, dates in sorted_sup:
                dates_str = ", ".join([d.strftime("%Y-%m-%d %H:%M") for d in dates[:5]])
                if len(dates) > 5:
                    dates_str += f" ... (+{len(dates)-5} more)"
                print(f"🟢 ${price:,.2f}  [{dates_str}]  ({len(dates)}x)")
        else:
            print("Tidak ada support exact match.")

    # =====================================================================================
    # CLUSTERING
    # =====================================================================================
    else:
        print("="*70)
        print("   📈 RESISTANCE LEVELS (Clustering - Sorted by Touch)")
        print("="*70)
        resistance_cluster = find_clustered_levels(df, col="high", tolerance_percent=0.5, min_touches=2)
        for r in resistance_cluster:
            dates = r['dates']
            dates_str = ", ".join([d.strftime("%Y-%m-%d %H:%M") for d in dates[:5]])
            if len(dates) > 5:
                dates_str += f" ... (+{len(dates)-5} more)"
            print(f"🔴 ${r['price']:,.2f} ({r['count']} touches) [{dates_str}]")

        print("\n" + "="*70)
        print("   📉 SUPPORT LEVELS (Clustering - Sorted by Touch)")
        print("="*70)
        support_cluster = find_clustered_levels(df, col="low", tolerance_percent=0.5, min_touches=2)
        for s in support_cluster:
            dates = s['dates']
            dates_str = ", ".join([d.strftime("%Y-%m-%d %H:%M") for d in dates[:5]])
            if len(dates) > 5:
                dates_str += f" ... (+{len(dates)-5} more)"
            print(f"🟢 ${s['price']:,.2f} ({s['count']} touches) [{dates_str}]")

        # 🔥 GET TOP 1 & TOP 2 DENGAN FILTER 3.5% - 4% DIFFERENCE
        top1, top2 = get_top_two_smart(support_cluster, resistance_cluster, min_diff_percent=3.5, max_diff_percent=4.0)

        if top1:
            print("\n=== TOP LEVELS BY TOUCH (Diff: 3.5% - 4%) ===")
            print(f"Line 56 (Touch 1): ${top1['price']:.2f}  ({top1['count']} touches) [{top1['type']}]")
        
        if top2:
            price_diff = abs(top2['price'] - top1['price']) / top1['price'] * 100
            print(f"Line 55 (Touch 2): ${top2['price']:.2f}  ({top2['count']} touches) [{top2['type']}] [Diff: {price_diff:.2f}%]")
        else:
            print("Line 55 (Touch 2): TIDAK ADA (tidak ada level dengan selisih 3.5% - 4%)")

        # ================================================================
        #  GRID SYSTEM D54 ↓ 0 + D57 ↑ 200
        # ================================================================
        if top1 and top2:
            D = {}
            D[56] = top1['price']
            D[55] = top2['price']

            # KE BAWAH D54 → D0
            for line in range(54, -1, -1):
                D[line] = (D[line+1]**2) / D[line+2]

            # KE ATAS 57 → 200
            for line in range(57, 201):
                D[line] = (D[line-1]**2) / D[line-2]

            print("\n=== GENERATED GRID LEVELS ===")
            
            # Deteksi arah grid BERDASARKAN HARGA Line 0 vs Line 200
            grid_direction = "descending" if D[0] > D[200] else "ascending"
            print(f"Grid Direction: {grid_direction.upper()} (Line 0 = ${D[0]:,.2f}, Line 200 = ${D[200]:,.2f})")
            
            print(f"\n{'Line':<6} {'Price':>12} {'SL Long':>12} {'TP Long':>12} {'SL Short':>12} {'TP Short':>12}")
            print("-" * 80)
            
            for i in range(0, 201):
                price = D[i]
                sl_long = price * 0.995   # -0.5% untuk LONG
                sl_short = price * 1.005  # +0.5% untuk SHORT
                
                # TP LONG selalu ke harga LEBIH TINGGI, TP SHORT selalu ke harga LEBIH RENDAH
                if grid_direction == "descending":
                    # Grid turun: Line 0 paling TINGGI, Line 200 paling RENDAH
                    # LONG → TP ke line dengan HARGA LEBIH TINGGI (i-1, ke atas)
                    # SHORT → TP ke line dengan HARGA LEBIH RENDAH (i+1, ke bawah)
                    tp_long = D[i-1] if i > 0 else None
                    tp_short = D[i+1] if i < 200 else None
                else:
                    # Grid naik: Line 0 paling RENDAH, Line 200 paling TINGGI
                    # LONG → TP ke line dengan HARGA LEBIH TINGGI (i+1, ke atas)
                    # SHORT → TP ke line dengan HARGA LEBIH RENDAH (i-1, ke bawah)
                    tp_long = D[i+1] if i < 200 else None
                    tp_short = D[i-1] if i > 0 else None
                
                tp_long_str = f"${tp_long:>11,.2f}" if tp_long else "     N/A    "
                tp_short_str = f"${tp_short:>11,.2f}" if tp_short else "     N/A    "
                
                print(f"{i:<6} ${price:>11,.2f} ${sl_long:>11,.2f} {tp_long_str} ${sl_short:>11,.2f} {tp_short_str}")
            
            # ================================================================
            # EXPORT TO CSV
            # ================================================================
            export = input("\n💾 Export grid ke CSV? (y/n): ").strip().lower()
            if export == "y":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"btc_grid_levels_{timestamp}.csv"
                
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Header
                    writer.writerow(['Line', 'Price', 'SL Long', 'TP Long', 'SL Short', 'TP Short'])
                    
                    # Data
                    for i in range(0, 201):
                        price = D[i]
                        sl_long = price * 0.995
                        sl_short = price * 1.005
                        
                        if grid_direction == "descending":
                            tp_long = D[i-1] if i > 0 else None
                            tp_short = D[i+1] if i < 200 else None
                        else:
                            tp_long = D[i+1] if i < 200 else None
                            tp_short = D[i-1] if i > 0 else None
                        
                        tp_long_val = f"{tp_long:.2f}" if tp_long else "N/A"
                        tp_short_val = f"{tp_short:.2f}" if tp_short else "N/A"
                        
                        writer.writerow([
                            i,
                            f"{price:.2f}",
                            f"{sl_long:.2f}",
                            tp_long_val,
                            f"{sl_short:.2f}",
                            tp_short_val
                        ])
                
                print(f"✅ File berhasil di-export: {filename}")
                print(f"📂 Buka dengan VSCode atau Excel!")
        else:
            print("\n⚠️ Grid tidak bisa di-generate (top2 tidak memenuhi syarat 3.5% - 4%)")


# ============================================
# MAIN LOOP
# ============================================
if __name__ == "__main__":
    while True:
        run_analysis()
        print("\n")
        x = input("Kembali ke menu? (y/n): ").strip().lower()
        if x != "y":
            print("\nExiting...\n")
            break
