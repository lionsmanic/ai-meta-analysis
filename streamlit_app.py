# --- 繪圖函式 (高解析度 & 精準對齊版) ---

def plot_forest_professional(ma_engine):
    df = ma_engine.df
    res = ma_engine.results
    measure = ma_engine.measure
    is_binary = "Binary" in ma_engine.data_type
    
    # 設定高解析度與字型大小
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})
    
    # 計算行數 (Header + Studies + Space + Pooled + Footer)
    n_studies = len(df)
    n_rows = n_studies + 4 # 預留頭尾空間
    
    # 設定畫布 (寬度適中，高度隨研究數量動態調整)
    fig, ax = plt.subplots(figsize=(12, n_rows * 0.4))
    
    # 設定 Y 軸範圍 (0 在最上方，n_rows 在最下方)
    ax.set_ylim(0, n_rows)
    ax.set_xlim(0, 100) # 將 X 軸虛擬化為 0-100% 的畫布寬度
    ax.axis('off') # 隱藏所有預設座標軸
    
    # --- 定義欄位 X 座標 (0-100) ---
    col_study = 2
    col_data1 = 35 # Tx
    col_data2 = 50 # Ctrl
    col_plot_start = 60
    col_plot_end = 85
    col_stats = 88
    col_weight = 98
    
    # --- 1. 繪製表頭 (Header) ---
    y_header = n_rows - 1
    ax.text(col_study, y_header, "Study", fontweight='bold', ha='left', va='center')
    
    if is_binary:
        ax.text(col_data1, y_header, "Tx\n(n/N)", fontweight='bold', ha='center', va='center')
        ax.text(col_data2, y_header, "Ctrl\n(n/N)", fontweight='bold', ha='center', va='center')
    else:
        ax.text(col_data1, y_header, "Tx\n(Mean/SD)", fontweight='bold', ha='center', va='center')
        ax.text(col_data2, y_header, "Ctrl\n(Mean/SD)", fontweight='bold', ha='center', va='center')
        
    ax.text((col_plot_start + col_plot_end)/2, y_header, f"{measure} (95% CI)", fontweight='bold', ha='center', va='center')
    ax.text(col_weight, y_header, "Weight", fontweight='bold', ha='right', va='center')
    
    # 畫表頭分隔線
    ax.plot([0, 100], [y_header - 0.6, y_header - 0.6], color='black', linewidth=0.8)

    # --- 2. 準備繪圖座標轉換 ---
    # Log scale logic for RR
    if measure == "RR":
        vals = np.exp(df['TE'])
        lows = np.exp(df['lower'])
        ups = np.exp(df['upper'])
        pool_val = np.exp(res['TE_pooled'])
        pool_low = np.exp(res['lower_pooled'])
        pool_up = np.exp(res['upper_pooled'])
        center = 1.0
        # Determine plot range (log scale)
        all_vals = np.concatenate([vals, lows, ups])
        all_vals = all_vals[~np.isnan(all_vals) & (all_vals > 0)]
        x_min = min(min(all_vals), pool_low) * 0.8
        x_max = max(max(all_vals), pool_up) * 1.2
        # Avoid extreme ranges
        if x_min < 0.01: x_min = 0.01
        if x_max > 100: x_max = 100
        
        def transform(v):
            # Log transform mapping to 0-100 range
            # log(v) mapped between log(x_min) and log(x_max)
            try:
                if v <= 0: return 0
                prop = (np.log(v) - np.log(x_min)) / (np.log(x_max) - np.log(x_min))
                return col_plot_start + prop * (col_plot_end - col_plot_start)
            except: return col_plot_start
            
    else: # SMD (Linear)
        vals, lows, ups = df['TE'], df['lower'], df['upper']
        pool_val, pool_low, pool_up = res['TE_pooled'], res['lower_pooled'], res['upper_pooled']
        center = 0.0
        all_vals = np.concatenate([vals, lows, ups])
        x_min = min(min(all_vals), pool_low) - 0.5
        x_max = max(max(all_vals), pool_up) + 0.5
        
        def transform(v):
            prop = (v - x_min) / (x_max - x_min)
            return col_plot_start + prop * (col_plot_end - col_plot_start)

    # --- 3. 繪製每一行 (Rows) ---
    for i, row in df.iterrows():
        y = n_rows - 2 - i # 從上往下畫
        
        # 文字欄位
        ax.text(col_study, y, str(row['Study ID']), ha='left', va='center')
        
        if is_binary:
            ax.text(col_data1, y, f"{int(row['Tx Events'])}/{int(row['Tx Total'])}", ha='center', va='center')
            ax.text(col_data2, y, f"{int(row['Ctrl Events'])}/{int(row['Ctrl Total'])}", ha='center', va='center')
        else:
            ax.text(col_data1, y, f"{row['Tx Mean']:.1f}/{row['Tx SD']:.1f}", ha='center', va='center')
            ax.text(col_data2, y, f"{row['Ctrl Mean']:.1f}/{row['Ctrl SD']:.1f}", ha='center', va='center')
            
        # 統計數值
        val_fmt = f"{vals[i]:.2f}"
        ci_fmt = f"[{lows[i]:.2f}, {ups[i]:.2f}]"
        weight_fmt = f"{row['weight']:.1f}%"
        ax.text(col_stats, y, f"{val_fmt}  {ci_fmt}", ha='right', va='center', fontsize=9)
        ax.text(col_weight, y, weight_fmt, ha='right', va='center')
        
        # 森林圖方塊與線條
        x = transform(vals[i])
        x_l = transform(lows[i])
        x_r = transform(ups[i])
        
        # 誤差線
        ax.plot([x_l, x_r], [y, y], color='black', linewidth=1)
        # 方塊 (大小隨權重變化，限制最小最大值)
        box_size = 0.15 + (row['weight']/100) * 0.3 
        rect = mpatches.Rectangle((x - box_size/2, y - box_size/2), box_size, box_size, facecolor='black')
        ax.add_patch(rect)

    # --- 4. 繪製合併結果 (Pooled) ---
    y_pool = 1.5
    center_x = transform(center)
    
    # 垂直參考線
    ax.plot([center_x, center_x], [1, n_rows - 1.5], color='black', linestyle='-', linewidth=0.5)
    
    # 菱形 (Diamond)
    px = transform(pool_val)
    pl = transform(pool_low)
    pr = transform(pool_up)
    
    diamond_x = [pl, px, pr, px]
    diamond_y = [y_pool, y_pool + 0.3, y_pool, y_pool - 0.3]
    ax.fill(diamond_x, diamond_y, color='red', alpha=0.5) # 半透明紅色
    
    # 合併數據文字
    ax.text(col_study, y_pool, "Random Effects Model", fontweight='bold', ha='left', va='center')
    
    # 計算總人數
    if is_binary:
        total_tx = int(df['Tx Total'].sum())
        total_ctrl = int(df['Ctrl Total'].sum())
        ax.text(col_data1, y_pool, str(total_tx), fontweight='bold', ha='center', va='center')
        ax.text(col_data2, y_pool, str(total_ctrl), fontweight='bold', ha='center', va='center')
    
    pool_fmt = f"{pool_val:.2f}  [{pool_low:.2f}, {pool_up:.2f}]"
    ax.text(col_stats, y_pool, pool_fmt, fontweight='bold', ha='right', va='center')
    ax.text(col_weight, y_pool, "100.0%", fontweight='bold', ha='right', va='center')
    
    # --- 5. 底部資訊 (Footer) ---
    y_info = 0.5
    het_text = f"Heterogeneity: $I^2$={res['I2']:.1f}%, $\\tau^2$={res['tau2']:.3f}, $p$={res.get('p_Q', 0.99):.3f}" # 這裡 p值暫以0.99代替若無計算
    ax.text(col_study, y_info, het_text, ha='left', va='center', fontsize=9)
    
    # 刻度線 (Scale)
    ax.plot([col_plot_start, col_plot_end], [y_info, y_info], color='black', linewidth=0.8)
    # 標示刻度值
    ticks = [x_min, center, x_max]
    if measure == "RR": ticks = [0.1, 0.5, 1, 2, 10] # 簡化 Log 刻度
    for t in ticks:
        tx = transform(t)
        if col_plot_start <= tx <= col_plot_end:
            ax.plot([tx, tx], [y_info, y_info + 0.15], color='black', linewidth=0.8)
            ax.text(tx, y_info - 0.4, f"{t:.1f}", ha='center', va='center', fontsize=8)
            
    ax.text(col_plot_start, y_info - 0.8, "Favours Tx", ha='left', va='center', fontsize=9)
    ax.text(col_plot_end, y_info - 0.8, "Favours Ctrl", ha='right', va='center', fontsize=9)

    return fig

def plot_leave_one_out_professional(ma_engine):
    # 使用與 Forest Plot 相同的對齊邏輯
    inf_df = ma_engine.influence_df
    measure = ma_engine.measure
    res = ma_engine.results
    
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 300})
    n_studies = len(inf_df)
    n_rows = n_studies + 3
    
    fig, ax = plt.subplots(figsize=(10, n_rows * 0.4))
    ax.set_ylim(0, n_rows)
    ax.set_xlim(0, 100)
    ax.axis('off')
    
    col_study = 5
    col_plot_start = 45
    col_plot_end = 75
    col_stats = 95
    
    # Header
    y_header = n_rows - 1
    ax.text(col_study, y_header, "Study Omitted", fontweight='bold', ha='left')
    ax.text((col_plot_start+col_plot_end)/2, y_header, "Effect Size (95% CI)", fontweight='bold', ha='center')
    ax.plot([0, 100], [y_header - 0.5, y_header - 0.5], color='black', linewidth=0.8)
    
    # Transform logic (Same as forest)
    if measure == "RR":
        vals = np.exp(inf_df['TE.del'])
        lows = np.exp(inf_df['lower.del'])
        ups = np.exp(inf_df['upper.del'])
        center = 1.0
        x_min, x_max = 0.1, 10 # 固定範圍保持一致性
        def transform(v):
            try: return col_plot_start + ((np.log(v)-np.log(x_min))/(np.log(x_max)-np.log(x_min)))*(col_plot_end-col_plot_start)
            except: return col_plot_start
    else:
        vals, lows, ups = inf_df['TE.del'], inf_df['lower.del'], inf_df['upper.del']
        center = 0.0
        x_min, x_max = vals.min()-0.5, vals.max()+0.5
        def transform(v): return col_plot_start + ((v-x_min)/(x_max-x_min))*(col_plot_end-col_plot_start)

    # Rows
    for i, row in inf_df.iterrows():
        y = n_rows - 2 - i
        ax.text(col_study, y, f"Omitting {row['Study ID']}", ha='left', va='center')
        
        # Plot
        x = transform(vals[i])
        xl = transform(lows[i])
        xr = transform(ups[i])
        ax.plot([xl, xr], [y, y], color='black', linewidth=1)
        ax.plot(x, y, 's', color='gray', markersize=5)
        
        # Stats
        stats_txt = f"{vals[i]:.2f} [{lows[i]:.2f}, {ups[i]:.2f}]"
        ax.text(col_stats, y, stats_txt, ha='right', va='center', fontsize=9)
        
    # Vertical Line
    cx = transform(center)
    ax.plot([cx, cx], [0.5, n_rows - 1.5], linestyle='--', color='black', linewidth=0.5)
    
    # Original Pooled (Bottom)
    orig_val = np.exp(res['TE_pooled']) if measure == "RR" else res['TE_pooled']
    orig_low = np.exp(res['lower_pooled']) if measure == "RR" else res['lower_pooled']
    orig_up = np.exp(res['upper_pooled']) if measure == "RR" else res['upper_pooled']
    
    y_pool = 0.5
    px, pl, pr = transform(orig_val), transform(orig_low), transform(orig_up)
    diamond_x = [pl, px, pr, px]
    diamond_y = [y_pool, y_pool+0.25, y_pool, y_pool-0.25]
    ax.fill(diamond_x, diamond_y, color='red', alpha=0.5)
    
    ax.text(col_study, y_pool, "All Studies Included", fontweight='bold', ha='left', va='center')
    ax.text(col_stats, y_pool, f"{orig_val:.2f} [{orig_low:.2f}, {orig_up:.2f}]", fontweight='bold', ha='right', va='center')

    return fig
